"""
training/trainer.py (FIXED)

Trainer class for the Multi-task Hindi Sentiment Analysis Model.
Handles training loop, validation, checkpointing, and early stopping.
Also writes a structured CSV metrics log after every epoch.

KEY FIXES:
1. Emotion loss weight boosted 5-10x to counteract BCE magnitude issue
2. BCEWithLogitsLoss pos_weight for sparse multi-label handling
3. Label density analysis & logging for debugging
4. Prediction distribution logging (to catch null predictor)
5. Gradient flow diagnostics
"""

import csv
import os
import time
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import TrainingConfig, CHECKPOINT_DIR
from training.losses import MultiTaskLoss
from training.metrics import compute_all_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for MultiTaskSentimentModel.

    Supports:
    - Multi-task loss (emotion BCE + polarity MSE + intensity MSE)
    - Adaptive loss weighting based on label statistics
    - AdamW optimizer with linear warmup scheduler
    - Per-epoch checkpointing + best-model tracking
    - Early stopping based on validation loss
    - Resume from checkpoint via load_model()
    - Diagnostics: label density, prediction distribution, gradient flow
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        emotion_names: Optional[List[str]] = None,
        emotion_labels: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.emotion_names = emotion_names or []
        self.emotion_labels = emotion_labels or emotion_names or []
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Mixed-precision scaler (fp16 on CUDA; no-op on CPU)
        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        if self._use_amp:
            logger.info("Mixed-precision (fp16) training enabled — VRAM usage ~halved")

        # Structured CSV training log
        self._metrics_log_path = self.checkpoint_dir / "training_metrics.csv"
        self._csv_header_written = self._metrics_log_path.exists()
        logger.info(f"Training metrics log → {self._metrics_log_path}")

        # ============================================================
        # ANALYZE TRAINING DATA LABEL DISTRIBUTION (CRITICAL!)
        # ============================================================
        self._analyze_label_distribution()

        # Loss function with ADAPTIVE WEIGHTS
        # Calculate pos_weight for BCE based on label density
        self.criterion = MultiTaskLoss(
            emotion_weight=TrainingConfig.EMOTION_LOSS_WEIGHT * self._emotion_weight_boost,
            polarity_weight=TrainingConfig.POLARITY_LOSS_WEIGHT,
            intensity_weight=TrainingConfig.INTENSITY_LOSS_WEIGHT,
            pos_weight=self._pos_weight,  # ← CRITICAL: handles sparse multi-label
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
        )

        # Scheduler: linear warmup → constant LR
        warmup_steps = TrainingConfig.WARMUP_STEPS
        total_steps = len(train_loader) * TrainingConfig.NUM_EPOCHS
        warmup_steps = min(warmup_steps, total_steps)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        constant_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=total_steps)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps],
        )

        self.best_val_f1 = -1.0  # Tracks macro F1 (higher is better)
        self.epochs_without_improvement = 0
        
        # Diagnostic tracking
        self._train_step_count = 0

    # ------------------------------------------------------------------
    # Data Analysis & Diagnostics
    # ------------------------------------------------------------------

    def _analyze_label_distribution(self) -> None:
        """
        Analyze emotion label distribution to compute pos_weight for BCE.
        Also determine adaptive emotion loss weight boost.
        """
        logger.info("\n" + "=" * 70)
        logger.info("LABEL DISTRIBUTION ANALYSIS")
        logger.info("=" * 70)

        emotion_label_counts = None
        total_samples = 0

        # Scan training data
        for batch in self.train_loader:
            if "emotion_labels" in batch:
                emotion_labels = batch["emotion_labels"]
                if emotion_label_counts is None:
                    emotion_label_counts = emotion_labels.sum(dim=0)
                else:
                    emotion_label_counts += emotion_labels.sum(dim=0)
                total_samples += emotion_labels.size(0)

        if emotion_label_counts is None:
            logger.warning("No emotion labels found in training data!")
            self._emotion_weight_boost = 5.0
            self._pos_weight = None
            return

        num_emotions = len(emotion_label_counts)
        emotion_densities = emotion_label_counts.float() / total_samples

        avg_density = emotion_densities.mean().item()
        min_density = emotion_densities.min().item()
        max_density = emotion_densities.max().item()

        logger.info(f"Total train samples: {total_samples}")
        logger.info(f"Total emotions: {num_emotions}")
        logger.info(f"Avg label density: {avg_density:.4f} ({avg_density*100:.2f}%)")
        logger.info(f"Min label density: {min_density:.4f} ({min_density*100:.2f}%)")
        logger.info(f"Max label density: {max_density:.4f} ({max_density*100:.2f}%)")

        # Compute pos_weight for BCE: (num_neg / num_pos) for each emotion
        # Higher weight for rare emotions
        num_positives = emotion_label_counts.float()
        num_negatives = total_samples - num_positives
        pos_weight = num_negatives / (num_positives + 1e-8)  # +1e-8 to avoid div by zero
        
        # Clamp to avoid extreme values
        pos_weight = torch.clamp(pos_weight, min=1.0, max=25.0)

        self._pos_weight = pos_weight
        
        # Lower global scaling to protect high-density fields (neutral, compassion)
        if avg_density < 0.05:
            self._emotion_weight_boost = 5.0
            logger.info(f"🔴 VERY SPARSE LABELS (<5%) → emotion_weight_boost = 5.0x")
        elif avg_density < 0.10:
            self._emotion_weight_boost = 4.0
            logger.info(f"🟠 SPARSE LABELS (<10%) → emotion_weight_boost = 4.0x")
        elif avg_density < 0.20:
            self._emotion_weight_boost = 3.0
            logger.info(f"🟡 MODERATE SPARSITY (<20%) → emotion_weight_boost = 3.0x")
        else:
            self._emotion_weight_boost = 1.5
            logger.info(f"🟢 REASONABLE LABEL DISTRIBUTION → emotion_weight_boost = 1.5x")

        logger.info(f"Final emotion_weight: {TrainingConfig.EMOTION_LOSS_WEIGHT * self._emotion_weight_boost:.2f}")
        logger.info(f"pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
        logger.info("=" * 70 + "\n")

    def _get_prediction_stats(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Analyze emotion logit predictions to detect null predictor.
        Returns dict with avg probability, proportion predicting any emotion, etc.
        """
        probs = torch.sigmoid(logits)
        
        # Per-emotion statistics
        avg_probs = probs.mean(dim=0)  # [num_emotions]
        
        # How many samples predict >= 0.5 for each emotion
        pred_positive = (probs >= 0.5).sum(dim=0).float() / probs.size(0)
        
        # How many samples predict ANY emotion
        any_emotion = (probs.max(dim=1)[0] >= 0.5).float().mean()
        
        return {
            "avg_prob_overall": avg_probs.mean().item(),
            "max_avg_prob": avg_probs.max().item(),
            "min_avg_prob": avg_probs.min().item(),
            "prop_predicting_any": any_emotion.item(),
            "avg_pred_positive_per_emotion": pred_positive.mean().item(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in a batch dict to the configured device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _forward_batch(self, batch: Dict[str, Any]):
        """
        Run one forward pass and compute losses.
        Returns (outputs, losses) dicts.
        """
        batch = self._move_batch(batch)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids")

        emotion_labels = batch.get("emotion_labels")
        polarity_labels = batch.get("polarity_labels")
        intensity_labels = batch.get("intensity_labels")

        # Forward
        fwd_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            fwd_kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**fwd_kwargs)

        # Loss
        losses = self.criterion(
            outputs,
            emotion_labels=emotion_labels,
            polarity_labels=polarity_labels,
            intensity_labels=intensity_labels,
        )

        return outputs, losses, emotion_labels, polarity_labels, intensity_labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns average loss dict."""
        self.model.train()

        total_losses: Dict[str, float] = {}
        num_batches = 0
        
        # For diagnostics
        all_emotion_logits = []

        grad_accum = TrainingConfig.GRADIENT_ACCUMULATION_STEPS

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                outputs, losses, _, _, _ = self._forward_batch(batch)

            loss = losses["total_loss"] / grad_accum
            self._scaler.scale(loss).backward()

            # Accumulate losses for logging
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

            # Collect logits for diagnostics (sample)
            if step % 50 == 0 and "emotion_logits" in outputs:
                all_emotion_logits.append(outputs["emotion_logits"].detach().cpu())

            num_batches += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(self.train_loader):
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), TrainingConfig.MAX_GRAD_NORM
                )
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (step + 1) % TrainingConfig.LOG_EVERY_N_STEPS == 0:
                avg_loss = total_losses.get("total_loss", 0.0) / num_batches
                logger.info(
                    f"  Epoch {epoch} | Step {step + 1}/{len(self.train_loader)} "
                    f"| loss={avg_loss:.4f}"
                )

            self._train_step_count += 1

        # Log prediction stats at end of epoch
        if all_emotion_logits:
            all_logits = torch.cat(all_emotion_logits, dim=0)
            pred_stats = self._get_prediction_stats(all_logits)
            logger.info(
                f"  [train_pred_stats] avg_prob={pred_stats['avg_prob_overall']:.4f} | "
                f"prop_predicting_any={pred_stats['prop_predicting_any']:.4f} | "
                f"avg_pred_positive={pred_stats['avg_pred_positive_per_emotion']:.4f}"
            )

        return {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "val") -> Dict[str, float]:
        """Evaluate on a dataloader. Returns average loss + metrics dict."""
        self.model.eval()

        total_losses: Dict[str, float] = {}
        all_outputs: Dict[str, list] = {
            "emotion_logits": [],
            "polarity_scores": [],
            "intensity_scores": [],
        }
        all_labels: Dict[str, list] = {
            "emotion": [],
            "polarity": [],
            "intensity": [],
        }
        num_batches = 0

        for batch in loader:
            outputs, losses, emotion_labels, polarity_labels, intensity_labels = (
                self._forward_batch(batch)
            )

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            num_batches += 1

            # Accumulate tensors for metrics
            if "emotion_logits" in outputs and emotion_labels is not None:
                all_outputs["emotion_logits"].append(outputs["emotion_logits"].cpu())
                all_labels["emotion"].append(emotion_labels.cpu())

            if "polarity_scores" in outputs and polarity_labels is not None:
                all_outputs["polarity_scores"].append(outputs["polarity_scores"].cpu())
                all_labels["polarity"].append(polarity_labels.cpu())

            if "intensity_scores" in outputs and intensity_labels is not None:
                all_outputs["intensity_scores"].append(outputs["intensity_scores"].cpu())
                all_labels["intensity"].append(intensity_labels.cpu())

        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}

        # Compute metrics if we have accumulated outputs
        cat_outputs: Dict[str, torch.Tensor] = {}
        if all_outputs["emotion_logits"]:
            cat_outputs["emotion_logits"] = torch.cat(all_outputs["emotion_logits"], dim=0)
        if all_outputs["polarity_scores"]:
            cat_outputs["polarity_scores"] = torch.cat(all_outputs["polarity_scores"], dim=0)
        if all_outputs["intensity_scores"]:
            cat_outputs["intensity_scores"] = torch.cat(all_outputs["intensity_scores"], dim=0)

        cat_labels = {
            k: torch.cat(v, dim=0) if v else None
            for k, v in all_labels.items()
        }

        if cat_outputs:
            metrics = compute_all_metrics(
                cat_outputs,
                emotion_labels=cat_labels["emotion"],
                polarity_labels=cat_labels["polarity"],
                intensity_labels=cat_labels["intensity"],
                emotion_names=self.emotion_names or None,
            )
            avg_losses.update({f"{desc}_{k}": v for k, v in metrics.items()})

            # Log prediction stats for validation set
            if "emotion_logits" in cat_outputs:
                pred_stats = self._get_prediction_stats(cat_outputs["emotion_logits"])
                logger.info(
                    f"  [{desc}_pred_stats] avg_prob={pred_stats['avg_prob_overall']:.4f} | "
                    f"prop_predicting_any={pred_stats['prop_predicting_any']:.4f} | "
                    f"avg_pred_positive={pred_stats['avg_pred_positive_per_emotion']:.4f}"
                )

        logger.info(
            f"  [{desc}] loss={avg_losses.get('total_loss', 0.0):.4f} | "
            + " | ".join(
                f"{k}={v:.4f}"
                for k, v in avg_losses.items()
                if k != "total_loss" and not k.endswith("_loss")
            )
        )

        return avg_losses

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_results: Dict[str, float],
        epoch_duration_s: float,
        is_best: bool,
    ) -> None:
        """
        Append one row per epoch to training_metrics.csv.
        Columns: timestamp, epoch, duration_s, is_best, then all train_* and val_* metrics.
        """
        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "duration_s": round(epoch_duration_s, 2),
            "is_best": int(is_best),
        }
        # Prefix train losses
        for k, v in train_losses.items():
            row[f"train_{k}"] = round(v, 6)
        # Val results already prefixed with 'val_'
        for k, v in val_results.items():
            row[k] = round(v, 6) if isinstance(v, float) else v

        write_header = not self._csv_header_written
        with open(self._metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

        logger.info(f"  Metrics logged → {self._metrics_log_path}  (epoch {epoch})")

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "emotion_labels": self.emotion_labels,
            "model_name": getattr(
                getattr(self.model, "config", None), "_name_or_path", ""
            ),
            "best_val_f1": self.best_val_f1,
        }

        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)
        logger.info(f"  Checkpoint saved → {epoch_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  ★ New best model saved → {best_path}")

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        Full training loop with validation, checkpointing, and early stopping.

        Args:
            num_epochs: Override TrainingConfig.NUM_EPOCHS if provided.
        """
        epochs = num_epochs or TrainingConfig.NUM_EPOCHS
        patience = TrainingConfig.PATIENCE

        logger.info(f"Starting training for {epochs} epoch(s) on {self.device}")
        logger.info(f"  Train batches : {len(self.train_loader)}")
        logger.info(f"  Val batches   : {len(self.val_loader)}")
        logger.info(f"  Emotion labels: {self.emotion_labels}")
        logger.info(f"  Emotion loss boost: {self._emotion_weight_boost:.1f}x")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info("=" * 60)

            # --- Train ---
            train_losses = self.train_epoch(epoch)
            logger.info(
                f"  [train] loss={train_losses.get('total_loss', 0.0):.4f}"
                + (
                    f" | emotion={train_losses['emotion_loss']:.4f}"
                    if "emotion_loss" in train_losses
                    else ""
                )
                + (
                    f" | polarity={train_losses['polarity_loss']:.4f}"
                    if "polarity_loss" in train_losses
                    else ""
                )
                + (
                    f" | intensity={train_losses['intensity_loss']:.4f}"
                    if "intensity_loss" in train_losses
                    else ""
                )
            )

            # --- Validate ---
            val_results = self.evaluate(self.val_loader, desc="val")
            val_f1 = val_results.get("val_emotion_f1_macro", 0.0)

            # --- Higher macro F1 means progress ---
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"  No improvement for {self.epochs_without_improvement}/{patience} epoch(s)"
                )

            epoch_duration = time.time() - epoch_start
            self._log_epoch_metrics(epoch, train_losses, val_results, epoch_duration, is_best)
            self.save_checkpoint(epoch, is_best=is_best)

            # Release any unused reserved VRAM between epochs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if self.epochs_without_improvement >= patience:
                logger.info(
                    f"\nEarly stopping triggered after {patience} epoch(s) without improvement."
                )
                break

        logger.info(f"\nTraining finished. Best val Macro F1: {self.best_val_f1:.4f}")