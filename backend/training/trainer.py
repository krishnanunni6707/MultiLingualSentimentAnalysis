"""
training/trainer.py

Training loop, validation loop, checkpointing, and early stopping for the
multi-task Hindi sentiment model.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import TrainingConfig, CHECKPOINT_DIR
from models.sentiment_model import MultiTaskSentimentModel
from training.losses import MultiTaskLoss
from training.metrics import compute_all_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Handles the full training loop for MultiTaskSentimentModel: optimizer
    and scheduler setup, epoch/step loops, validation, checkpointing, and
    early stopping based on validation total_loss.
    """

    def __init__(
        self,
        model: MultiTaskSentimentModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        emotion_names: Optional[List[str]] = None,
        emotion_labels: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = CHECKPOINT_DIR,
    ):
        """
        Args:
            model: The multi-task model to train
            train_loader: Training DataLoader (yields dicts with input_ids,
                attention_mask, emotion_labels, polarity_labels, intensity_labels)
            val_loader: Validation DataLoader, same format
            emotion_names: Optional emotion label names, for per-class metrics
            emotion_labels: Optional vocabulary order to persist in checkpoints
            device: Device to train on (defaults to CUDA if available, else CPU)
            checkpoint_dir: Directory to save checkpoints to
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.emotion_names = emotion_names
        self.emotion_labels = emotion_labels or emotion_names or []

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = MultiTaskLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
        )

        total_steps = max(1, len(self.train_loader) * TrainingConfig.NUM_EPOCHS)
        warmup_steps = min(TrainingConfig.WARMUP_STEPS, total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return average losses across batches."""
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        for batch in progress_bar:
            batch = self._move_batch_to_device(batch)

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            losses = self.loss_fn(
                outputs,
                emotion_labels=batch["emotion_labels"],
                polarity_labels=batch["polarity_labels"],
                intensity_labels=batch["intensity_labels"],
            )

            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TrainingConfig.MAX_GRAD_NORM
            )
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            num_batches += 1

            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value.item()

            progress_bar.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})

        return {key: value / num_batches for key, value in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "eval") -> Dict[str, float]:
        """
        Run evaluation over a data loader.

        Returns:
            Dict combining average losses (e.g. "total_loss") and task
            metrics (e.g. "emotion_f1_macro", "polarity_pearson", ...)
        """
        self.model.eval()
        epoch_losses: Dict[str, float] = {}
        all_outputs = {"emotion_logits": [], "polarity_scores": [], "intensity_scores": []}
        all_labels = {"emotion_labels": [], "polarity_labels": [], "intensity_labels": []}
        num_batches = 0

        for batch in tqdm(data_loader, desc=desc):
            batch = self._move_batch_to_device(batch)

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            losses = self.loss_fn(
                outputs,
                emotion_labels=batch["emotion_labels"],
                polarity_labels=batch["polarity_labels"],
                intensity_labels=batch["intensity_labels"],
            )

            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value.item()
            num_batches += 1

            all_outputs["emotion_logits"].append(outputs["emotion_logits"].cpu())
            all_outputs["polarity_scores"].append(outputs["polarity_scores"].cpu())
            all_outputs["intensity_scores"].append(outputs["intensity_scores"].cpu())
            all_labels["emotion_labels"].append(batch["emotion_labels"].cpu())
            all_labels["polarity_labels"].append(batch["polarity_labels"].cpu())
            all_labels["intensity_labels"].append(batch["intensity_labels"].cpu())

        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        concatenated_outputs = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
        concatenated_labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}

        metrics = compute_all_metrics(
            concatenated_outputs,
            emotion_labels=concatenated_labels["emotion_labels"],
            polarity_labels=concatenated_labels["polarity_labels"],
            intensity_labels=concatenated_labels["intensity_labels"],
            emotion_names=self.emotion_names,
        )

        return {**avg_losses, **metrics}

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save the model's state dict, additionally tagging the best checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_payload = {
            "model_state_dict": self.model.state_dict(),
            "emotion_labels": self.emotion_labels,
            "num_emotions": self.model.num_emotions,
            "model_name": getattr(self.model.encoder.config, "name_or_path", None),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_payload, best_path)
            logger.info(f"Saved new best model: {best_path}")

    def train(self, num_epochs: int = TrainingConfig.NUM_EPOCHS) -> None:
        """
        Run the full training loop: for each epoch, train then validate,
        save a checkpoint (marking the best one so far), and stop early if
        validation total_loss hasn't improved for TrainingConfig.EARLY_STOPPING_PATIENCE
        epochs in a row.
        """
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, desc=f"Epoch {epoch} [val]")

            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {elapsed:.1f}s")
            logger.info(f"Train: {train_metrics}")
            logger.info(f"Val:   {val_metrics}")

            val_loss = val_metrics["total_loss"]
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.epochs_without_improvement >= TrainingConfig.PATIENCE:
                logger.info(
                    f"Early stopping triggered after epoch {epoch} "
                    f"({self.epochs_without_improvement} epochs without improvement)"
                )
                break