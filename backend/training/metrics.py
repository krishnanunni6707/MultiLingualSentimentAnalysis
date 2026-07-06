"""
training/metrics.py

Evaluation metrics for the three sentiment tasks: emotion (multi-label
classification), polarity (regression), and intensity (regression).
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None

from utils.logger import get_logger

logger = get_logger(__name__)

if pearsonr is None:
    logger.warning("scipy not installed; Pearson correlation metrics will be reported as 0.0. "
                    "Install with: pip install scipy")


def compute_emotion_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    emotion_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multi-label emotion classification metrics.

    Args:
        logits: [N, num_emotions] raw logits (pre-sigmoid)
        labels: [N, num_emotions] binary/soft ground-truth labels
        threshold: Probability threshold for binarizing predictions/labels
        emotion_names: Optional list of emotion names for per-class F1

    Returns:
        Dict with micro/macro F1, precision, recall, subset accuracy, and
        per-class F1 (if emotion_names is given)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true = (labels.detach().cpu().numpy() >= threshold).astype(int)

    metrics = {
        "emotion_f1_micro": float(f1_score(true, preds, average="micro", zero_division=0)),
        "emotion_f1_macro": float(f1_score(true, preds, average="macro", zero_division=0)),
        "emotion_precision_micro": float(precision_score(true, preds, average="micro", zero_division=0)),
        "emotion_recall_micro": float(recall_score(true, preds, average="micro", zero_division=0)),
        "emotion_subset_accuracy": float(accuracy_score(true, preds)),
    }

    if emotion_names:
        per_class_f1 = f1_score(true, preds, average=None, zero_division=0)
        for name, score in zip(emotion_names, per_class_f1):
            metrics[f"emotion_f1_{name}"] = float(score)

    return metrics


def compute_regression_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    """
    Compute regression metrics (MSE, MAE, Pearson correlation) for a single
    continuous target (polarity or intensity).

    Args:
        preds: [N, 1] or [N] predicted values
        labels: [N, 1] or [N] ground-truth values
        prefix: Metric key prefix, e.g. "polarity" or "intensity"

    Returns:
        Dict of metric name -> value
    """
    preds_np = preds.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)

    metrics = {
        f"{prefix}_mse": float(mean_squared_error(labels_np, preds_np)),
        f"{prefix}_mae": float(mean_absolute_error(labels_np, preds_np)),
    }

    if pearsonr is not None and len(preds_np) > 1 and np.std(preds_np) > 0 and np.std(labels_np) > 0:
        corr, _ = pearsonr(preds_np, labels_np)
        metrics[f"{prefix}_pearson"] = float(corr)
    else:
        metrics[f"{prefix}_pearson"] = 0.0

    return metrics


def compute_all_metrics(
    outputs: Dict[str, torch.Tensor],
    emotion_labels: Optional[torch.Tensor] = None,
    polarity_labels: Optional[torch.Tensor] = None,
    intensity_labels: Optional[torch.Tensor] = None,
    emotion_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper computing metrics for whichever tasks have labels
    available, from a single model output dict (as produced by
    MultiTaskSentimentModel.forward).
    """
    metrics: Dict[str, float] = {}

    if emotion_labels is not None:
        metrics.update(
            compute_emotion_metrics(
                outputs["emotion_logits"], emotion_labels, emotion_names=emotion_names
            )
        )

    if polarity_labels is not None:
        metrics.update(
            compute_regression_metrics(outputs["polarity_scores"], polarity_labels, prefix="polarity")
        )

    if intensity_labels is not None:
        metrics.update(
            compute_regression_metrics(outputs["intensity_scores"], intensity_labels, prefix="intensity")
        )

    return metrics