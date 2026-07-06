"""
training package

Exposes the Trainer, multi-task loss module, and evaluation metrics.
"""

from training.trainer import Trainer
from training.losses import MultiTaskLoss
from training.metrics import (
    compute_emotion_metrics,
    compute_regression_metrics,
    compute_all_metrics,
)

__all__ = [
    "Trainer",
    "MultiTaskLoss",
    "compute_emotion_metrics",
    "compute_regression_metrics",
    "compute_all_metrics",
]