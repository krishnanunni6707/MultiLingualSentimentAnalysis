"""
training/losses.py

Multi-task loss functions for the Hindi sentiment model.
Decoupled from the model itself so the loss logic can be reused, tuned,
or swapped (e.g. class-weighted BCE for imbalanced emotions) without
touching model architecture code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from config import TrainingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Combines emotion (multi-label BCE), polarity (MSE), and intensity (MSE)
    losses into a single weighted total loss.
    """

    def __init__(
        self,
        emotion_weight: float = TrainingConfig.EMOTION_LOSS_WEIGHT,
        polarity_weight: float = TrainingConfig.POLARITY_LOSS_WEIGHT,
        intensity_weight: float = TrainingConfig.INTENSITY_LOSS_WEIGHT,
        emotion_pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            emotion_weight: Weight applied to the emotion loss term
            polarity_weight: Weight applied to the polarity loss term
            intensity_weight: Weight applied to the intensity loss term
            emotion_pos_weight: Optional per-class positive weight for BCE
                (shape: [num_emotions]), useful if emotion labels are
                imbalanced (e.g. rare emotions under-predicted)
        """
        super().__init__()
        self.emotion_weight = emotion_weight
        self.polarity_weight = polarity_weight
        self.intensity_weight = intensity_weight
        self.emotion_pos_weight = emotion_pos_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        emotion_labels: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict containing "emotion_logits",
                "polarity_scores", "intensity_scores"
            emotion_labels: [batch_size, num_emotions] binary/soft labels
            polarity_labels: [batch_size, 1]
            intensity_labels: [batch_size, 1]

        Returns:
            Dict with individual task losses plus "total_loss". Only tasks
            with labels provided contribute to total_loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = outputs["emotion_logits"].device
        total_loss = torch.zeros((), device=device)

        if emotion_labels is not None:
            pos_weight = (
                self.emotion_pos_weight.to(device)
                if self.emotion_pos_weight is not None
                else None
            )
            emotion_loss = F.binary_cross_entropy_with_logits(
                outputs["emotion_logits"],
                emotion_labels.float(),
                pos_weight=pos_weight,
            )
            losses["emotion_loss"] = emotion_loss
            total_loss = total_loss + emotion_loss * self.emotion_weight

        if polarity_labels is not None:
            polarity_loss = F.mse_loss(outputs["polarity_scores"], polarity_labels.float())
            losses["polarity_loss"] = polarity_loss
            total_loss = total_loss + polarity_loss * self.polarity_weight

        if intensity_labels is not None:
            intensity_loss = F.mse_loss(outputs["intensity_scores"], intensity_labels.float())
            losses["intensity_loss"] = intensity_loss
            total_loss = total_loss + intensity_loss * self.intensity_weight

        losses["total_loss"] = total_loss
        return losses