"""
training/losses.py (FIXED - Device Handling)

Multi-task loss function for emotion + polarity + intensity prediction.

KEY FIXES:
1. BCEWithLogitsLoss pos_weight parameter for sparse multi-label
2. Proper loss weighting to prevent polarity/intensity from dominating
3. DEVICE HANDLING: pos_weight moved to same device as inputs
4. Detailed loss logging for debugging
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    - Emotion: Multi-label classification (BCEWithLogitsLoss)
    - Polarity: Regression (MSE)
    - Intensity: Regression (MSE)
    
    Args:
        emotion_weight: Weight for emotion loss
        polarity_weight: Weight for polarity loss
        intensity_weight: Weight for intensity loss
        pos_weight: Per-emotion positive class weights for BCE (tensor or None)
                   Higher values boost gradient for rare emotions
    """

    def __init__(
        self,
        emotion_weight: float = 1.0,
        polarity_weight: float = 0.8,
        intensity_weight: float = 0.8,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.polarity_weight = polarity_weight
        self.intensity_weight = intensity_weight
        
        # Store pos_weight as buffer so it moves to device with model
        # CRITICAL: We'll move it to the correct device in forward()
        if pos_weight is not None:
            self.register_buffer("pos_weight_buffer", pos_weight, persistent=False)
        else:
            self.register_buffer("pos_weight_buffer", torch.tensor([]), persistent=False)
        
        # Create a base BCE loss without pos_weight (for fallback if pos_weight is None)
        self.bce_loss_base = nn.BCEWithLogitsLoss(reduction="mean")
        
        # MSE loss for regression tasks
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        emotion_labels: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Dict with keys 'emotion_logits', 'polarity_scores', 'intensity_scores'
            emotion_labels: [batch_size, num_emotions] binary labels
            polarity_labels: [batch_size] continuous labels in [-1, 1] or [0, 1]
            intensity_labels: [batch_size] continuous labels in [0, 1]
            
        Returns:
            Dict with keys: 'emotion_loss', 'polarity_loss', 'intensity_loss', 'total_loss'
        """
        losses = {}
        total_loss = 0.0

        # ============================================================
        # Emotion Loss (Multi-label BCE with pos_weight)
        # ============================================================
        if emotion_labels is not None and "emotion_logits" in outputs:
            emotion_logits = outputs["emotion_logits"]
            
            # Ensure labels are float for BCE
            emotion_labels_float = emotion_labels.float()
            
            # CRITICAL FIX: Handle pos_weight device mismatch
            # pos_weight may be on CPU while logits are on CUDA
            if self.pos_weight_buffer is not None and self.pos_weight_buffer.numel() > 0:
                # Move pos_weight to same device as emotion_logits
                pos_weight_device = self.pos_weight_buffer.to(emotion_logits.device)
                
                # Create BCE loss with pos_weight on correct device
                bce_loss_fn = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight_device,
                    reduction="mean"
                )
                emotion_loss = bce_loss_fn(emotion_logits, emotion_labels_float)
            else:
                # Fallback: use base BCE without pos_weight
                emotion_loss = self.bce_loss_base(emotion_logits, emotion_labels_float)
            
            emotion_loss_weighted = self.emotion_weight * emotion_loss
            
            losses["emotion_loss"] = emotion_loss
            total_loss = total_loss + emotion_loss_weighted
        else:
            losses["emotion_loss"] = torch.tensor(0.0, device=emotion_logits.device if emotion_labels is not None else 'cpu')

        # ============================================================
        # Polarity Loss (Regression MSE)
        # ============================================================
        if polarity_labels is not None and "polarity_scores" in outputs:
            polarity_scores = outputs["polarity_scores"].squeeze(-1)
            
            # Handle case where polarity_scores might be [batch, 1]
            if polarity_scores.dim() == 1:
                pass
            elif polarity_scores.dim() == 2 and polarity_scores.size(1) == 1:
                polarity_scores = polarity_scores.squeeze(-1)
            
            # Ensure labels are float for MSE
            polarity_labels_float = polarity_labels.float()
            
            # Clamp predictions to reasonable range to avoid NaN
            polarity_scores = torch.clamp(polarity_scores, min=-2.0, max=2.0)
            
            polarity_loss = self.mse_loss(polarity_scores, polarity_labels_float)
            polarity_loss_weighted = self.polarity_weight * polarity_loss
            
            losses["polarity_loss"] = polarity_loss
            total_loss = total_loss + polarity_loss_weighted
        else:
            losses["polarity_loss"] = torch.tensor(0.0)

        # ============================================================
        # Intensity Loss (Regression MSE)
        # ============================================================
        if intensity_labels is not None and "intensity_scores" in outputs:
            intensity_scores = outputs["intensity_scores"].squeeze(-1)
            
            # Handle case where intensity_scores might be [batch, 1]
            if intensity_scores.dim() == 1:
                pass
            elif intensity_scores.dim() == 2 and intensity_scores.size(1) == 1:
                intensity_scores = intensity_scores.squeeze(-1)
            
            # Ensure labels are float for MSE
            intensity_labels_float = intensity_labels.float()
            
            # Clamp predictions to reasonable range to avoid NaN
            intensity_scores = torch.clamp(intensity_scores, min=0.0, max=1.0)
            
            intensity_loss = self.mse_loss(intensity_scores, intensity_labels_float)
            intensity_loss_weighted = self.intensity_weight * intensity_loss
            
            losses["intensity_loss"] = intensity_loss
            total_loss = total_loss + intensity_loss_weighted
        else:
            losses["intensity_loss"] = torch.tensor(0.0)

        # ============================================================
        # Total Loss
        # ============================================================
        losses["total_loss"] = total_loss

        return losses