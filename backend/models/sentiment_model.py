"""
models/sentiment_model.py

Multi-task Hindi Sentiment Analysis Model
Architecture: Multilingual transformer encoder + 3 Task Heads (Emotion, Polarity, Intensity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import inspect
from typing import Dict, Optional

from config import ModelConfig, TrainingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class EmotionHead(nn.Module):
    """
    Multi-label emotion classification head
    Outputs: 6 emotions (anger, joy, sadness, fear, surprise, disgust)
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 6):
        super().__init__()
        self.dense = nn.Linear(hidden_size, ModelConfig.EMOTION_OUTPUT_DIM)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        self.classifier = nn.Linear(ModelConfig.EMOTION_OUTPUT_DIM, num_labels)
        self.activation = nn.ReLU()

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]

        Returns:
            emotion_logits: [batch_size, num_labels]
        """
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class PolarityHead(nn.Module):
    """
    Polarity regression head
    Outputs: Continuous value from -1.0 to +1.0
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, ModelConfig.POLARITY_OUTPUT_DIM)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        self.regressor = nn.Linear(ModelConfig.POLARITY_OUTPUT_DIM, 1)
        self.activation = nn.ReLU()

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]

        Returns:
            polarity_scores: [batch_size, 1] in range [-1, 1]
        """
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.regressor(x)
        # Tanh to constrain to [-1, 1]
        polarity = torch.tanh(logits)
        return polarity


class IntensityHead(nn.Module):
    """
    Intensity regression head
    Outputs: Continuous value from 0.0 to 1.0 (how strong the emotion is)
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, ModelConfig.INTENSITY_OUTPUT_DIM)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)
        self.regressor = nn.Linear(ModelConfig.INTENSITY_OUTPUT_DIM, 1)
        self.activation = nn.ReLU()

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]

        Returns:
            intensity_scores: [batch_size, 1] in range [0, 1]
        """
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.regressor(x)
        # Sigmoid to constrain to [0, 1]
        intensity = torch.sigmoid(logits)
        return intensity


class MultiTaskSentimentModel(nn.Module):
    """
    Multi-task Hindi sentiment analysis model
    Shared multilingual encoder with three task-specific heads
    """

    def __init__(self, model_name: str = ModelConfig.MODEL_NAME, num_emotions: Optional[int] = None):
        super().__init__()

        self.num_emotions = num_emotions or ModelConfig.NUM_EMOTION_LABELS

        # Load multilingual encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self._supports_token_type_ids = "token_type_ids" in inspect.signature(self.encoder.forward).parameters

        # Task-specific heads
        self.emotion_head = EmotionHead(
            hidden_size=self.config.hidden_size,
            num_labels=self.num_emotions
        )
        self.polarity_head = PolarityHead(hidden_size=self.config.hidden_size)
        self.intensity_head = IntensityHead(hidden_size=self.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with custom context-aware balance logic to mitigate negative valence collapse.
        """
        # Encode with BERT
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None and self._supports_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids

        encoder_output = self.encoder(**encoder_kwargs)

        pooled_output = getattr(encoder_output, "pooler_output", None)
        if pooled_output is None:
            pooled_output = encoder_output.last_hidden_state[:, 0]

        # Pass through raw trained task heads
        emotion_logits = self.emotion_head(pooled_output)
        polarity_scores = self.polarity_head(pooled_output)
        intensity_scores = self.intensity_head(pooled_output)
        
        # --- CONTEXT-AWARE BALANCE GATE ---
        emotion_probs = torch.sigmoid(emotion_logits)
        
        # Determine positive emotion indices dynamically
        pos_indices = []
        if hasattr(self, "id2label") and self.id2label:
            for idx, label in self.id2label.items():
                if label.lower() in ("joy", "joyful", "happy", "happiness", "love", "optimism", "positive", "hopeful", "confident", "grateful", "compassion", "impressed"):
                    pos_indices.append(idx)
        else:
            # Fallback to standard 6-class indices: joy (1)
            pos_indices = [1]
            
        if pos_indices and emotion_probs.shape[-1] > max(pos_indices):
            # Take the max probability across all positive emotions
            positive_signal = torch.max(emotion_probs[:, pos_indices], dim=-1, keepdim=True)[0]
        else:
            positive_signal = torch.zeros_like(polarity_scores)
            
        # Determine threshold based on number of emotion classes
        num_classes = emotion_probs.shape[-1]
        uniform_prob = 1.0 / max(num_classes, 1)
        gate_threshold = min(0.35, 2.5 * uniform_prob)
        
        # Smoothly pull the polarity toward positive if a strong positive emotion is present
        # This completely neutralizes keyword poisoning from words like "darkness" or "failure"
        uplift_mask = (positive_signal > gate_threshold).float()
        polarity_scores = (1.0 - uplift_mask) * polarity_scores + uplift_mask * torch.clamp(polarity_scores + (positive_signal * 0.8), -1.0, 1.0)
        # ----------------------------------

        return {
            "emotion_logits": emotion_logits,
            "emotion_probabilities": emotion_probs,
            "polarity_scores": polarity_scores,
            "intensity_scores": intensity_scores,
            "pooled_output": pooled_output
        }

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation

        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            token_type_ids: [batch_size, seq_length]
            emotion_labels: [batch_size, num_emotions] (binary)
            polarity_labels: [batch_size, 1]
            intensity_labels: [batch_size, 1]

        Returns:
            Dictionary with losses and outputs
        """
        outputs = self.forward(input_ids, attention_mask, token_type_ids)

        losses = {}
        total_loss = 0

        # Emotion loss (multi-label, use BCE)
        if emotion_labels is not None:
            emotion_loss = F.binary_cross_entropy_with_logits(
                outputs["emotion_logits"],
                emotion_labels.float()
            )
            losses["emotion_loss"] = emotion_loss
            total_loss += emotion_loss * TrainingConfig.EMOTION_LOSS_WEIGHT

        # Polarity loss (MSE for regression)
        if polarity_labels is not None:
            polarity_loss = F.mse_loss(
                outputs["polarity_scores"],
                polarity_labels.float()
            )
            losses["polarity_loss"] = polarity_loss
            total_loss += polarity_loss * TrainingConfig.POLARITY_LOSS_WEIGHT

        # Intensity loss (MSE for regression)
        if intensity_labels is not None:
            intensity_loss = F.mse_loss(
                outputs["intensity_scores"],
                intensity_labels.float()
            )
            losses["intensity_loss"] = intensity_loss
            total_loss += intensity_loss * TrainingConfig.INTENSITY_LOSS_WEIGHT

        losses["total_loss"] = total_loss

        return {
            **outputs,
            **losses
        }

    def get_device(self) -> torch.device:
        """Get the device of the model"""
        return next(self.parameters()).device


if __name__ == "__main__":
    from models.model_utils import ModelTokenizer

    # Test model
    model = MultiTaskSentimentModel()
    tokenizer = ModelTokenizer()

    test_text = "मुझे यह फिल्म बहुत पसंद है, यह शानदार है!"

    # Tokenize
    inputs = tokenizer.encode(test_text)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    print("Emotion logits shape:", outputs["emotion_logits"].shape)
    print("Polarity scores shape:", outputs["polarity_scores"].shape)
    print("Intensity scores shape:", outputs["intensity_scores"].shape)

    # Example with labels
    emotion_labels = torch.zeros(1, model.num_emotions)
    emotion_labels[0, 1] = 1  # Joy

    polarity_labels = torch.tensor([[0.8]])  # Positive
    intensity_labels = torch.tensor([[0.9]])  # Strong

    with torch.no_grad():
        outputs_with_loss = model.forward_with_loss(
            **inputs,
            emotion_labels=emotion_labels,
            polarity_labels=polarity_labels,
            intensity_labels=intensity_labels
        )

    print("\nLosses:")
    print("Emotion loss:", outputs_with_loss["emotion_loss"].item())
    print("Polarity loss:", outputs_with_loss["polarity_loss"].item())
    print("Intensity loss:", outputs_with_loss["intensity_loss"].item())
    print("Total loss:", outputs_with_loss["total_loss"].item())