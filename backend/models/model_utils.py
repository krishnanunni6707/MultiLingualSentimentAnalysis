"""
models/model_utils.py

Utilities around the sentiment model: tokenizer wrapper and
model/checkpoint loading.
"""

import torch
from transformers import AutoTokenizer
from typing import Any, Dict, Tuple, Optional, List

from config import ModelConfig
from utils.logger import get_logger
from models.sentiment_model import MultiTaskSentimentModel

logger = get_logger(__name__)


class ModelTokenizer:
    """
    Wrapper for IndicBERT tokenizer
    """

    def __init__(self, model_name: str = ModelConfig.MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = ModelConfig.MAX_SEQUENCE_LENGTH

    def encode(
        self,
        text: str,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format (pt for PyTorch)

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_token_type_ids=False,
            return_tensors=return_tensors
        )
        return encoding

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode multiple texts at once"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
            **kwargs
        )

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )


def _default_emotion_labels() -> List[str]:
    return list(ModelConfig.DEFAULT_EMOTION_LABELS)


def _load_checkpoint_payload(checkpoint_path: str) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict) and "model_state_dict" in payload:
        return {
            "state_dict": payload["model_state_dict"],
            "emotion_labels": list(payload.get("emotion_labels") or []),
            "model_name": payload.get("model_name", ModelConfig.MODEL_NAME),
        }

    return {
        "state_dict": payload,
        "emotion_labels": [],
        "model_name": ModelConfig.MODEL_NAME,
    }


def load_model(
    checkpoint_path: Optional[str] = None,
    emotion_labels: Optional[List[str]] = None,
    model_name: str = ModelConfig.MODEL_NAME,
) -> Tuple[MultiTaskSentimentModel, ModelTokenizer, List[str]]:
    """
    Load model and tokenizer

    Args:
        checkpoint_path: Path to checkpoint (if None, loads pretrained)

        emotion_labels: Optional label vocabulary to use when a checkpoint
            does not provide one
        model_name: Base transformer model name

    Returns:
        Tuple of (model, tokenizer, emotion_labels)
    """
    resolved_emotion_labels = list(emotion_labels) if emotion_labels else _default_emotion_labels()
    resolved_model_name = model_name
    state_dict = None

    if checkpoint_path:
        try:
            checkpoint = _load_checkpoint_payload(checkpoint_path)
            state_dict = checkpoint["state_dict"]

            checkpoint_emotion_labels = checkpoint["emotion_labels"]
            if checkpoint_emotion_labels:
                resolved_emotion_labels = checkpoint_emotion_labels

            resolved_model_name = checkpoint.get("model_name") or resolved_model_name

            if emotion_labels and checkpoint_emotion_labels and emotion_labels != checkpoint_emotion_labels:
                logger.warning(
                    "Checkpoint emotion labels differ from provided labels; using checkpoint labels"
                )

            logger.info(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    model = MultiTaskSentimentModel(
        model_name=resolved_model_name,
        num_emotions=len(resolved_emotion_labels),
    )
    tokenizer = ModelTokenizer(model_name=resolved_model_name)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, tokenizer, resolved_emotion_labels