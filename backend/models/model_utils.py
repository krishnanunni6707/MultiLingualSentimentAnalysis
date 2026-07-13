"""
models/model_utils.py

Utilities around the sentiment model: tokenizer wrapper and
model/checkpoint loading.
"""

import torch
import os
from transformers import AutoTokenizer
from typing import Any, Dict, Tuple, Optional, List

from config import ModelConfig
from utils.logger import get_logger
from models.sentiment_model import MultiTaskSentimentModel

logger = get_logger(__name__)


import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class AutoSequenceClassifierWrapper(nn.Module):
    """
    Wrapper for sequence classification models to conform to the MultiTaskSentimentModel API.
    """
    def __init__(self, model_name_or_path: str, classifier: Optional[nn.Module] = None, temperature: float = 1.0):
        super().__init__()
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)
        self.config = self.classifier.config
        self.id2label = self.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        emotion_probabilities = torch.softmax(logits, dim=-1)
        
        # Determine polarity mapping dynamically
        polarity_weights = []
        for idx in range(len(self.id2label)):
            lbl = self.id2label[idx].lower()
            #=======Update the emotions according to the trained ones...
            if lbl in ("joy", "joyful", "happy", "happiness", "love", "optimism", "positive", "hopeful", "confident", "grateful", "compassion", "impressed"):
                polarity_weights.append(1.0)
            elif lbl in ("anger", "angry", "annoyed", "sadness", "sad", "fear", "fearful", "scared", "disgust", "disgusted", "negative", "hate", "guilty", "apprehensive"):
                polarity_weights.append(-1.0)
            else:
                polarity_weights.append(0.0)
                
        device = logits.device
        polarity_weights_t = torch.tensor(polarity_weights, device=device, dtype=torch.float)
        
        polarity_scores = torch.sum(emotion_probabilities * polarity_weights_t, dim=-1, keepdim=True)
        
        # Ambiguity Gate: If max class confidence is low (< 0.45), pull polarity 70% closer to 0.0 (neutral)
        max_conf = torch.max(emotion_probabilities, dim=-1, keepdim=True)[0]
        is_ambiguous = (max_conf < 0.45).float()
        polarity_scores = (1.0 - is_ambiguous) * polarity_scores + is_ambiguous * (polarity_scores * 0.3)
        
        neutral_idx = None
        for idx, lbl in self.id2label.items():
            if lbl.lower() == "neutral":
                neutral_idx = idx
                break
        if neutral_idx is not None:
            intensity_scores = 1.0 - emotion_probabilities[:, neutral_idx : neutral_idx + 1]
        else:
            intensity_scores = torch.max(emotion_probabilities, dim=-1, keepdim=True)[0]
            
        return {
            "emotion_probabilities": emotion_probabilities,
            "polarity_scores": polarity_scores,
            "intensity_scores": intensity_scores,
            "logits": logits
        }


class ModelTokenizer:
    """
    Wrapper for tokenizers supporting fallback to slow mode if fast model loading fails.
    """

    def __init__(self, model_name: str = ModelConfig.MODEL_NAME):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
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
            Dictionary with input_ids, attention_mask
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
) -> Tuple[nn.Module, ModelTokenizer, List[str]]:
    
    # 1. Prioritize your custom checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading custom model from: {checkpoint_path}")
        payload = _load_checkpoint_payload(checkpoint_path)
        
        # FIX: Define the model name correctly
        payload_model_name = payload.get("model_name", "j-hartmann/emotion-english-distilroberta-base")
        labels = emotion_labels or payload.get("emotion_labels")
        
        # Check if the checkpoint belongs to the custom MultiTaskSentimentModel
        is_multitask = any(
            k.startswith("encoder.") or
            k.startswith("emotion_head.") or
            k.startswith("polarity_head.") or
            k.startswith("intensity_head.")
            for k in payload["state_dict"].keys()
        )

        if is_multitask:
            logger.info("Detected custom MultiTaskSentimentModel checkpoint.")
            model = MultiTaskSentimentModel(payload_model_name, num_emotions=len(labels))
            model.load_state_dict(payload["state_dict"])
            
            # Attach id2label and label2id mappings for pipeline compatibility
            model.id2label = {i: label for i, label in enumerate(labels)}
            model.label2id = {label: i for i, label in enumerate(labels)}
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            return model, ModelTokenizer(payload_model_name), labels
        else:
            logger.info("Detected sequence classification checkpoint. Wrapping model...")
            # Initialize architecture
            model = AutoModelForSequenceClassification.from_pretrained(
                payload_model_name,
                num_labels=len(labels),
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
            model.load_state_dict(payload["state_dict"])
            
            # We pass the model directly to the wrapper to avoid double-loading
            wrapper = AutoSequenceClassifierWrapper(payload_model_name, classifier=model)
            wrapper.id2label = {i: label for i, label in enumerate(labels)}
            wrapper.label2id = {label: i for i, label in enumerate(labels)}
            
            # Ensure device consistency
            device = "cuda" if torch.cuda.is_available() else "cpu"
            wrapper.to(device)
            
            return wrapper, ModelTokenizer(payload_model_name), labels

    # 1. Determine if this is Hindi or English path
    is_english = False
    if model_name == "bert-base-uncased":
        is_english = True
    elif checkpoint_path and "/en/" in str(checkpoint_path):
        is_english = True

    resolved_model_name = ""
    
    # 2. Upgrade Model Checkpoints with robust context-aware alternatives
    if is_english:
        logger.info("Resolving English emotion model...")
        # English: Try 'j-hartmann/emotion-english-distilroberta-base'
        for name in ["j-hartmann/emotion-english-distilroberta-base"]:
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(name, local_files_only=True)
                resolved_model_name = name
                logger.info(f"Successfully selected English model: {resolved_model_name}")
                break
            except Exception as e:
                logger.warning(f"Could not load English model {name}: {e}")
        if not resolved_model_name:
            resolved_model_name = "j-hartmann/emotion-english-distilroberta-base"
    else:
        logger.info("Resolving Hindi emotion model...")
        # Hindi: Try 'vashuag/HindiEmotion'
        for name in ["vashuag/HindiEmotion"]:
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(name, local_files_only=True)
                resolved_model_name = name
                logger.info(f"Successfully selected Hindi model: {resolved_model_name}")
                break
            except Exception as e:
                logger.warning(f"Could not load Hindi model {name}: {e}")
        if not resolved_model_name:
            resolved_model_name = "vashuag/HindiEmotion"

    # 3. Load model wrapped in AutoSequenceClassifierWrapper with logit temperature smoothing
    logger.info("Loading weights...")
    model = AutoSequenceClassifierWrapper(resolved_model_name, temperature=1.2)
    tokenizer = ModelTokenizer(model_name=resolved_model_name)

    # Configure CUDA device if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    resolved_emotion_labels = list(model.id2label.values())
    logger.info(f"Loaded model {resolved_model_name} on device {device} with labels: {resolved_emotion_labels}")

    return model, tokenizer, resolved_emotion_labels