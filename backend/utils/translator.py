"""
utils/translator.py

Hindi to English translation helper used by the dual-path inference
pipeline. The translator is batch-oriented so segment translation can be
run efficiently before English sentiment analysis.
"""

from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TranslationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class HindiToEnglishTranslator:
    """Translate Hindi text into English using a seq2seq model."""

    def __init__(
        self,
        model_name: str = TranslationConfig.MODEL_NAME,
        device: str = TranslationConfig.DEVICE,
        max_length: int = TranslationConfig.MAX_LENGTH,
    ):
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str) -> str:
        """Translate a single text span."""
        if not text or not isinstance(text, str):
            return ""

        return self.translate_batch([text])[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple text spans in one call."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
            )

        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)