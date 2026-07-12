"""
utils/translator.py

Hindi to English translation helper used by the dual-path inference
pipeline. The translator is batch-oriented so segment translation can be
run efficiently before English sentiment analysis.

LAZY-LOADED: Model only loads on first translate() call to save memory.
"""

from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TranslationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class HindiToEnglishTranslator:
    """Translate Hindi text into English using a seq2seq model (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = TranslationConfig.MODEL_NAME,
        device: str = TranslationConfig.DEVICE,
        max_length: int = TranslationConfig.MAX_LENGTH,
    ):
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length

        # Lazy load: store config, but don't load model yet ✅
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """Load model and tokenizer on first use (lazy loading)."""
        if self._loaded:
            return  # Already loaded, skip
        
        logger.info(f"[Lazy Load] Loading translator: {self.model_name}")
        print(f"[~] Loading Hindi-English translator...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"Translator ready on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load translator: {str(e)}")
            raise

    def translate(self, text: str) -> str:
        """Translate a single text span."""
        self._ensure_loaded()  # ✅ Load on first call
        
        if not text or not isinstance(text, str):
            return ""

        return self.translate_batch([text])[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple text spans in one call."""
        self._ensure_loaded()  # ✅ Load on first call
        
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