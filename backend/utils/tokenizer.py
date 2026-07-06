"""
utils/tokenizer.py

Word-level tokenization utilities for Hindi text.

Note: this is distinct from `models_sentiment_model.ModelTokenizer`, which
handles subword/model tokenization (input_ids, attention_mask, etc.) for
feeding text into the neural network. HindiWordTokenizer here is a plain
linguistic word tokenizer, used e.g. for stopword removal in the
preprocessing pipeline.
"""

import logging
from typing import List

try:
    from indic_nlp.tokenize import indic_tokenize
except ImportError:
    logging.warning("indic-nlp-library not installed. Install with: pip install indic-nlp-library")
    indic_tokenize = None

from utils.logger import get_logger

logger = get_logger(__name__)


class HindiWordTokenizer:
    """
    Word-level tokenizer for Hindi text.

    Uses indic-nlp-library's trivial tokenizer when available, falling
    back to simple whitespace splitting if the library is missing or
    tokenization fails for a given input.
    """

    def __init__(self, lang: str = "hi"):
        """
        Args:
            lang: Language code passed to indic_tokenize (default "hi" for Hindi)
        """
        self.lang = lang

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not isinstance(text, str) or not text:
            return []

        if indic_tokenize:
            try:
                return indic_tokenize.trivial_tokenize(text, lang=self.lang)
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")

        # Fallback to simple whitespace tokenization
        return text.split()