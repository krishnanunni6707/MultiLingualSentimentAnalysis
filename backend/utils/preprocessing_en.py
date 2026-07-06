"""
utils/preprocessing_en.py

Text preprocessing utilities for English sentiment analysis.
Keeps the same public method names/signatures as HindiPreprocessor so the
two preprocessors are interchangeable.
"""

import re
import unicodedata
from typing import Dict, List, Tuple

import nltk

from config import DataConfig
from utils.logger import get_logger
from utils.preprocessing import LanguageDetector, Segmentation

logger = get_logger(__name__)


class EnglishPreprocessor:
    """Comprehensive English text preprocessing pipeline."""

    def __init__(self, remove_diacritics: bool = False):
        self.remove_diacritics = remove_diacritics
        self.stop_words = set(
            [
                "a", "an", "and", "are", "as", "at", "be", "because",
                "but", "by", "for", "from", "had", "has", "have", "he",
                "her", "his", "i", "if", "in", "is", "it", "its",
                "me", "my", "not", "of", "on", "or", "our", "she",
                "so", "that", "the", "their", "them", "there", "these",
                "they", "this", "to", "was", "we", "were", "what",
                "when", "where", "which", "who", "will", "with", "you",
                "your",
            ]
        )

    def clean_text(self, text: str) -> str:
        """Comprehensively clean English text."""
        if not isinstance(text, str):
            return ""

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^A-Za-z0-9\s\.,!?;:'\"\-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_text(self, text: str) -> str:
        """Normalize English text to a consistent lowercase form."""
        text = text.lower()

        if self.remove_diacritics:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(ch for ch in text if not unicodedata.combining(ch))

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text."""
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return re.findall(r"[A-Za-z0-9']+", text)

    def remove_stopwords(self, tokens: List[str], keep_stopwords: bool = False) -> List[str]:
        """Remove or keep English stopwords."""
        if keep_stopwords:
            return tokens

        return [token for token in tokens if token.lower() not in self.stop_words]

    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.normalize_text(text)

        if remove_stopwords:
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens, keep_stopwords=False)
            text = " ".join(tokens)

        return text


def preprocess_pipeline_en(
    text: str,
    max_length: int = 512,
    remove_stopwords: bool = False,
    segment_strategy: str = "sentence",
) -> Tuple[List[str], Dict]:
    """
    Complete English preprocessing pipeline.

    Mirrors utils.preprocessing.preprocess_pipeline so the shared
    DocumentSegmenter can switch between Hindi and English cleanly.
    """
    if not text or not isinstance(text, str):
        return [], {"error": "Invalid input text"}

    lang, lang_confidence = LanguageDetector.detect_language(text)

    preprocessor = EnglishPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    normalized_text = preprocessor.normalize_text(cleaned_text)

    if segment_strategy == "sentence":
        segments = Segmentation.segment_by_sentences(normalized_text)
    elif segment_strategy == "paragraph":
        segments = Segmentation.segment_by_paragraphs(normalized_text)
    elif segment_strategy == "fixed_length":
        segment_dicts = Segmentation.segment_by_length(normalized_text, max_length)
        segments = [seg["text"] for seg in segment_dicts]
    else:
        segments = [normalized_text]

    segments = [segment for segment in segments if len(segment) > DataConfig.MIN_TEXT_LENGTH]

    if remove_stopwords:
        filtered_segments = []
        for segment in segments:
            tokens = preprocessor.tokenize(segment)
            tokens = preprocessor.remove_stopwords(tokens, keep_stopwords=False)
            filtered_segments.append(" ".join(tokens))
        segments = [segment for segment in filtered_segments if segment]

    metadata = {
        "language": lang,
        "language_confidence": lang_confidence,
        "original_length": len(text),
        "cleaned_length": len(cleaned_text),
        "num_segments": len(segments),
        "segment_strategy": segment_strategy,
    }

    return segments, metadata