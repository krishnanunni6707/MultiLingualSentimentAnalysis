"""
utils/preprocessing.py

Text preprocessing utilities for Hindi sentiment analysis
"""
import re
from typing import Tuple, List, Dict

try:
    # Correcting 'indic_nlp' to 'indicnlp'
    from indicnlp.normalize import indic_normalize
    from indicnlp.tokenize import sentence_tokenize
except ImportError:
    indic_normalize = None
    sentence_tokenize = None

from langdetect import detect, detect_langs, LangDetectException
import nltk

from config import DataConfig
from utils.logger import get_logger
from utils.tokenizer import HindiWordTokenizer

logger = get_logger(__name__)

if indic_normalize is None:
    logger.warning("indic-nlp-library not installed. Install with: pip install indic-nlp-library")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class HindiPreprocessor:
    """
    Comprehensive Hindi text preprocessing pipeline
    """

    def __init__(self, remove_diacritics: bool = False):
        """
        Initialize the preprocessor

        Args:
            remove_diacritics: Whether to remove Hindi diacritical marks (matras)
        """
        self.remove_diacritics = remove_diacritics
        if indic_normalize:
            try:
                factory = indic_normalize.IndicNormalizerFactory()
                self.normalizer = factory.get_normalizer("hi")
            except Exception as e:
                logger.warning(f"Failed to initialize normalizer: {e}")
                self.normalizer = None
        else:
            self.normalizer = None
        self.tokenizer = HindiWordTokenizer(lang="hi")

        # Hindi Unicode ranges
        self.devanagari_range = range(0x0900, 0x097F)

        # Common Hindi stop patterns
        self.stop_words = set([
            'और', 'या', 'है', 'हैं', 'हो', 'होगा', 'होंगे', 'होगी', 'होगें',
            'की', 'का', 'के', 'में', 'से', 'को', 'ने', 'द्वारा',
            'जो', 'जब', 'तब', 'यह', 'वह', 'ये', 'वो', 'मैं', 'तुम', 'आप',
            'हम', 'वे', 'किस', 'कौन', 'क्या', 'कहाँ', 'कब'
        ])

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters but keep Hindi punctuation
        # Keep: Devanagari characters, spaces, common punctuation (। , ! ?)
        text = re.sub(
            r'[^\u0900-\u097F\s\.\,\!\?\;\:\'\"\-]',
            '',
            text
        )

        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize Hindi text using indic-nlp library

        Args:
            text: Cleaned text

        Returns:
            Normalized text
        """
        if self.normalizer:
            try:
                text = self.normalizer.normalize(text)
            except Exception as e:
                logger.warning(f"Normalization error: {e}")

        # Remove diacritics if specified
        if self.remove_diacritics:
            text = self._remove_diacritics(text)

        return text

    def _remove_diacritics(self, text: str) -> str:
        """Remove Hindi diacritical marks (matras)"""
        # Hindi matras (vowel modifiers)
        matras = '\u093E\u093F\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C'
        return re.sub(f'[{matras}]', '', text)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Hindi text

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)

    def remove_stopwords(self, tokens: List[str], keep_stopwords: bool = False) -> List[str]:
        """
        Remove or keep Hindi stopwords

        Args:
            tokens: List of tokens
            keep_stopwords: If False, remove stopwords; if True, keep them

        Returns:
            Filtered tokens
        """
        if keep_stopwords:
            return tokens

        return [token for token in tokens if token not in self.stop_words]

    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Raw text
            remove_stopwords: Whether to remove stopwords

        Returns:
            Preprocessed text
        """
        text = self.clean_text(text)
        text = self.normalize_text(text)

        if remove_stopwords:
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens, keep_stopwords=False)
            text = ' '.join(tokens)

        return text


class LanguageDetector:
    """
    Language detection for multilingual texts
    """

    @staticmethod
    def detect_language(text: str) -> Tuple[str, float]:
        """
        Detect language of text

        Args:
            text: Text to detect language

        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            lang = detect(text)
            probabilities = detect_langs(text)
            confidence = max([p.prob for p in probabilities if p.lang == lang], default=0.0)
            return lang, confidence
        except LangDetectException:
            return "unknown", 0.0

    @staticmethod
    def is_hindi(text: str, threshold: float = 0.7) -> bool:
        """
        Check if text is primarily Hindi

        Args:
            text: Text to check
            threshold: Minimum confidence required

        Returns:
            Boolean indicating if text is Hindi
        """
        lang, confidence = LanguageDetector.detect_language(text)
        return lang == "hi" and confidence >= threshold


class Segmentation:
    """
    Document segmentation strategies
    """

    @staticmethod
    def segment_by_sentences(text: str) -> List[str]:
        """
        Segment text by sentences using NLTK

        Args:
            text: Text to segment

        Returns:
            List of sentences
        """
        try:
            # Try using NLTK punkt tokenizer
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback: split by common Hindi sentence terminators
            sentences = re.split(r'।|।।|\.|!|\?', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    @staticmethod
    def segment_by_paragraphs(text: str) -> List[str]:
        """
        Segment text by paragraphs

        Args:
            text: Text to segment

        Returns:
            List of paragraphs
        """
        # Split by double newline
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def segment_by_length(text: str, max_length: int = 512, overlap: int = 50) -> List[Dict]:
        """
        Segment text by fixed character length with overlap

        Args:
            text: Text to segment
            max_length: Maximum segment length
            overlap: Number of characters to overlap

        Returns:
            List of segments with start/end positions
        """
        segments = []
        start = 0

        while start < len(text):
            end = min(start + max_length, len(text))

            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = text.rfind('।', start, end)
                if last_period > start:
                    end = last_period + 1

            segment = {
                "text": text[start:end].strip(),
                "start_char": start,
                "end_char": end
            }

            if segment["text"]:  # Only add non-empty segments
                segments.append(segment)

            start = end - overlap

            if start >= len(text):
                break

        return segments


def preprocess_pipeline(
    text: str,
    max_length: int = 512,
    remove_stopwords: bool = False,
    segment_strategy: str = "sentence"
) -> Tuple[List[str], Dict]:
    """
    Complete preprocessing pipeline combining all utilities

    Args:
        text: Raw input text
        max_length: Maximum segment length
        remove_stopwords: Whether to remove stopwords
        segment_strategy: How to segment (sentence, paragraph, fixed_length)

    Returns:
        Tuple of (segments, metadata)
    """
    if not text or not isinstance(text, str):
        return [], {"error": "Invalid input text"}

    # Language detection
    lang, lang_confidence = LanguageDetector.detect_language(text)

    # Preprocessing
    preprocessor = HindiPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    normalized_text = preprocessor.normalize_text(cleaned_text)

    # Segmentation
    if segment_strategy == "sentence":
        segments = Segmentation.segment_by_sentences(normalized_text)
    elif segment_strategy == "paragraph":
        segments = Segmentation.segment_by_paragraphs(normalized_text)
    elif segment_strategy == "fixed_length":
        segment_dicts = Segmentation.segment_by_length(normalized_text, max_length)
        segments = [seg["text"] for seg in segment_dicts]
    else:
        segments = [normalized_text]

    # Filter short segments
    segments = [s for s in segments if len(s) > DataConfig.MIN_TEXT_LENGTH]

    metadata = {
        "language": lang,
        "language_confidence": lang_confidence,
        "original_length": len(text),
        "cleaned_length": len(cleaned_text),
        "num_segments": len(segments),
        "segment_strategy": segment_strategy
    }

    return segments, metadata

