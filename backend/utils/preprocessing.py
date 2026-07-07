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
            sentences = re.split(r'[।.!?]+', text)
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

    @staticmethod
    def segment_hierarchical(
        raw_text: str,
        preprocessor: "HindiPreprocessor",
        max_length: int = 512,
    ) -> List[str]:
        """
        Hierarchical segmenter that preserves paragraph structure.

        Strategy (in order):
        1. Split *raw* text on double newlines (``\\n\\n``) BEFORE cleaning,
           so structural paragraph boundaries are not erased by whitespace
           normalization.
        2. Clean + normalize each paragraph individually.
        3. If a cleaned paragraph is within *max_length* characters, keep it
           as a single segment.
        4. If it exceeds *max_length*, sub-split by sentence boundaries
           (``।``, ``.``, ``!``, ``?``) and greedily pack sentences into
           chunks without crossing the limit.
        5. If a single sentence still exceeds *max_length*, word-chunk it.

        Args:
            raw_text:    Original (uncleaned) document text.
            preprocessor: A ``HindiPreprocessor`` instance used for cleaning
                          and normalisation.
            max_length:  Maximum character length per output segment.

        Returns:
            List of cleaned, length-bounded text segments.
        """
        # Step 1 — paragraph split on RAW text before any cleaning
        raw_paragraphs = re.split(r'\n\n+', raw_text)

        segments: List[str] = []

        for raw_para in raw_paragraphs:
            raw_para = raw_para.strip()
            if not raw_para:
                continue

            # Step 2 — clean + normalize this paragraph in isolation
            cleaned = preprocessor.clean_text(raw_para)
            cleaned = preprocessor.normalize_text(cleaned)
            cleaned = cleaned.strip()

            if not cleaned:
                continue

            # Step 3 — short enough → keep as one segment
            if len(cleaned) <= max_length:
                segments.append(cleaned)
                continue

            # Step 4 — sub-split by sentence boundaries
            sentences = re.split(r'(?<=[।.!?])\s+', cleaned)
            sentences = [s.strip() for s in sentences if s.strip()]

            current_chunk = ""
            for sentence in sentences:
                candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                if len(candidate) <= max_length:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        segments.append(current_chunk)
                    # Step 5 — single sentence too long → word-chunk it
                    if len(sentence) > max_length:
                        words = sentence.split()
                        word_chunk = ""
                        for word in words:
                            trial = (word_chunk + " " + word).strip() if word_chunk else word
                            if len(trial) <= max_length:
                                word_chunk = trial
                            else:
                                if word_chunk:
                                    segments.append(word_chunk)
                                word_chunk = word
                        if word_chunk:
                            segments.append(word_chunk)
                        current_chunk = ""
                    else:
                        current_chunk = sentence

            if current_chunk:
                segments.append(current_chunk)

        return segments


# ---------------------------------------------------------------------------
# Pivot-conjunction polarity adjuster
# ---------------------------------------------------------------------------

# Hindi transitional conjunctions that signal a sentiment inversion or
# contrast ("however", "but", "therefore", "nevertheless", etc.).
_PIVOT_CONJUNCTIONS: List[str] = [
    'लेकिन', 'परंतु', 'अतः', 'फिर भी', 'बल्कि', 'किंतु',
    'तथापि', 'मगर', 'पर ', 'हालाँकि', 'हालांकि',
]

# How much to attenuate (pull toward 0) the polarity when a pivot is found.
# 0.35 means a score of +0.8 becomes +0.52 and −0.9 becomes −0.585.
_PIVOT_WEIGHT: float = 0.35


class PolarityPivotAdjuster:
    """
    Post-inference polarity correction for segments containing pivot
    conjunctions (e.g. 'परंतु', 'लेकिन') that structurally invert or
    moderate the sentiment expressed before them.

    Usage::

        result = PolarityPivotAdjuster.adjust(result, segment_text)
    """

    @staticmethod
    def contains_pivot(text: str) -> bool:
        """Return True if *text* contains any known pivot conjunction."""
        return any(pivot in text for pivot in _PIVOT_CONJUNCTIONS)

    @staticmethod
    def adjust(result: Dict, text: str, pivot_weight: float = _PIVOT_WEIGHT) -> Dict:
        """
        Attenuate ``polarity_score`` toward zero when a pivot conjunction
        is present, signalling a mixed-valence context.

        Args:
            result:       Inference result dict with a ``polarity_score`` key.
            text:         The original segment text (used for pivot detection).
            pivot_weight: Attenuation factor in [0, 1].  Default 0.35.

        Returns:
            A shallow copy of *result* with an adjusted ``polarity_score``
            and an added ``pivot_adjusted`` boolean flag.
        """
        result = dict(result)  # shallow copy — do not mutate caller's dict
        if PolarityPivotAdjuster.contains_pivot(text):
            old = result.get("polarity_score", 0.0)
            result["polarity_score"] = float(old * (1.0 - pivot_weight))
            result["pivot_adjusted"] = True
        else:
            result["pivot_adjusted"] = False
        return result


def preprocess_pipeline(
    text: str,
    max_length: int = 512,
    remove_stopwords: bool = False,
    segment_strategy: str = "sentence",
) -> Tuple[List[str], Dict]:
    """
    Complete preprocessing pipeline combining all utilities.

    The default strategy (``"sentence"``) now uses the hierarchical segmenter
    which splits on ``\\n\\n`` paragraph boundaries *before* cleaning so that
    structural whitespace is preserved.  Subsequent long paragraphs are
    sub-split by sentence, then by word-boundary if needed.

    Args:
        text:             Raw input text.
        max_length:       Maximum segment character length.
        remove_stopwords: Whether to remove Hindi stopwords.
        segment_strategy: ``"sentence"`` (default / hierarchical),
                          ``"paragraph"``, or ``"fixed_length"``.

    Returns:
        Tuple of (segments, metadata).
    """
    if not text or not isinstance(text, str):
        return [], {"error": "Invalid input text"}

    # Language detection runs on raw text before any cleaning
    lang, lang_confidence = LanguageDetector.detect_language(text)

    preprocessor = HindiPreprocessor()

    # ---------------------------------------------------------------------------
    # Segmentation — choose strategy
    # ---------------------------------------------------------------------------
    if segment_strategy in ("sentence", "hierarchical"):
        # Hierarchical: paragraph-first split on RAW text, then sub-split long
        # paragraphs by sentence.  This is the default and handles \n\n inputs.
        segments = Segmentation.segment_hierarchical(
            raw_text=text,
            preprocessor=preprocessor,
            max_length=max_length,
        )
    elif segment_strategy == "paragraph":
        # Legacy paragraph split: clean first, then split on \n\n.
        # NOTE: cleaning removes newlines, so this only works if the caller
        # preserved newlines. Prefer the hierarchical strategy instead.
        cleaned_text = preprocessor.clean_text(text)
        normalized_text = preprocessor.normalize_text(cleaned_text)
        segments = Segmentation.segment_by_paragraphs(normalized_text)
    elif segment_strategy == "fixed_length":
        cleaned_text = preprocessor.clean_text(text)
        normalized_text = preprocessor.normalize_text(cleaned_text)
        segment_dicts = Segmentation.segment_by_length(normalized_text, max_length)
        segments = [seg["text"] for seg in segment_dicts]
    else:
        cleaned_text = preprocessor.clean_text(text)
        segments = [preprocessor.normalize_text(cleaned_text)]

    # Optionally strip stopwords from each segment
    if remove_stopwords:
        processed = []
        for seg in segments:
            tokens = preprocessor.tokenize(seg)
            tokens = preprocessor.remove_stopwords(tokens, keep_stopwords=False)
            processed.append(' '.join(tokens))
        segments = processed

    # Filter out very short fragments
    segments = [s for s in segments if len(s) > DataConfig.MIN_TEXT_LENGTH]

    # Build metadata from a representative cleaned version for length tracking
    representative_clean = preprocessor.clean_text(text)
    metadata = {
        "language": lang,
        "language_confidence": lang_confidence,
        "original_length": len(text),
        "cleaned_length": len(representative_clean),
        "num_segments": len(segments),
        "segment_strategy": segment_strategy,
    }

    return segments, metadata

