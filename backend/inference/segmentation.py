"""
inference/segmentation.py

Document segmentation for inference: wraps the shared preprocessing
pipeline with inference-specific defaults (segment length, strategy),
and can switch between Hindi and English preprocessing.
"""

from typing import Callable, Dict, List, Tuple

from config import InferenceConfig
from utils.logger import get_logger
from utils.preprocessing import preprocess_pipeline
from utils.preprocessing_en import preprocess_pipeline_en

logger = get_logger(__name__)


class DocumentSegmenter:
    """
    Segments raw documents into model-sized chunks for inference, using
    the shared preprocessing pipeline (utils.preprocessing.preprocess_pipeline)
    under the hood.
    """

    def __init__(
        self,
        max_length: int = InferenceConfig.SEGMENT_MAX_LENGTH,
        strategy: str = InferenceConfig.SEGMENTATION_STRATEGY,
        language: str = "hi",
        preprocess_fn: Callable | None = None,
    ):
        """
        Args:
            max_length: Maximum segment length (used by "fixed_length" strategy)
            strategy: Segmentation strategy — "sentence", "paragraph", or
                "fixed_length"
            language: Language code used to choose the default preprocessing
                pipeline ("hi" or "en")
            preprocess_fn: Optional custom preprocessing function
        """
        self.max_length = max_length
        self.strategy = strategy
        self.language = language
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        elif language == "en":
            self.preprocess_fn = preprocess_pipeline_en
        else:
            self.preprocess_fn = preprocess_pipeline

    def segment(self, text: str) -> Tuple[List[str], Dict]:
        """
        Preprocess and segment a document.

        Args:
            text: Raw input text

        Returns:
            Tuple of (segments, metadata) — metadata includes detected
            language, confidence, original/cleaned length, and segment
            count (see utils.preprocessing.preprocess_pipeline)
        """
        segments, metadata = self.preprocess_fn(
            text,
            max_length=self.max_length,
            segment_strategy=self.strategy,
        )
        return segments, metadata