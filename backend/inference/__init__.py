"""
inference package

Exposes the document segmenter, output assembler, inference pipeline,
and real-time inference wrapper.
"""

from inference.segmentation import DocumentSegmenter
from inference.output_assembler import (
    OutputAssembler,
    get_intensity_description,
    get_polarity_description,
)
from inference.comparator import ResultComparator
from inference.pipeline import SentimentInferencePipeline, RealTimeInference, EMOTION_LABELS
from inference.pipeline_en import SentimentInferencePipelineEN, RealTimeInferenceEN, EMOTION_LABELS_EN

__all__ = [
    "DocumentSegmenter",
    "OutputAssembler",
    "get_intensity_description",
    "get_polarity_description",
    "ResultComparator",
    "SentimentInferencePipeline",
    "RealTimeInference",
    "EMOTION_LABELS",
    "SentimentInferencePipelineEN",
    "RealTimeInferenceEN",
    "EMOTION_LABELS_EN",
]