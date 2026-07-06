"""
inference/output_assembler.py

Aggregates segment-level predictions into document-level sentiment,
formats results into the pipeline's structured JSON output schema, and
saves results to disk.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def get_intensity_description(intensity: float) -> str:
    """Get human-readable intensity description"""
    if intensity < 0.2:
        return "Very weak"
    elif intensity < 0.4:
        return "Weak"
    elif intensity < 0.6:
        return "Moderate"
    elif intensity < 0.8:
        return "Strong"
    else:
        return "Very strong"


def get_polarity_description(polarity: float) -> str:
    """Get human-readable polarity description"""
    if polarity < -0.5:
        return "Highly negative"
    elif polarity < -0.2:
        return "Negative"
    elif polarity < 0.2:
        return "Neutral"
    elif polarity < 0.5:
        return "Positive"
    else:
        return "Highly positive"


def _top_n_emotions(emotions: Dict[str, float], n: int = 2) -> List[str]:
    """Return the top-n emotion labels sorted by score (descending)."""
    return [label for label, _ in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:n]]


class OutputAssembler:
    """
    Combines segment-level model predictions into document-level results,
    in both the internal aggregate format and the pipeline's structured
    JSON schema, and handles saving results to disk.
    """

    def __init__(self, emotion_labels: List[str]):
        """
        Args:
            emotion_labels: Ordered list of emotion names matching the
                model's emotion output dimension (must match the label set
                the model was trained on)
        """
        self.emotion_labels = emotion_labels

    def aggregate_results(self, segment_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate segment-level results to document level (mean across
        segments for each emotion, polarity, intensity, and confidence).

        Args:
            segment_results: List of segment predictions (as produced by
                SentimentInferencePipeline.batch_inference)

        Returns:
            Aggregated document-level sentiment dict
        """
        if not segment_results:
            return {}

        emotion_scores = {emotion: [] for emotion in self.emotion_labels}
        polarity_scores = []
        intensities = []
        confidences = []

        for result in segment_results:
            for emotion, score in result["emotions"].items():
                emotion_scores.setdefault(emotion, []).append(score)
            polarity_scores.append(result["polarity_score"])
            intensities.append(result["intensity"])
            confidences.append(result["confidence"])

        aggregated_emotions = {
            emotion: float(np.mean(scores)) if scores else 0.0
            for emotion, scores in emotion_scores.items()
        }

        avg_polarity = float(np.mean(polarity_scores))
        avg_intensity = float(np.mean(intensities))
        avg_confidence = float(np.mean(confidences))

        return {
            "emotions": aggregated_emotions,
            "dominant_emotion": max(aggregated_emotions, key=aggregated_emotions.get),
            "polarity_score": avg_polarity,
            "polarity_description": get_polarity_description(avg_polarity),
            "intensity": avg_intensity,
            "intensity_description": get_intensity_description(avg_intensity),
            "confidence": avg_confidence,
            "num_segments": len(segment_results),
        }

    def format_segments(
        self,
        segments: List[str],
        segment_results: List[Dict],
        segment_id: str,
        lang: str,
        translation_used: bool,
        char_offsets: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format per-segment results into the pipeline's structured JSON schema.

        Args:
            segments: The text segments (original text, pre-tokenisation)
            segment_results: Per-segment model predictions (same length/order
                as `segments`)
            segment_id: Identifier for the parent document (e.g. "doc_001")
            lang: Detected language code
            translation_used: Whether the input required translation
            char_offsets: Optional list of cumulative start-char positions for
                each segment within the cleaned document. When omitted the
                offsets are reconstructed from segment lengths assuming they
                are contiguous (with a single space separator).

        Returns:
            List of per-segment output dicts matching the full JSON schema.
        """
        formatted = []

        # Build cumulative char offsets if not provided.
        # Segments are assumed to be contiguous slices of the cleaned text
        # separated by a single space, which is the typical output of the
        # sentence/paragraph segmentation strategies.
        if char_offsets is None:
            char_offsets = []
            cursor = 0
            for seg in segments:
                char_offsets.append(cursor)
                cursor += len(seg) + 1  # +1 for separator space

        for seg_idx, (segment, result) in enumerate(zip(segments, segment_results)):
            start = char_offsets[seg_idx]
            end = start + len(segment)

            output = {
                "segment_id": f"{segment_id}_seg_{seg_idx:03d}",
                "text_span": f"char_span: {start}-{end}",
                "sentiment": {
                    "label": result["dominant_emotion"],
                    "polarity_score": result["polarity_score"],
                    "intensity": result["intensity"],
                    "confidence": result["confidence"],
                },
                # Top-2 emotions by score — gives meaningful tags even when
                # no single emotion exceeds 0.5.
                "emotion_tags": _top_n_emotions(result["emotions"], n=2),
                "language_detected": lang,
                "translation_used": translation_used,
            }

            if "cross_check" in result:
                output["cross_check"] = result["cross_check"]

            formatted.append(output)

        return formatted

    def format_json_output(
        self,
        segments: List[str],
        segment_results: List[Dict],
        segment_id: str,
        lang: str,
        translation_used: bool,
        char_offsets: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format per-segment results into the pipeline's structured JSON schema.

        Kept for backward compatibility — delegates to ``format_segments``.

        Args:
            segments: The text segments
            segment_results: Per-segment model predictions (same length/order
                as `segments`)
            segment_id: Identifier for the parent document
            lang: Detected language code
            translation_used: Whether the input required translation (i.e.
                detected language wasn't Hindi)
            char_offsets: Optional cumulative start-char positions per segment.

        Returns:
            List of per-segment output dicts
        """
        return self.format_segments(
            segments=segments,
            segment_results=segment_results,
            segment_id=segment_id,
            lang=lang,
            translation_used=translation_used,
            char_offsets=char_offsets,
        )

    def format_full_output(
        self,
        segments: List[str],
        segment_results: List[Dict],
        metadata: Dict[str, Any],
        segment_id: str,
        lang: str,
        lang_confidence: float,
        translation_used: bool,
        cross_check_summary: Optional[Dict[str, Any]] = None,
        char_offsets: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete unified document output matching the target schema:

        {
          "overall_sentiment": { emotions, dominant_emotion, polarity_score,
                                  polarity_description, intensity,
                                  intensity_description, confidence,
                                  num_segments, cross_check_summary },
          "metadata": { language, language_confidence, original_length,
                        cleaned_length, num_segments, segment_strategy },
          "segments": [ { segment_id, text_span, sentiment, emotion_tags,
                          language_detected, translation_used, cross_check } ]
        }

        Args:
            segments: Text segments
            segment_results: Per-segment model predictions
            metadata: Preprocessing metadata from the segmenter
            segment_id: Parent document identifier (e.g. "doc_001")
            lang: Detected language code
            lang_confidence: Confidence score for language detection
            translation_used: Whether translation was applied
            cross_check_summary: Optional cross-check summary dict from
                ResultComparator.summarize()
            char_offsets: Optional cumulative start-char positions per segment

        Returns:
            Full unified output dict
        """
        # Build overall_sentiment
        overall = self.aggregate_results(segment_results)
        if cross_check_summary is not None:
            overall["cross_check_summary"] = cross_check_summary

        # Ensure metadata has language_confidence from the actual detection
        # (the segmenter's metadata already carries these, but we allow the
        # caller to override with the pipeline-level detection values).
        full_metadata = dict(metadata)
        full_metadata["language"] = lang
        full_metadata["language_confidence"] = lang_confidence

        # Build formatted segments
        formatted_segments = self.format_segments(
            segments=segments,
            segment_results=segment_results,
            segment_id=segment_id,
            lang=lang,
            translation_used=translation_used,
            char_offsets=char_offsets,
        )

        return {
            "overall_sentiment": overall,
            "metadata": full_metadata,
            "segments": formatted_segments,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save inference results to a JSON file.

        Args:
            results: Results dictionary
            output_path: Path to save results to
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {path}")