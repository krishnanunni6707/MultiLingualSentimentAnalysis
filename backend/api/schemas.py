"""
Shared Pydantic models for the sentiment analysis API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from config import APIConfig


class SentimentRequest(BaseModel):
	"""Request model for sentiment analysis."""

	text: str = Field(..., min_length=1, max_length=APIConfig.MAX_TEXT_LENGTH)
	return_segments: bool = Field(False, description="Return individual segment results")
	language: Optional[str] = Field(
		default=None,
		description="Language code, auto-detected if not provided",
	)

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"text": "यह फिल्म बहुत अच्छी थी, मुझे बहुत पसंद आई।",
				"return_segments": False,
			}
		}
	)


# ---------------------------------------------------------------------------
# Nested models for the unified output schema
# ---------------------------------------------------------------------------

class CrossCheckBranch(BaseModel):
	"""Sentiment scores from one branch (hindi_direct or english_translated)."""

	label: str
	polarity_score: float
	intensity: float
	confidence: float


class SegmentCrossCheck(BaseModel):
	"""Per-segment cross-check payload comparing both inference branches."""

	hindi_direct: CrossCheckBranch
	english_translated: CrossCheckBranch
	agreement: bool
	strategy_used: str
	primary_source: str


class SegmentSentiment(BaseModel):
	"""Sentiment sub-dict for a single segment."""

	label: str
	polarity_score: float
	intensity: float
	confidence: float


class SegmentResult(BaseModel):
	"""
	Fully formatted per-segment output.

	Example::

	    {
	      "segment_id": "doc_001_seg_000",
	      "text_span": "char_span: 0-312",
	      "sentiment": { "label": "joy", "polarity_score": 0.82, ... },
	      "emotion_tags": ["joy", "surprise"],
	      "language_detected": "hi",
	      "translation_used": true,
	      "cross_check": { ... }
	    }
	"""

	segment_id: str
	text_span: str
	sentiment: SegmentSentiment
	emotion_tags: List[str]
	language_detected: str
	translation_used: bool
	cross_check: Optional[SegmentCrossCheck] = None


class CrossCheckSummary(BaseModel):
	"""Document-level cross-check agreement summary."""

	strategy_used: str
	num_segments: int
	num_agree: int
	num_disagree: int
	agreement_rate: float


class OverallSentiment(BaseModel):
	"""
	Aggregated document-level sentiment.

	Example::

	    {
	      "emotions": { "anger": 0.12, "joy": 0.64, ... },
	      "dominant_emotion": "joy",
	      "polarity_score": 0.71,
	      "polarity_description": "Positive",
	      "intensity": 0.78,
	      "intensity_description": "Strong",
	      "confidence": 0.89,
	      "num_segments": 4,
	      "cross_check_summary": { ... }
	    }
	"""

	emotions: Dict[str, float]
	dominant_emotion: str
	polarity_score: float
	polarity_description: str
	intensity: float
	intensity_description: str
	confidence: float
	num_segments: int
	cross_check_summary: Optional[CrossCheckSummary] = None


class DocumentMetadata(BaseModel):
	"""Preprocessing metadata for the analysed document."""

	language: str
	language_confidence: float
	original_length: int
	cleaned_length: int
	num_segments: int
	segment_strategy: str


class DocumentSentimentResponse(BaseModel):
	"""
	Complete unified sentiment analysis response.

	Returned by ``/analyze``, ``/analyze_detailed``, and ``/analyze_json``.
	When ``segments`` is present it contains the full per-segment breakdown
	with cross-check details.
	"""

	overall_sentiment: OverallSentiment
	metadata: DocumentMetadata
	segments: Optional[List[SegmentResult]] = None


# ---------------------------------------------------------------------------
# Lightweight / legacy response models
# ---------------------------------------------------------------------------

class SentimentResponse(BaseModel):
	"""Per-segment JSON output model (legacy — prefer DocumentSentimentResponse)."""

	segment_id: str
	text_span: str
	sentiment: SegmentSentiment
	emotion_tags: List[str]
	language_detected: str
	translation_used: bool


class HealthResponse(BaseModel):
	"""Health check response."""

	status: str
	model: str
	device: str


class BatchRequest(BaseModel):
	"""Batch sentiment analysis request."""

	texts: List[str] = Field(..., min_length=1, max_length=100)


class EmotionLabelsResponse(BaseModel):
	"""Supported emotion labels response."""

	emotions: List[str]
	emotion_count: int


class StatsResponse(BaseModel):
	"""Model and API statistics response."""

	model_name: str
	base_model: str
	tasks: List[str]
	emotions_supported: int
	max_sequence_length: int
	api_version: str
	language: str


class StreamEvent(BaseModel):
	"""Event payload for the SSE endpoint."""

	type: str
	data: Optional[Dict[str, Any]] = None
	message: Optional[str] = None
