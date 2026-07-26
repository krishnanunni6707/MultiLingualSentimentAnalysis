"""
inference/comparator.py

Comparison and reconciliation utilities for the dual-path Hindi / English
sentiment analysis pipeline.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Valence grouping — maps every Ekman emotion to a broad sentiment polarity.
# Two branches that agree on valence but differ on fine-grained label are
# still considered to be in agreement.
#
# NOTE: 'surprise' is mapped to 'neutral' because surprise is polarity-
# ambivalent — it can be wonder/delight (positive) or shock/alarm (negative).
# Mapping it to 'positive' caused systematic mismatches when the English
# GoEmotions model defaults to 'surprise' on uncertain/translated inputs.
# ---------------------------------------------------------------------------
_VALENCE_MAP: Dict[str, str] = {
    # Canonical Ekman labels
    "joy":      "positive",
    "surprise": "neutral",    # ambivalent — not reliably positive or negative
    "anger":    "negative",
    "sadness":  "negative",
    "fear":     "negative",
    "disgust":  "negative",
    # Inflected / checkpoint-specific variants (e.g. EmoInHindi uses "disgusted")
    "disgusted":  "negative",
    "angry":      "negative",
    "sad":        "negative",
    "fearful":    "negative",
    "scared":     "negative",
    "joyful":     "positive",
    "happy":      "positive",
    "surprised":  "neutral",
    # Extended 17-class checkpoint labels
    "hopeful":       "positive",
    "confident":     "positive",
    "grateful":      "positive",
    "compassion":    "positive",
    "impressed":     "positive",
    "anticipation":  "neutral",
    "annoyed":       "negative",
    "apprehensive":  "negative",
    "guilty":        "negative",
    "confused":      "neutral",
    "neutral":       "neutral",
}
_POLARITY_PROXIMITY_THRESHOLD: float = 0.20
# Wider threshold used when the secondary branch has low confidence.
_POLARITY_PROXIMITY_THRESHOLD_LAX: float = 0.35
# Below this confidence the secondary branch is considered uncertain.
_LOW_CONFIDENCE_GATE: float = 0.50


def _valence(label: str) -> str:
    """Return the broad valence group for a given emotion label.

    Performs a direct lookup first, then strips common English morphological
    suffixes (``-ed``, ``-ful``, ``-ness``, ``-ity``) before a second lookup,
    so checkpoint-specific inflected forms (e.g. ``"disgusted"``) are handled
    even when not listed explicitly in ``_VALENCE_MAP``.
    """
    key = label.lower()
    if key in _VALENCE_MAP:
        return _VALENCE_MAP[key]
    # Suffix-strip fallback
    for suffix in ("ned", "ed", "ful", "ness", "ity", "ing"):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if base in _VALENCE_MAP:
                return _VALENCE_MAP[base]
    return "neutral"


class ResultComparator:
    """Reconcile Hindi-direct and English-translated sentiment results."""

    def __init__(self, strategy: str = "agreement_flag"):
        self.strategy = strategy

    @staticmethod
    def _minimal_result(result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": result["dominant_emotion"],
            "polarity_score": result["polarity_score"],
            "intensity": result["intensity"],
            "confidence": result["confidence"],
        }

    def reconcile(
        self,
        hindi_result: Dict[str, Any],
        english_result: Dict[str, Any],
        language: str = "hi",
    ) -> Dict[str, Any]:
        """Return a single merged result with a cross-check payload.

        Agreement is evaluated via three independent criteria; passing *any*
        one is sufficient to mark the segment as agreed:

        1. **Exact match** — both branches predict the same emotion label.
        2. **Valence match** — both labels map to the same broad valence group
           (positive / negative / neutral), so ``fear`` vs ``sadness`` still
           counts as agreement.
        3. **Polarity proximity** — the absolute difference between the two
           branches' polarity scores is within
           ``_POLARITY_PROXIMITY_THRESHOLD`` (default ±0.2).
        """
        hindi_label   = hindi_result["dominant_emotion"]
        english_label = english_result["dominant_emotion"]

        exact_match   = hindi_label == english_label
        valence_match = _valence(hindi_label) == _valence(english_label)

        polarity_diff = abs(
            hindi_result["polarity_score"] - english_result["polarity_score"]
        )

        # If the secondary (cross-check) branch has low confidence we apply a
        # more lenient polarity threshold — an uncertain model should not
        # definitively veto a high-confidence primary prediction.
        if language == "en":
            secondary_confidence = hindi_result["confidence"]
        else:
            secondary_confidence = english_result["confidence"]

        proximity_threshold = (
            _POLARITY_PROXIMITY_THRESHOLD_LAX
            if secondary_confidence < _LOW_CONFIDENCE_GATE
            else _POLARITY_PROXIMITY_THRESHOLD
        )
        polarity_proximate = polarity_diff <= proximity_threshold

        agreement = exact_match or valence_match or polarity_proximate

        agreement_detail = {
            "exact_label_match":       exact_match,
            "valence_match":           valence_match,
            "polarity_diff":           round(polarity_diff, 4),
            "polarity_proximate":      polarity_proximate,
            "polarity_threshold_used": round(proximity_threshold, 2),
            "secondary_confidence":    round(secondary_confidence, 4),
            "low_confidence_gate":     _LOW_CONFIDENCE_GATE,
        }

        if self.strategy == "confidence_weighted_pick":
            primary = hindi_result if hindi_result["confidence"] >= english_result["confidence"] else english_result
            primary_source = "hindi_direct" if primary is hindi_result else ("english_direct" if language == "en" else "english_translated")
        elif self.strategy == "average":
            emotions = {}
            for label in set(hindi_result["emotions"]) | set(english_result["emotions"]):
                emotions[label] = float(
                    (hindi_result["emotions"].get(label, 0.0) + english_result["emotions"].get(label, 0.0)) / 2.0
                )
            primary = {
                "emotions": emotions,
                "dominant_emotion": max(emotions, key=emotions.get),
                "polarity_score": float((hindi_result["polarity_score"] + english_result["polarity_score"]) / 2.0),
                "intensity": float((hindi_result["intensity"] + english_result["intensity"]) / 2.0),
                "confidence": float((hindi_result["confidence"] + english_result["confidence"]) / 2.0),
            }
            primary_source = "averaged"
        else:
            if language == "en":
                primary = english_result
                primary_source = "english_direct"
            else:
                # For Hindi inputs: if the two branches disagree on valence AND the
                # Hindi model predicts a strongly negative emotion while the English
                # cross-check is neutral/positive, defer to the English branch.
                # This compensates for the known keyword-bias of the undertrained
                # Hindi model (e.g. classifying grateful text as "disgusted").
                hindi_valence = _valence(hindi_result["dominant_emotion"])
                english_valence = _valence(english_result["dominant_emotion"])
                hindi_is_strongly_negative = (
                    hindi_valence == "negative"
                    and hindi_result["polarity_score"] < -0.5
                    and hindi_result["confidence"] > 0.85
                )
                english_is_nonnegative = english_valence in ("positive", "neutral")

                if not valence_match and hindi_is_strongly_negative and english_is_nonnegative:
                    primary = english_result
                    primary_source = "english_translated"
                else:
                    primary = hindi_result
                    primary_source = "hindi_direct"

        return {
            "emotions":         primary["emotions"],
            "dominant_emotion": primary["dominant_emotion"],
            "polarity_score":   primary["polarity_score"],
            "intensity":        primary["intensity"],
            "confidence":       primary["confidence"],
            "cross_check": {
                "hindi_direct":      self._minimal_result(hindi_result),
                "english_translated": self._minimal_result(english_result),
                "translated_text":   english_result.get("translated_text"),
                "agreement":         agreement,
                "agreement_detail":  agreement_detail,
                "strategy_used":     self.strategy,
                "primary_source":    primary_source,
            },
        }

    def summarize(self, reconciled_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize cross-check agreement across all segments."""
        if not reconciled_results:
            return {
                "strategy_used": self.strategy,
                "num_segments": 0,
                "agreement_rate": 0.0,
            }

        agreements = [bool(result.get("cross_check", {}).get("agreement", False)) for result in reconciled_results]
        return {
            "strategy_used": self.strategy,
            "num_segments": len(reconciled_results),
            "num_agree": int(sum(agreements)),
            "num_disagree": int(len(agreements) - sum(agreements)),
            "agreement_rate": float(sum(agreements) / len(agreements)),
        }