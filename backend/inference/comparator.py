"""
inference/comparator.py

Comparison and reconciliation utilities for the dual-path Hindi / English
sentiment analysis pipeline.
"""

from typing import Any, Dict, List


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
        """Return a single merged result with a cross-check payload."""
        hindi_label = hindi_result["dominant_emotion"]
        english_label = english_result["dominant_emotion"]
        agreement = hindi_label == english_label

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
                primary = hindi_result
                primary_source = "hindi_direct"

        return {
            "emotions": primary["emotions"],
            "dominant_emotion": primary["dominant_emotion"],
            "polarity_score": primary["polarity_score"],
            "intensity": primary["intensity"],
            "confidence": primary["confidence"],
            "cross_check": {
                "hindi_direct": self._minimal_result(hindi_result),
                "english_translated": self._minimal_result(english_result),
                "agreement": agreement,
                "strategy_used": self.strategy,
                "primary_source": primary_source,
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