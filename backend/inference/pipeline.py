"""
inference/pipeline.py

Inference pipeline for sentiment analysis on long Hindi documents.
Handles preprocessing/segmentation and batched model inference, and
delegates result aggregation/formatting/saving to OutputAssembler.
"""

import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from config import CHECKPOINT_DIR, InferenceConfig, ModelConfig, TranslationConfig
from models.sentiment_model import MultiTaskSentimentModel
from models.model_utils import ModelTokenizer, load_model

from utils.logger import get_logger
from utils.translator import HindiToEnglishTranslator
from inference.segmentation import DocumentSegmenter
from inference.output_assembler import OutputAssembler
from inference.pipeline_en import SentimentInferencePipelineEN, EMOTION_LABELS_EN
from inference.comparator import ResultComparator

logger = get_logger(__name__)

EMOTION_LABELS = list(ModelConfig.DEFAULT_EMOTION_LABELS)


class SentimentInferencePipeline:
    """
    Complete inference pipeline for sentiment analysis
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        english_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = InferenceConfig.BATCH_SIZE,
        emotion_labels: Optional[List[str]] = None,
        comparison_strategy: str = "agreement_flag",
    ):
        """
        Initialize inference pipeline

        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on
            batch_size: Batch size for inference
            emotion_labels: Optional ordered emotion label names matching the
                model's emotion output dimension
        """
        self.device = device
        self.batch_size = batch_size
        self.model, self.tokenizer, resolved_labels = load_model(
            checkpoint_path=model_path,
            emotion_labels=emotion_labels or EMOTION_LABELS,
            model_name=ModelConfig.MODEL_NAME,
        )
        self.emotion_labels = resolved_labels

        self.model.to(device)
        self.model.eval()

        # Segmentation + output assembly
        self.segmenter = DocumentSegmenter(language="hi")
        self.output_assembler = OutputAssembler(emotion_labels=self.emotion_labels)

        default_english_checkpoint = Path(CHECKPOINT_DIR) / "en" / "best_model.pt"
        if english_model_path is None and default_english_checkpoint.exists():
            english_model_path = str(default_english_checkpoint)

        self.translator = HindiToEnglishTranslator(
            model_name=TranslationConfig.MODEL_NAME,
            device=TranslationConfig.DEVICE,
            max_length=TranslationConfig.MAX_LENGTH,
        )
        self.english_pipeline = SentimentInferencePipelineEN(
            model_path=english_model_path,
            device=device,
            batch_size=batch_size,
            emotion_labels=EMOTION_LABELS_EN,
        )
        self.comparator = ResultComparator(strategy=comparison_strategy)

    def preprocess_text(self, text: str) -> Tuple[List[str], Dict]:
        """
        Preprocess and segment text

        Args:
            text: Raw input text

        Returns:
            Tuple of (segments, metadata)
        """
        return self.segmenter.segment(text)

    def batch_inference(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of texts

        Args:
            texts: List of text segments

        Returns:
            List of predictions for each segment
        """
        results = []

        # Tokenize all texts
        encodings = self.tokenizer.batch_encode(texts)

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Process outputs
        emotion_probs = outputs["emotion_probabilities"]
        polarity_scores = outputs["polarity_scores"]
        intensity_scores = outputs["intensity_scores"]

        for i in range(len(texts)):
            emotion_dict = {}
            for j, label in enumerate(self.emotion_labels):
                emotion_dict[label] = emotion_probs[i, j].item()

            result = {
                "text": texts[i],
                "emotions": emotion_dict,
                "polarity_score": polarity_scores[i].item(),
                "intensity": intensity_scores[i].item(),
                "dominant_emotion": max(emotion_dict, key=emotion_dict.get),
                "confidence": max(emotion_dict.values())
            }

            results.append(result)

        return results

    def _analyze_english_branch(self, segments: List[str], translate: bool = True) -> List[Dict[str, Any]]:
        """Translate segments to English when needed and run the English model."""
        translated_segments = self.translator.translate_batch(segments) if translate else segments
        return self.english_pipeline.batch_inference(translated_segments)

    def _run_parallel_analysis(
        self,
        segments: List[str],
        translate: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run Hindi direct inference and English translated inference in parallel."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            hindi_future = executor.submit(self.batch_inference, segments)
            english_future = executor.submit(self._analyze_english_branch, segments, translate)
            hindi_results = hindi_future.result()
            english_results = english_future.result()

        return hindi_results, english_results

    def process_document(
        self,
        text: str,
        return_segments: bool = False,
        segment_id: str = "doc_001",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete document.

        Args:
            text: Full document text
            return_segments: Whether to include per-segment detail in the
                output.  When True the returned dict contains a ``segments``
                key whose value is a list of fully-formatted segment dicts
                (with ``segment_id``, ``text_span``, ``sentiment`` sub-dict,
                ``emotion_tags``, ``language_detected``, ``translation_used``
                and ``cross_check``).
            segment_id: Parent document identifier used to build per-segment
                ``segment_id`` strings (e.g. ``"doc_001"`` produces
                ``"doc_001_seg_000"``, ``"doc_001_seg_001"`` …).
            language: Optional language code ("hi", "en", etc.). Detected if not provided.

        Returns:
            Dictionary with overall sentiment and optionally segment details.
        """
        # Detect language first if not specified
        if language is None:
            from utils.preprocessing import LanguageDetector
            lang, lang_confidence = LanguageDetector.detect_language(text)
        else:
            lang = language
            lang_confidence = 1.0

        # Preprocess and segment with language-specific segmenter
        if lang == "en":
            segmenter = DocumentSegmenter(language="en")
        else:
            segmenter = DocumentSegmenter(language="hi")

        segments, metadata = segmenter.segment(text)

        if not segments:
            return {
                "error": "No valid segments found",
                "metadata": metadata
            }

        # Ensure correct language in metadata
        metadata["language"] = lang
        metadata["language_confidence"] = lang_confidence

        # Process the Hindi-direct branch and the translated-English branch
        # in parallel, then reconcile each segment.
        translate = lang == "hi"
        hindi_results, english_results = self._run_parallel_analysis(segments, translate=translate)

        segment_results = [
            self.comparator.reconcile(hindi_result, english_result, language=lang)
            for hindi_result, english_result in zip(hindi_results, english_results)
        ]

        cross_check_summary = self.comparator.summarize(segment_results)

        if return_segments:
            # Return the complete unified output document.
            return self.output_assembler.format_full_output(
                segments=segments,
                segment_results=segment_results,
                metadata=metadata,
                segment_id=segment_id,
                lang=lang,
                lang_confidence=lang_confidence,
                translation_used=translate,
                cross_check_summary=cross_check_summary,
            )

        # Without segments: return aggregate + metadata only.
        aggregated = self.output_assembler.aggregate_results(segment_results)
        aggregated["cross_check_summary"] = cross_check_summary

        return {
            "overall_sentiment": aggregated,
            "metadata": metadata,
        }

    def process_json_output(
        self,
        text: str,
        segment_id: str = "doc_001",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process document and return the complete unified JSON output.

        Returns the same structure as
        ``process_document(return_segments=True)``:

        .. code-block:: json

            {
              "overall_sentiment": { ... },
              "metadata": { ... },
              "segments": [ ... ]
            }

        Args:
            text: Input text
            segment_id: Parent document identifier
            language: Optional language code

        Returns:
            Full unified output dict
        """
        return self.process_document(
            text=text,
            return_segments=True,
            segment_id=segment_id,
            language=language,
        )

    def save_results(self, results: Dict, output_path: str):
        """
        Save inference results to JSON file

        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        self.output_assembler.save_results(results, output_path)


class RealTimeInference:
    """
    Real-time inference for streaming or API use
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        emotion_labels: Optional[List[str]] = None,
        english_model_path: Optional[str] = None,
        comparison_strategy: str = "agreement_flag",
    ):
        self.pipeline = SentimentInferencePipeline(
            model_path=model_path,
            emotion_labels=emotion_labels,
            english_model_path=english_model_path,
            comparison_strategy=comparison_strategy,
        )

    def predict(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Single prediction (for API endpoints)

        Args:
            text: Input text
            language: Optional language code

        Returns:
            Sentiment prediction
        """
        result = self.pipeline.process_document(text, return_segments=True, language=language)
        if "error" in result:
            return result

        return result

    def predict_batch(self, texts: List[str], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Batch prediction

        Args:
            texts: List of input texts
            language: Optional language code

        Returns:
            List of predictions
        """
        predictions = []

        for text in texts:
            pred = self.predict(text, language=language)
            predictions.append(pred)

        return predictions


if __name__ == "__main__":
    import json

    # Test inference pipeline
    pipeline = SentimentInferencePipeline()

    test_text = """
    मुझे यह फिल्म बहुत पसंद है, यह बेहद शानदार और प्रेरणादायक है।
    मैं इस फिल्म को लेकर बहुत खुश हूं और इसे सभी को देखना चाहिए।
    """

    # Process document — returns full unified output with formatted segments
    result = pipeline.process_document(
        test_text,
        return_segments=True,
        segment_id="doc_001",
    )

    print("\nInference Results:")
    print(json.dumps(result, ensure_ascii=False, indent=2))