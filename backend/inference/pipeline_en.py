"""
inference/pipeline_en.py

Inference pipeline for English sentiment analysis (GoEmotions-trained
model). A parallel counterpart to inference/pipeline.py's Hindi pipeline
— kept as its own module rather than a language switch, so each
language's pipeline can evolve independently without risk of one
language's changes breaking the other.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from config import InferenceConfig, ModelConfigEN
from models.model_utils import load_model
from utils.preprocessing import LanguageDetector
from utils.logger import get_logger
from inference.segmentation import DocumentSegmenter
from inference.output_assembler import OutputAssembler

logger = get_logger(__name__)

# Same 6-category Ekman set as the Hindi pipeline's EMOTION_LABELS.
# GoEmotions' 27 fine-grained emotions are grouped into these categories
# by data.dataset_en.GOEMOTIONS_TO_EKMAN, so the model architecture is
# identical to the Hindi model — only the pretrained encoder differs
# (see config.ModelConfigEN).
EMOTION_LABELS_EN = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise"
]


class SentimentInferencePipelineEN:
    """
    Complete inference pipeline for English sentiment analysis (lazy-loaded).
    Model loads on first inference call to save memory.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = InferenceConfig.BATCH_SIZE,
        emotion_labels: List[str] = EMOTION_LABELS_EN,
    ):
        """
        Initialize inference pipeline (lazy-loads model on first use)

        Args:
            model_path: Path to model checkpoint (trained via
                data.dataloader_en.create_english_data_loaders)
            device: Device to run inference on
            batch_size: Batch size for inference
            emotion_labels: Ordered emotion label names matching the model's
                emotion output dimension
        """
        self.device = device
        self.batch_size = batch_size
        self.emotion_labels = emotion_labels

        # Lazy load: store config, don't load model yet 
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._model_path = model_path
        self._device = device

        # Segmentation + output assembly
        self.segmenter = DocumentSegmenter(language="en")
        self.output_assembler = OutputAssembler(emotion_labels=self.emotion_labels)

    def _ensure_loaded(self):
        """Load model and tokenizer on first use (lazy loading)."""
        if self._loaded:
            return  # Already loaded, skip
        
        logger.info(f"[Lazy Load] Loading English sentiment model..")
        print(f"[~] Loading English sentiment model...")
        
        try:
            # Load model and tokenizer using the shared checkpoint loader
            self.model, self.tokenizer, resolved_labels = load_model(
                checkpoint_path=self._model_path,
                emotion_labels=None,
                model_name=ModelConfigEN.MODEL_NAME,
            )
            # model.id2label is the canonical source
            self.emotion_labels = resolved_labels
            
            self.model.to(self._device)
            self.model.eval()
            self._loaded = True
            logger.info(f"English sentiment model loaded on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load English model: {str(e)}")
            raise

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
        self._ensure_loaded()  # Load on first call
        
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
            # Decode using the model's own id2label — the authoritative
            # index→label mapping derived directly from the checkpoint.
            id2label = self.model.id2label

            emotion_dict = {
                id2label[j]: emotion_probs[i, j].item()
                for j in range(emotion_probs.shape[1])
            }

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

    def process_document(
        self,
        text: str,
        return_segments: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete document

        Args:
            text: Full document text
            return_segments: Whether to return individual segment results

        Returns:
            Dictionary with overall sentiment and optionally segment details
        """
        # Preprocess and segment
        segments, metadata = self.preprocess_text(text)

        if not segments:
            return {
                "error": "No valid segments found",
                "metadata": metadata
            }

        segment_results = []

        # Process segments in batches
        for i in tqdm(range(0, len(segments), self.batch_size)):
            batch_segments = segments[i:i + self.batch_size]
            batch_results = self.batch_inference(batch_segments)
            segment_results.extend(batch_results)

        # Aggregate results
        aggregated = self.output_assembler.aggregate_results(segment_results)

        output = {
            "overall_sentiment": aggregated,
            "metadata": metadata
        }

        if return_segments:
            output["segments"] = segment_results

        return output

    def process_json_output(
        self,
        text: str,
        segment_id: str = "doc_001"
    ) -> List[Dict[str, Any]]:
        """
        Process document and return in the standard JSON format

        Args:
            text: Input text
            segment_id: Identifier for segment

        Returns:
            List of segment outputs in JSON format
        """
        segments, metadata = self.preprocess_text(text)

        # Language detection
        lang, lang_confidence = LanguageDetector.detect_language(text)
        translation_used = lang != "en"

        segment_results = [self.batch_inference([segment])[0] for segment in segments]

        return self.output_assembler.format_json_output(
            segments=segments,
            segment_results=segment_results,
            segment_id=segment_id,
            lang=lang,
            translation_used=translation_used,
        )

    def save_results(self, results: Dict, output_path: str):
        """
        Save inference results to JSON file

        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        self.output_assembler.save_results(results, output_path)


class RealTimeInferenceEN:
    """
    Real-time English inference for streaming or API use
    """

    def __init__(self, model_path: Optional[str] = None):
        self.pipeline = SentimentInferencePipelineEN(model_path=model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Single prediction (for API endpoints)

        Args:
            text: Input text

        Returns:
            Sentiment prediction
        """
        segments, metadata = self.pipeline.preprocess_text(text)

        if not segments:
            return {"error": "Invalid input"}

        results = self.pipeline.batch_inference(segments)
        aggregated = self.pipeline.output_assembler.aggregate_results(results)

        return {
            "sentiment": aggregated,
            "metadata": metadata
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction

        Args:
            texts: List of input texts

        Returns:
            List of predictions
        """
        predictions = []

        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)

        return predictions


if __name__ == "__main__":
    import json

    # Test inference pipeline
    pipeline = SentimentInferencePipelineEN()

    test_text = "I absolutely loved this movie, it was so inspiring and wonderful!"

    # Process document
    result = pipeline.process_document(test_text, return_segments=True)

    print("\nInference Results:")
    print(json.dumps(result, ensure_ascii=False, indent=2))