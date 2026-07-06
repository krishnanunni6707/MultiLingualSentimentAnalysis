"""
FastAPI routes for sentiment analysis API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from inference.pipeline import SentimentInferencePipeline, RealTimeInference
from config import APIConfig
from api.schemas import (
    BatchRequest,
    DocumentSentimentResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)

logger = logging.getLogger(__name__)

# Initialize inference
inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)

# Create router
router = APIRouter(prefix="/api/v1", tags=["sentiment"])


# ==================== Endpoints ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "MultiTaskSentimentModel",
        "device": str(inference.pipeline.device)
    }


@router.post("/analyze")
async def analyze_sentiment(request: SentimentRequest) -> DocumentSentimentResponse:
    """
    Analyze sentiment of a Hindi/English text
    
    Args:
        request: SentimentRequest with text and options
        
    Returns:
        DocumentSentimentResponse with sentiment analysis results
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Run inference
        result = inference.predict(request.text, language=request.language)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze_detailed")
async def analyze_sentiment_detailed(request: SentimentRequest) -> DocumentSentimentResponse:
    """
    Analyze sentiment with detailed segment-level information.

    Returns the full unified output:

    - ``overall_sentiment`` — aggregated emotions, polarity, intensity,
      confidence, and cross-check summary
    - ``metadata`` — language, lengths, segmentation strategy
    - ``segments`` — per-segment dicts with ``segment_id``, ``text_span``,
      ``sentiment``, ``emotion_tags``, ``language_detected``,
      ``translation_used``, and ``cross_check``
    """
    try:
        pipeline = inference.pipeline
        result = pipeline.process_document(
            request.text,
            return_segments=True,
            language=request.language,
        )

        return result

    except Exception as e:
        logger.error(f"Error in analyze_sentiment_detailed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze_json")
async def analyze_sentiment_json(request: SentimentRequest) -> DocumentSentimentResponse:
    """
    Analyze sentiment and return the full unified JSON document.

    Matches the output format:

    .. code-block:: json

        {
          "overall_sentiment": {
            "emotions": { "anger": 0.12, "joy": 0.64, ... },
            "dominant_emotion": "joy",
            "polarity_score": 0.71,
            "polarity_description": "Positive",
            "intensity": 0.78,
            "intensity_description": "Strong",
            "confidence": 0.89,
            "num_segments": 4,
            "cross_check_summary": { ... }
          },
          "metadata": {
            "language": "hi",
            "language_confidence": 0.98,
            "original_length": 1240,
            "cleaned_length": 1188,
            "num_segments": 4,
            "segment_strategy": "sentence"
          },
          "segments": [
            {
              "segment_id": "doc_001_seg_000",
              "text_span": "char_span: 0-312",
              "sentiment": { "label": "joy", "polarity_score": 0.82, ... },
              "emotion_tags": ["joy", "surprise"],
              "language_detected": "hi",
              "translation_used": true,
              "cross_check": { ... }
            }
          ]
        }
    """
    try:
        pipeline = inference.pipeline
        result = pipeline.process_json_output(request.text, language=request.language)

        return result

    except Exception as e:
        logger.error(f"Error in analyze_sentiment_json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch_analyze")
async def batch_analyze(request: BatchRequest) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for multiple texts in batch
    
    Args:
        request: BatchRequest with list of texts
        
    Returns:
        List of sentiment analyses
    """
    try:
        results = inference.predict_batch(request.texts)
        return results
    
    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/emotions")
async def get_emotion_labels() -> Dict[str, List[str]]:
    """
    Get list of supported emotion labels
    
    Returns:
        Dictionary with emotion labels
    """
    return {
        "emotions": inference.pipeline.emotion_labels,
        "emotion_count": len(inference.pipeline.emotion_labels)
    }


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get model and API statistics
    
    Returns:
        Statistics dictionary
    """
    return {
        "model_name": "MultiTaskSentimentModel",
        "base_model": inference.pipeline.model.encoder.config.name_or_path,
        "tasks": ["emotion_classification", "polarity_regression", "intensity_regression"],
        "emotions_supported": len(inference.pipeline.emotion_labels),
        "max_sequence_length": 512,
        "api_version": "v1",
        "language": "Hindi (hi)"
    }


# ==================== Error Handlers ====================

@router.post("/analyze_stream")
async def analyze_sentiment_stream(request: SentimentRequest):
    """
    Stream sentiment analysis results (for long documents)
    Uses Server-Sent Events for streaming
    
    Args:
        request: SentimentRequest with text
        
    Returns:
        Streaming response with segment results
    """
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        try:
            pipeline = inference.pipeline
            segments, metadata = pipeline.preprocess_text(request.text)
            
            # Yield metadata first
            yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
            
            # Yield segment results as they complete
            for i in range(0, len(segments), pipeline.batch_size):
                batch_segments = segments[i:i + pipeline.batch_size]
                batch_results = pipeline.batch_inference(batch_segments)
                
                for result in batch_results:
                    yield f"data: {json.dumps({'type': 'segment', 'data': result})}\n\n"
            
            # Yield completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    # This module is imported by main.py
    pass
