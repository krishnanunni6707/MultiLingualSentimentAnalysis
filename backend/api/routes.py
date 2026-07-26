"""
FastAPI routes for sentiment analysis API
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import logging
import io
import pypdf
import docx

from inference.pipeline import SentimentInferencePipeline, RealTimeInference
from config import APIConfig
from api.schemas import (
    BatchRequest,
    DocumentSentimentResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
    SpeakRequest,
)
import edge_tts
import os

logger = logging.getLogger(__name__)

# Initialize inference
inference = RealTimeInference(model_path=APIConfig.MODEL_PATH)

# Create router
router = APIRouter(prefix="/api/v1", tags=["sentiment"])


# ==================== Endpoints ====================

import time
from starlette.concurrency import run_in_threadpool

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
    """
    start_time = time.perf_counter()
    text_len = len(request.text) if request.text else 0
    logger.info(f"📥 [/analyze] Received request (length={text_len} chars, lang={request.language}, summarize={request.summarize})")
    logger.info(f"📝 [/analyze] Text input content:\n--- START INPUT ---\n{request.text}\n--- END INPUT ---")

    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Run inference in thread pool to prevent blocking the FastAPI event loop
        t0 = time.perf_counter()
        result = await run_in_threadpool(
            inference.predict,
            request.text,
            language=request.language,
            summarize=request.summarize,
        )
        t_infer = time.perf_counter() - t0
        logger.info(f"⏱️ [/analyze] Model inference step completed in {t_infer:.3f}s")
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        total_time = time.perf_counter() - start_time
        logger.info(f"✅ [/analyze] Request processed in {total_time:.3f}s total")
        return result
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [/analyze] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze_file")
async def analyze_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    summarize: bool = Form(False),
) -> DocumentSentimentResponse:
    """
    Extract text from uploaded .txt, .pdf, or .docx file and perform sentiment analysis.
    """
    start_time = time.perf_counter()
    filename = file.filename.lower()
    logger.info(f"📥 [/analyze_file] Received file '{file.filename}' (summarize={summarize})")
    text = ""

    try:
        t0 = time.perf_counter()
        content = await file.read()
        t_read = time.perf_counter() - t0
        logger.info(f"📄 [/analyze_file] File read into memory ({len(content)} bytes) in {t_read:.3f}s")
        
        t_extract_start = time.perf_counter()
        if filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
            
        elif filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
            
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif filename.endswith(".doc"):
            try:
                text = content.decode("utf-8", errors="ignore")
                text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
            except Exception:
                raise HTTPException(status_code=400, detail="Legacy .doc format extraction failed. Please convert to .docx or .pdf.")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .txt, .pdf, or .docx")

        t_extract = time.perf_counter() - t_extract_start
        logger.info(f"🔍 [/analyze_file] Text extraction from {filename} completed in {t_extract:.3f}s (extracted {len(text)} chars)")
        logger.info(f"📝 [/analyze_file] Extracted text content:\n--- START EXTRACTED TEXT ---\n{text}\n--- END EXTRACTED TEXT ---")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the file")

        # Run inference in thread pool to prevent blocking main event loop
        t_infer_start = time.perf_counter()
        result = await run_in_threadpool(
            inference.predict,
            text,
            language=language,
            summarize=summarize,
        )
        t_infer = time.perf_counter() - t_infer_start
        logger.info(f"⏱️ [/analyze_file] Model inference step completed in {t_infer:.3f}s")
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        total_time = time.perf_counter() - start_time
        logger.info(f"✅ [/analyze_file] File request processed in {total_time:.3f}s total")
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [/analyze_file] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


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


#==============> Voice Configurations <===========
OUTPUT_FILE = "temp_voice_output.mp3"

async def Speak(text, VOICE):
    com = edge_tts.Communicate(text, VOICE)
    await com.save(OUTPUT_FILE)
    os.system(f"mpv --no-video {OUTPUT_FILE}")
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)


@router.post("/speak")
async def speak_text(request: SpeakRequest):
    """
    Read aloud the text using edge_tts
    """
    lang = request.language.lower().strip()
    logger.info(f"🎙️ [/speak] Speech request received for language: {lang}")
    
    if lang == "hi" or "hindi" in lang:
        VOICE = "hi-IN-MadhurNeural"
    elif lang == "en" or "english" in lang:
        VOICE = "en-US-AndrewNeural"
    else:
        VOICE = "en-US-AndrewNeural"
        
    try:
        await Speak(request.text, VOICE)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"❌ [/speak] Error in Speak: {e}")
        raise HTTPException(status_code=500, detail=f"Speech failed: {str(e)}")


if __name__ == "__main__":
    # This module is imported by main.py
    pass
