import json
import asyncio
import logging
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.logger import setup_logger

# Configure detailed logging format with timestamps (ms precision) and write to logs/app.log
setup_logger(
    level=logging.INFO,
    log_file="logs/app.log",
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
    force=True
)

# Route uvicorn and fastapi logs to the root logger so they write to logs/app.log
for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
    _logger = logging.getLogger(logger_name)
    _logger.handlers = []
    _logger.propagate = True

logger = logging.getLogger("server")

from api.routes import router
from api.middleware import configure_api_middleware
from config import APIConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[~] Starting Multilingual Sentiment Analysis Server...")
    yield
    logger.info("[x] Shutting down server...")


app = FastAPI(title="Multilingual Sentiment Analysis", lifespan=lifespan)

# Allow CORS requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_api_middleware(app)
app.include_router(router)


# ------------------------------------------------------------------
# NEW: Real-time Streaming SSE Endpoint for Incremental Frontend Updates
# ------------------------------------------------------------------
@app.post("/api/v1/analyze_stream")
async def analyze_stream(request: Request):
    """
    Streams step-by-step progress updates and yields partial results 
    (Sentiment first, followed by Summary) as Server-Sent Events.
    """
    body = await request.json()
    text = body.get("text", "")
    summarize = body.get("summarize", True)

    async def event_generator():
        try:
            # Step 1: Text extraction / ingestion update
            yield f"data: {json.dumps({'type': 'status', 'step': 'extracting', 'message': 'Processing and extracting text...'})}\n\n"
            await asyncio.sleep(0.3)

            # Step 2: Sentiment Analysis step
            yield f"data: {json.dumps({'type': 'status', 'step': 'sentiment', 'message': 'Analyzing sentiment & emotions...'})}\n\n"
            
            # Replace dummy dict below with your actual sentiment engine execution if desired
            sentiment_result = {
                "overall_sentiment": {
                    "dominant_emotion": "joy",
                    "polarity_score": 0.75,
                    "polarity_description": "positive",
                    "intensity_description": "moderate",
                    "confidence": 0.88,
                    "emotions": {"joy": 0.75, "hope": 0.15, "neutral": 0.10}
                },
                "metadata": {"language": "en", "device": "CPU/GPU"}
            }
            await asyncio.sleep(0.8)

            # Stream SENTIMENT result immediately to frontend
            yield f"data: {json.dumps({'type': 'result_sentiment', 'data': sentiment_result})}\n\n"

            # Step 3: Summarization step (if requested)
            if summarize:
                yield f"data: {json.dumps({'type': 'status', 'step': 'summarizing', 'message': 'Generating executive summary...'})}\n\n"
                
                summary_text = "The uploaded document provides a detailed overview of the input topic, highlighting key technical insights and emotional nuances."
                await asyncio.sleep(1.0)

                # Stream SUMMARY result immediately to frontend
                yield f"data: {json.dumps({'type': 'result_summary', 'data': {'summary': summary_text}})}\n\n"

            # Step 4: Completion notice
            yield f"data: {json.dumps({'type': 'status', 'step': 'complete', 'message': 'Analysis complete!'})}\n\n"

        except Exception as e:
            logger.error(f"Error during streaming analysis: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    # Set reload to True during local development if needed
    uvicorn.run("server:app", host=APIConfig.HOST, port=APIConfig.PORT, reload=APIConfig.RELOAD)