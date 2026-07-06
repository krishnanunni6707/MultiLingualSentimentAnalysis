"""
FastAPI main application for Hindi Sentiment Analysis
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch

from config import APIConfig, LOG_DIR
from api.routes import router
from api.middleware import configure_api_middleware
from utils.logger import setup_logger, get_logger

# Configure logging (writes to logs/api.log + console, via shared logger)
setup_logger(log_file=str(LOG_DIR / "api.log"))
logger = get_logger(__name__)


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown
    """
    # Startup
    logger.info("Starting up Hindi Sentiment Analysis API...")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0)}")

    yield

    # Shutdown
    logger.info("Shutting down Hindi Sentiment Analysis API...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Create FastAPI App ====================

app = FastAPI(
    title="Hindi Sentiment Analysis API",
    description="Multi-task sentiment analysis for Hindi text using IndicBERT",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== Middleware ====================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_api_middleware(app)


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle global exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url),
            "method": request.method
        }
    )


# ==================== Root Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Hindi Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/info")
async def info():
    """API information"""
    return {
        "title": "Hindi Sentiment Analysis API",
        "description": "Multi-task sentiment analysis for Hindi text",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "analyze_detailed": "/api/v1/analyze_detailed",
            "analyze_json": "/api/v1/analyze_json",
            "batch_analyze": "/api/v1/batch_analyze",
            "emotions": "/api/v1/emotions",
            "stats": "/api/v1/stats"
        },
        "documentation": "/docs"
    }


# ==================== Include Routes ====================

app.include_router(router)


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        workers=APIConfig.WORKERS,
        reload=APIConfig.RELOAD,
        log_level="info"
    )