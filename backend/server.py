import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
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


if __name__ == "__main__":
    import uvicorn
    # Set reload to True during local development if needed
    uvicorn.run("server:app", host=APIConfig.HOST, port=APIConfig.PORT, reload=APIConfig.RELOAD)
