"""
Configuration file for Hindi Sentiment Analysis System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RAW_DATA_DIR = DATA_DIR / "raw"

# Create directories
for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, RAW_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
class ModelConfig:
    # Public multilingual base model used for training/inference.
    # DistilBERT-multilingual is smaller than XLM-R and is openly accessible.
    MODEL_NAME = "distilbert-base-multilingual-cased"
    HIDDEN_SIZE = 768
    NUM_ATTENTION_HEADS = 12
    NUM_HIDDEN_LAYERS = 12
    INTERMEDIATE_SIZE = 3072
    MAX_SEQUENCE_LENGTH = 512
    VOCAB_SIZE = 30522

    # Task-specific heads
    DEFAULT_EMOTION_LABELS = ("anger", "joy", "sadness", "fear", "surprise", "disgust")
    NUM_EMOTION_LABELS = len(DEFAULT_EMOTION_LABELS)  # legacy fallback only
    EMOTION_OUTPUT_DIM = 64

    POLARITY_OUTPUT_DIM = 16  # -1 to +1 continuous
    INTENSITY_OUTPUT_DIM = 16  # 0 to 1 continuous

    DROPOUT_RATE = 0.1
    LAYER_NORM_EPS = 1e-12


class ModelConfigEN(ModelConfig):
    """English variant of the base model configuration."""

    MODEL_NAME = "bert-base-uncased"

# Training Configuration
class TrainingConfig:
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 15
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0

    # Loss weights for multi-task learning
    EMOTION_LOSS_WEIGHT = 1.0
    POLARITY_LOSS_WEIGHT = 0.8
    INTENSITY_LOSS_WEIGHT = 0.8

    # Optimizer
    OPTIMIZER = "adamw"  # adamw, adam, sgd
    SCHEDULER = "linear"  # linear, cosine, constant

    # Validation
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    PATIENCE = 3

    # Device
    DEVICE = "cuda"  # cuda or cpu
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Logging
    LOG_EVERY_N_STEPS = 100
    SEED = 42

# Data Configuration
class DataConfig:
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # Datasets to use
    DATASETS = [
        "emotion_hindi",
        "sentiment_corpora"
    ]

    # Text preprocessing
    MIN_TEXT_LENGTH = 5
    MAX_TEXT_LENGTH = 512

    # Language detection
    LANGUAGE_THRESHOLD = 0.8

# Inference Configuration
class InferenceConfig:
    SEGMENT_MAX_LENGTH = 512
    OVERLAP = 50  # tokens overlap for context
    BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6

    # Segmentation strategy
    SEGMENTATION_STRATEGY = "sentence"  # sentence, paragraph, fixed_length

    # Output options
    INCLUDE_CONFIDENCE = True
    INCLUDE_TRANSLATION = True
    CACHE_SIZE = 100


class TranslationConfig:
    MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME", "Helsinki-NLP/opus-mt-hi-en")
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    DEVICE = os.getenv("TRANSLATION_DEVICE", "cpu")

# API Configuration
class APIConfig:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    WORKERS = int(os.getenv("API_WORKERS", 4))
    RELOAD = os.getenv("RELOAD", "False").lower() == "true"

    # Optional trained model checkpoint to load at API startup.
    MODEL_PATH = os.getenv("MODEL_PATH")

    # Request limits
    MAX_TEXT_LENGTH = 50000
    TIMEOUT = 300

# Combine configs for easy access
CONFIG = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "data": DataConfig,
    "inference": InferenceConfig,
    "api": APIConfig
}

if __name__ == "__main__":
    print("Model Config:", ModelConfig.__dict__)
    print("Training Config:", TrainingConfig.__dict__)