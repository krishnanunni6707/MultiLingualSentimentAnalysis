"""
Configuration file for Sentiment Analysis System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RAW_DATA_DIR = DATA_DIR / "raw"

for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, RAW_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class ModelConfig:
    
    
    MODEL_NAME = "distilbert-base-multilingual-cased"
    HIDDEN_SIZE = 768
    NUM_ATTENTION_HEADS = 12
    NUM_HIDDEN_LAYERS = 12
    INTERMEDIATE_SIZE = 3072
    MAX_SEQUENCE_LENGTH = 512
    VOCAB_SIZE = 30522

   
    DEFAULT_EMOTION_LABELS = ("anger", "joy", "sadness", "fear", "surprise", "disgust")
    NUM_EMOTION_LABELS = len(DEFAULT_EMOTION_LABELS)  
    EMOTION_OUTPUT_DIM = 64

    POLARITY_OUTPUT_DIM = 16  
    INTENSITY_OUTPUT_DIM = 16  

    DROPOUT_RATE = 0.1
    LAYER_NORM_EPS = 1e-12


class ModelConfigEN(ModelConfig):

    MODEL_NAME = "bert-base-uncased"

class TrainingConfig:
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 15
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0

    
    EMOTION_LOSS_WEIGHT = 5.0  # FIXED: Increased from 1.0 to handle sparse multi-label
    POLARITY_LOSS_WEIGHT = 0.8
    INTENSITY_LOSS_WEIGHT = 0.8

    
    OPTIMIZER = "adamw"  
    SCHEDULER = "linear"  

    
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    PATIENCE = 3

    
    DEVICE = "cuda"  
    NUM_WORKERS = 4
    PIN_MEMORY = True

    
    LOG_EVERY_N_STEPS = 100
    SEED = 42


class DataConfig:
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    
    DATASETS = [
        "emotion_hindi",
        "sentiment_corpora"
    ]

    
    MIN_TEXT_LENGTH = 5
    MAX_TEXT_LENGTH = 512

    
    LANGUAGE_THRESHOLD = 0.8

class InferenceConfig:
    SEGMENT_MAX_LENGTH = 512
    OVERLAP = 50  # tokens overlap for context
    BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6

    
    SEGMENTATION_STRATEGY = "sentence"  # sentence, paragraph, fixed_length

    
    INCLUDE_CONFIDENCE = True
    INCLUDE_TRANSLATION = True
    CACHE_SIZE = 100


class TranslationConfig:
    MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME", "Helsinki-NLP/opus-mt-hi-en")
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    DEVICE = os.getenv("TRANSLATION_DEVICE", "cpu")


class APIConfig:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    WORKERS = int(os.getenv("API_WORKERS", 4))
    RELOAD = os.getenv("RELOAD", "False").lower() == "true"

    
    MODEL_PATH = os.getenv("MODEL_PATH")

    
    MAX_TEXT_LENGTH = 50000
    TIMEOUT = 300


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