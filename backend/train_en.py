"""
Complete training script for English Sentiment Analysis Model.

Requires: pip install datasets
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import CHECKPOINT_DIR, LOG_DIR, ModelConfigEN, TrainingConfig
from data.dataloader_en import create_english_data_loaders
from inference.pipeline_en import EMOTION_LABELS_EN
from models.model_utils import ModelTokenizer, load_model
from models.sentiment_model import MultiTaskSentimentModel
from training.trainer import Trainer
from utils.logger import get_logger, setup_logger


# Configure logging (writes to logs/training_en.log + console, via shared logger)
setup_logger(log_file=str(LOG_DIR / "training_en.log"))
logger = get_logger(__name__)


def setup_device() -> str:
    """Setup training device."""
    device = TrainingConfig.DEVICE

    if device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available, using CPU")
            device = "cpu"
    else:
        logger.info("Using CPU")

    return device


def train(
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    checkpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """
    Train the English sentiment analysis model.

    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train
        batch_size: Training batch size
        learning_rate: Learning rate
        checkpoint: Path to checkpoint to resume from
        cache_dir: HuggingFace cache directory for GoEmotions downloads
    """

    device_str = setup_device()
    device = torch.device(device_str)

    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "en"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_epochs:
        TrainingConfig.NUM_EPOCHS = num_epochs
    if batch_size:
        TrainingConfig.BATCH_SIZE = batch_size
    if learning_rate:
        TrainingConfig.LEARNING_RATE = learning_rate

    logger.info("=" * 80)
    logger.info("English Sentiment Analysis - Training")
    logger.info("=" * 80)

    logger.info("\nConfiguration:")
    logger.info(f"  Model: {ModelConfigEN.MODEL_NAME}")
    logger.info(f"  Epochs: {TrainingConfig.NUM_EPOCHS}")
    logger.info(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {TrainingConfig.LEARNING_RATE}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output Directory: {output_dir}")
    if cache_dir:
        logger.info(f"  Cache Directory: {cache_dir}")
    logger.info("")

    logger.info("Loading tokenizer...")
    tokenizer = ModelTokenizer(ModelConfigEN.MODEL_NAME)

    logger.info("Loading GoEmotions data...")
    try:
        train_loader, val_loader, test_loader = create_english_data_loaders(
            tokenizer=tokenizer,
            batch_size=TrainingConfig.BATCH_SIZE,
            num_workers=TrainingConfig.NUM_WORKERS,
            pin_memory=TrainingConfig.PIN_MEMORY,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    emotion_labels = list(EMOTION_LABELS_EN)
    logger.info(f"Using emotion labels: {emotion_labels}")

    logger.info("Loading model...")
    if checkpoint:
        model, _, checkpoint_labels = load_model(
            checkpoint_path=checkpoint,
            emotion_labels=emotion_labels,
            model_name=ModelConfigEN.MODEL_NAME,
        )
        if checkpoint_labels:
            emotion_labels = checkpoint_labels
    else:
        model = MultiTaskSentimentModel(
            ModelConfigEN.MODEL_NAME,
            num_emotions=len(emotion_labels),
        )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}\n")

    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        emotion_names=emotion_labels,
        emotion_labels=emotion_labels,
        device=device,
        checkpoint_dir=str(output_dir),
    )

    logger.info("Starting training...\n")
    try:
        trainer.train(num_epochs=TrainingConfig.NUM_EPOCHS)
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"  Best model saved to: {output_dir / 'best_model.pt'}")
        logger.info(f"  Logs saved to: {LOG_DIR / 'training_en.log'}")

    except KeyboardInterrupt:
        logger.info("\n! Training interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}", exc_info=True)
        return


def evaluate(
    checkpoint: str,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
):
    """
    Evaluate model on the GoEmotions test split.

    Args:
        checkpoint: Path to model checkpoint
        device: Device to use ("cuda" or "cpu"; falls back to "cpu" if CUDA
            is requested but unavailable)
        cache_dir: HuggingFace cache directory for GoEmotions downloads
    """
    device_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device_str != device:
        logger.warning("CUDA not available, falling back to CPU")
    torch_device = torch.device(device_str)

    logger.info("=" * 80)
    logger.info("English Sentiment Analysis - Evaluation")
    logger.info("=" * 80)

    logger.info("Loading tokenizer...")
    tokenizer = ModelTokenizer(ModelConfigEN.MODEL_NAME)

    logger.info("Loading GoEmotions test data...")
    try:
        _, _, test_loader = create_english_data_loaders(
            tokenizer=tokenizer,
            batch_size=TrainingConfig.BATCH_SIZE,
            num_workers=TrainingConfig.NUM_WORKERS,
            pin_memory=TrainingConfig.PIN_MEMORY,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    emotion_labels = list(EMOTION_LABELS_EN)

    logger.info(f"Loading model from {checkpoint}...")
    model, _, checkpoint_labels = load_model(
        checkpoint_path=checkpoint,
        emotion_labels=emotion_labels,
        model_name=ModelConfigEN.MODEL_NAME,
    )
    if checkpoint_labels:
        emotion_labels = checkpoint_labels

    logger.info("Running evaluation...\n")
    trainer = Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        emotion_names=emotion_labels,
        emotion_labels=emotion_labels,
        device=torch_device,
        checkpoint_dir=str(CHECKPOINT_DIR / "en"),
    )

    results = trainer.evaluate(test_loader, desc="test")

    logger.info("\nTest Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="English Sentiment Analysis Training Script"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume from",
    )
    train_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for the GoEmotions dataset",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    eval_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for the GoEmotions dataset",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
        )
    elif args.command == "evaluate":
        evaluate(
            checkpoint=args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()