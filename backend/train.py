"""
Complete training script for Hindi Sentiment Analysis Model
"""

import torch
import argparse
from pathlib import Path
from typing import Optional

from config import (
    TrainingConfig, ModelConfig, CHECKPOINT_DIR, LOG_DIR, DataConfig
)
from models.sentiment_model import MultiTaskSentimentModel
from models.model_utils import ModelTokenizer, load_model
from data.dataloader import DEFAULT_DATA_FILE, create_data_loaders
from training.trainer import Trainer
from utils.logger import setup_logger, get_logger

# Configure logging (writes to logs/training.log + console, via shared logger).
# force=True is required: importing training.trainer above triggers get_logger()
# inside sub-modules which calls setup_logger() with no log_file and sets
# _ROOT_CONFIGURED=True. Without force, this call is silently skipped.
setup_logger(log_file=str(LOG_DIR / "training.log"), force=True)
logger = get_logger(__name__)


def setup_device() -> str:
    """Setup training device"""
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
    data_file: str,
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    checkpoint: Optional[str] = None
):
    """
    Train the Hindi sentiment analysis model

    Args:
        data_file: Path to training data (CSV file, list of CSVs, or a
            directory of CSVs — see data.dataset.HindiSentimentCSVDataset)
        output_dir: Directory to save checkpoints
        num_epochs: Number of epochs to train
        batch_size: Training batch size
        learning_rate: Learning rate
        checkpoint: Path to checkpoint to resume from
    """

    # Setup
    device_str = setup_device()
    device = torch.device(device_str)

    if output_dir is None:
        output_dir = CHECKPOINT_DIR
    output_dir = Path(output_dir)

    # Override config if specified
    if num_epochs:
        TrainingConfig.NUM_EPOCHS = num_epochs
    if batch_size:
        TrainingConfig.BATCH_SIZE = batch_size
    if learning_rate:
        TrainingConfig.LEARNING_RATE = learning_rate

    logger.info("=" * 80)
    logger.info("Hindi Sentiment Analysis - Training")
    logger.info("=" * 80)

    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model: {ModelConfig.MODEL_NAME}")
    logger.info(f"  Epochs: {TrainingConfig.NUM_EPOCHS}")
    logger.info(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {TrainingConfig.LEARNING_RATE}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Text log file:    {LOG_DIR / 'training.log'}")
    logger.info(f"  Metrics CSV:      {output_dir / 'training_metrics.csv'}\n")

    # Create tokenizer first, then data loaders so we can derive the label set
    logger.info("Loading tokenizer...")
    tokenizer = ModelTokenizer(ModelConfig.MODEL_NAME)

    logger.info(f"Loading data from {data_file}...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_file,
            tokenizer=tokenizer,
            batch_size=TrainingConfig.BATCH_SIZE,
            num_workers=TrainingConfig.NUM_WORKERS
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    emotion_labels = list(getattr(train_loader.dataset, "emotion_vocab", []))
    if not emotion_labels:
        emotion_labels = list(ModelConfig.DEFAULT_EMOTION_LABELS)

    logger.info(f"Discovered emotion labels: {emotion_labels}")

    # Create model after the dataset vocab is known so the head size matches.
    logger.info("Loading model...")
    if checkpoint:
        model, _, checkpoint_labels = load_model(
            checkpoint_path=checkpoint,
            emotion_labels=emotion_labels,
            model_name=ModelConfig.MODEL_NAME,
        )
        if checkpoint_labels:
            emotion_labels = checkpoint_labels
    else:
        model = MultiTaskSentimentModel(ModelConfig.MODEL_NAME, num_emotions=len(emotion_labels))

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}\n")

    # Create trainer
    # NOTE: emotion_names=EMOTION_LABELS assumes the model's 6 emotion
    # outputs correspond to the fixed Ekman set defined in
    # inference/pipeline.py. If your training data uses a different/larger
    # emotion vocabulary (see HindiSentimentCSVDataset.emotion_vocab), this
    # will mislabel per-class metrics — pass the dataset's real emotion_vocab
    # here instead once that's wired up.
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

    # Train
    logger.info("Starting training...\n")
    try:
        trainer.train(num_epochs=TrainingConfig.NUM_EPOCHS)
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"  Best model saved to: {output_dir / 'best_model.pt'}")
        logger.info(f"  Logs saved to: {LOG_DIR / 'training.log'}")

    except KeyboardInterrupt:
        logger.info("\n! Training interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}", exc_info=True)
        return


def evaluate(
    data_file: str,
    checkpoint: str,
    device: str = "cuda"
):
    """
    Evaluate model on test set

    Args:
        data_file: Path to data file (CSV, list of CSVs, or a directory of CSVs)
        checkpoint: Path to model checkpoint
        device: Device to use ("cuda" or "cpu"; falls back to "cpu" if CUDA
            is requested but unavailable)
    """
    device_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device_str != device:
        logger.warning("CUDA not available, falling back to CPU")
    torch_device = torch.device(device_str)

    logger.info("=" * 80)
    logger.info("Hindi Sentiment Analysis - Evaluation")
    logger.info("=" * 80)

    # Load tokenizer
    tokenizer = ModelTokenizer()

    # Load test data
    logger.info(f"Loading test data from {data_file}...")
    _, _, test_loader = create_data_loaders(
        data_file,
        tokenizer=tokenizer,
        batch_size=TrainingConfig.BATCH_SIZE
    )

    emotion_labels = list(getattr(test_loader.dataset, "emotion_vocab", []))
    if not emotion_labels:
        emotion_labels = list(ModelConfig.DEFAULT_EMOTION_LABELS)

    # Load model after the dataset vocab is known.
    logger.info(f"Loading model from {checkpoint}...")
    model, _, checkpoint_labels = load_model(
        checkpoint_path=checkpoint,
        emotion_labels=emotion_labels,
        model_name=ModelConfig.MODEL_NAME,
    )
    if checkpoint_labels:
        emotion_labels = checkpoint_labels

    # Evaluate — reuse Trainer.evaluate() (loss computation + metrics)
    # rather than duplicating that batch loop here. train_loader is set to
    # test_loader since we never call trainer.train()/train_epoch() in this
    # path; only .evaluate() is used.
    logger.info("Running evaluation...\n")
    trainer = Trainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        emotion_names=emotion_labels,
        emotion_labels=emotion_labels,
        device=torch_device,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    results = trainer.evaluate(test_loader, desc="test")

    logger.info("\nTest Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Hindi Sentiment Analysis Training Script"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "data_file",
        type=str,
        nargs="?",
        default=DEFAULT_DATA_FILE,
        help="Path to training data (CSV file, list of CSVs, or a directory of CSVs)"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "data_file",
        type=str,
        help="Path to data file"
    )
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_file=args.data_file,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint=args.checkpoint
        )
    elif args.command == "evaluate":
        evaluate(
            data_file=args.data_file,
            checkpoint=args.checkpoint,
            device=args.device
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()