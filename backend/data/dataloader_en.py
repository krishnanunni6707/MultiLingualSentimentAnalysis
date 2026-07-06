"""
data/dataloader_en.py

Batching / DataLoader construction for the English GoEmotions dataset.
Reuses the shared Hindi collate function so training code stays generic.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from config import TrainingConfig
from data.dataset_en import GoEmotionsDataset
from data.dataloader import CollateFunction
from models.model_utils import ModelTokenizer


def create_english_data_loaders(
    tokenizer: ModelTokenizer,
    batch_size: int = TrainingConfig.BATCH_SIZE,
    num_workers: int = TrainingConfig.NUM_WORKERS,
    pin_memory: bool = TrainingConfig.PIN_MEMORY,
    cache_dir: str | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test data loaders from GoEmotions.

    Args:
        tokenizer: Model tokenizer built with ModelConfigEN.MODEL_NAME
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory
        cache_dir: Optional HuggingFace datasets cache directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    collate_fn = CollateFunction()

    train_dataset = GoEmotionsDataset(tokenizer, split="train", cache_dir=cache_dir)
    val_dataset = GoEmotionsDataset(tokenizer, split="val", cache_dir=cache_dir)
    test_dataset = GoEmotionsDataset(tokenizer, split="test", cache_dir=cache_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader