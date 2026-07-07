"""
data/dataloader.py

Batching / DataLoader construction for the Hindi sentiment dataset.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

from config import TrainingConfig
from models.model_utils import ModelTokenizer
from data.dataset import HindiSentimentCSVDataset

DEFAULT_DATA_FILE = "data/EmoInHindi/LREC_EmoInHindi_reduced.csv"


class CollateFunction:
    """Custom collate function for batching samples from HindiSentimentCSVDataset"""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([s["input_ids"] for s in batch]),
            "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
            "emotion_labels": torch.stack([s["emotion_labels"] for s in batch]),
            "polarity_labels": torch.stack([s["polarity_labels"] for s in batch]),
            "intensity_labels": torch.stack([s["intensity_labels"] for s in batch]),
        }


def create_data_loaders(
    data_file: Union[str, List[str]] = DEFAULT_DATA_FILE,
    *,
    tokenizer: ModelTokenizer,
    batch_size: int = TrainingConfig.BATCH_SIZE,
    num_workers: int = TrainingConfig.NUM_WORKERS,
    pin_memory: bool = TrainingConfig.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test data loaders from CSV data.

    Args:
        data_file: Path to a single CSV, a list of CSV paths, or a directory
            (e.g. "data/raw") whose "*.csv" files will be auto-discovered and
            combined
        tokenizer: Model tokenizer
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    collate_fn = CollateFunction()

    train_dataset = HindiSentimentCSVDataset(data_file, tokenizer, split="train")
    val_dataset = HindiSentimentCSVDataset(data_file, tokenizer, split="val")
    test_dataset = HindiSentimentCSVDataset(data_file, tokenizer, split="test")

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


if __name__ == "__main__":
    tokenizer = ModelTokenizer()
    train_loader, val_loader, test_loader = create_data_loaders(tokenizer=tokenizer)

    batch = next(iter(train_loader))
    print("Batch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")