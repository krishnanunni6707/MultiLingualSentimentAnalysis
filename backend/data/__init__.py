"""
data package

Exposes the dataset class, collate function, and data loader factory
for Hindi sentiment analysis.
"""

from data.dataset import HindiSentimentCSVDataset, DEFAULT_EMOTION_POLARITY
from data.dataloader import CollateFunction, create_data_loaders
from data.dataset_en import GoEmotionsDataset
from data.dataloader_en import create_english_data_loaders

__all__ = [
    "HindiSentimentCSVDataset",
    "GoEmotionsDataset",
    "DEFAULT_EMOTION_POLARITY",
    "CollateFunction",
    "create_data_loaders",
    "create_english_data_loaders",
]