"""
data/dataset_en.py

English multi-task sentiment dataset, built on GoEmotions (58k Reddit
comments), hosted at https://huggingface.co/datasets/google-research-datasets/go_emotions
— downloads automatically the first time this module is used, no manual
download or account needed. Requires:
    pip install datasets

GoEmotions' 27 fine-grained emotions are grouped into the SAME 6 Ekman
categories the Hindi model uses (anger, joy, sadness, fear, surprise,
disgust), so MultiTaskSentimentModel needs no architecture change to
handle English — only a different pretrained encoder (see
config.ModelConfigEN).

IMPORTANT CAVEAT: GoEmotions' public "simplified" config only gives
binary presence/absence per emotion label (not a graded intensity like
Hindi's emoIntensity column). Intensity here is therefore a placeholder
(1.0 if any mapped emotion is present, else 0.0) — much weaker than the
Hindi data's real graded signal. If you need genuine graded intensity,
GoEmotions' "raw" config exposes one row per individual rater; the
fraction of raters agreeing on a label could be aggregated into a real
0-1 intensity score, at the cost of more preprocessing (not implemented
here to keep this a working starting point rather than half-finished).
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from models.model_utils import ModelTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# GoEmotions' 27 fine-grained emotions -> Ekman 6-group mapping.
# "neutral" has no Ekman mapping and contributes only to polarity (as 0).
GOEMOTIONS_TO_EKMAN = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy",
    "pride": "joy", "admiration": "joy", "caring": "joy", "desire": "joy",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "confusion": "surprise", "curiosity": "surprise",
    "realization": "surprise",
}

EKMAN_EMOTIONS = ["anger", "joy", "sadness", "fear", "surprise", "disgust"]

# GoEmotions' own positive/negative/ambiguous grouping, used the same way
# Hindi's DEFAULT_EMOTION_POLARITY derives polarity from emotion labels.
GOEMOTIONS_SENTIMENT = {
    "admiration": 1, "amusement": 1, "approval": 1, "caring": 1, "desire": 1,
    "excitement": 1, "gratitude": 1, "joy": 1, "love": 1, "optimism": 1,
    "pride": 1, "relief": 1,
    "anger": -1, "annoyance": -1, "disappointment": -1, "disapproval": -1,
    "disgust": -1, "embarrassment": -1, "fear": -1, "grief": -1,
    "nervousness": -1, "remorse": -1, "sadness": -1,
    "confusion": 0, "curiosity": 0, "realization": 0, "surprise": 0, "neutral": 0,
}


class GoEmotionsDataset(Dataset):
    """
    English multi-task sentiment dataset from GoEmotions, shaped to match
    HindiSentimentCSVDataset's __getitem__ output exactly (same keys),
    so the same Trainer/DataLoader/collate code works unchanged.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            tokenizer: Model tokenizer (use one built with ModelConfigEN.MODEL_NAME)
            split: "train", "val", or "test" (mapped to GoEmotions' own
                official train/validation/test split)
            cache_dir: Optional HuggingFace datasets cache directory
        """
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' library is required for GoEmotionsDataset. "
                "Install with: pip install datasets"
            )

        self.tokenizer = tokenizer
        self.split = split

        hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
        hf_dataset = load_dataset(
            "google-research-datasets/go_emotions", "simplified", cache_dir=cache_dir
        )[hf_split]

        self.records = self._parse_records(hf_dataset)
        logger.info(f"Loaded {len(self.records)} GoEmotions samples for {split} split")

    def _parse_records(self, hf_dataset) -> List[Dict]:
        emotion_names = hf_dataset.features["labels"].feature.names  # 28 names incl. "neutral"
        records = []

        for example in hf_dataset:
            text = example["text"]
            labels = [emotion_names[i] for i in example["labels"]]

            emotion_vec = torch.zeros(len(EKMAN_EMOTIONS))
            polarity_sum = 0.0
            polarity_weight = 0

            for label in labels:
                ekman_group = GOEMOTIONS_TO_EKMAN.get(label)
                if ekman_group:
                    idx = EKMAN_EMOTIONS.index(ekman_group)
                    emotion_vec[idx] = 1.0  # binary presence (see module caveat above)

                polarity_sum += GOEMOTIONS_SENTIMENT.get(label, 0)
                polarity_weight += 1

            polarity = polarity_sum / polarity_weight if polarity_weight > 0 else 0.0
            intensity = float(emotion_vec.max())  # placeholder — see module docstring

            records.append({
                "text": text,
                "emotion_labels": emotion_vec,
                "polarity": polarity,
                "intensity": intensity,
            })

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.records[idx]

        if not sample["text"]:
            raise ValueError(f"Missing text in sample {idx}")

        encoding = self.tokenizer.encode(sample["text"])

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "emotion_labels": sample["emotion_labels"],
            "polarity_labels": torch.tensor([sample["polarity"]], dtype=torch.float),
            "intensity_labels": torch.tensor([sample["intensity"]], dtype=torch.float),
        }