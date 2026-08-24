"""
data/dataset.py

Dataset class for Hindi sentiment analysis, loaded from dialogue-style CSV files.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


from config import DataConfig, TrainingConfig
from models.model_utils import ModelTokenizer

logger = logging.getLogger(__name__)


# Polarity assigned to each emotion label (-1 negative, +1 positive, 0 neutral/ambiguous).
DEFAULT_EMOTION_POLARITY = {
    # Original categories
    "anger": -1.0,
    "sad": -1.0,
    "sadness": -1.0,
    "fear": -1.0,
    "disgust": -1.0,
    "disgusted": -1.0,
    "shame": -1.0,
    "guilt": -1.0,
    "joy": 1.0,
    "happy": 1.0,
    "happiness": 1.0,
    "confident": 1.0,
    "trust": 1.0,
    "anticipation": 0.0,
    "surprise": 0.0,
    "neutral": 0.0,

    # Newly identified categories
    "admiring": 0.8,
    "amused": 0.6,
    "anxiety": -0.6,
    "compassion": 0.8,
    "determined": 0.4,
    "enthusiastic": 0.9,
    "frustrated": -0.7,
    "impatience": -0.4,
    "optimistic": 0.8,
    "patience": 0.5,
    "proud": 0.7,
    "realization": 0.2,
    "resentful": -0.8,
    "relieved": 0.6,
    "serene": 0.7,
    "serious": 0.0,
    "serenity": 0.7,
    "wisdom": 0.5,

    # Missing spectrum & remapped categories
    "hopeful": 0.7,
    "annoyed": -0.6,
    "apprehensive": -0.5,
    "surprised": 0.0,
    "grateful": 0.9,
    "impressed": 0.8,
    "confused": -0.2,
    "guilty": -0.7,
}

EMOTION_REMAPPING = {
    # Fear spectrum
    "anxiety": "fear",
    "apprehensive": "fear",
    
    # Anger spectrum
    "frustrated": "anger",
    "annoyed": "anger",
    "resentful": "anger",
    "impatience": "anger",
    
    # Hopeful / Positive spectrum
    "optimistic": "hopeful",
    "determined": "confident",
    "enthusiastic": "joy",
    "proud": "joy",
    "amused": "joy",
    
    # Neutral / Calm spectrum
    "serene": "neutral",
    "serenity": "neutral",
    "patience": "neutral",
    "serious": "neutral",
    
    # Impressed / Other spectrum
    "admiring": "impressed",
    "wisdom": "impressed",
    "trust": "confident",
    "realization": "surprised",
    "relieved": "grateful",
}


class HindiSentimentCSVDataset(Dataset):
    """
    Dataset for Hindi sentiment analysis loaded from dialogue-style CSV files.

    Expected CSV columns:
        dialogueId, utterance_no, authorRole, utterance, emotions, emoIntensity
    """

    def __init__(
        self,
        data_file: Union[str, List[str]],
        tokenizer: ModelTokenizer,
        max_length: int = 512,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        max_intensity: float = 3.0,
        emotion_polarity_map: Optional[Dict[str, float]] = None,
        author_role_filter: Optional[str] = "user",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.max_intensity = max_intensity
        self.emotion_polarity_map = emotion_polarity_map or DEFAULT_EMOTION_POLARITY

        # Load + filter
        raw_df = self._load_csv(data_file, author_role_filter)

        # Build stable emotion vocabulary
        self.emotion_vocab = self._build_emotion_vocab(raw_df)
        logger.info(f"Discovered {len(self.emotion_vocab)} emotion labels: {self.emotion_vocab}")

        records = self._parse_records(raw_df)

        # Split
        train_data, val_data, test_data = self._split_data(records, split_ratio, seed)

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        logger.info(f"Loaded {len(self.data)} samples for {split} split")

    def _resolve_data_files(self, data_file: Union[str, List[str]]) -> List[Path]:
        if isinstance(data_file, (list, tuple)):
            paths = [Path(p) for p in data_file]
        else:
            p = Path(data_file)
            if p.is_dir():
                paths = sorted(p.rglob("*.csv"))
                if not paths:
                    raise FileNotFoundError(f"No CSV files found under directory: {data_file}")
                logger.info(f"Auto-discovered {len(paths)} CSV file(s) under {data_file}")
            else:
                paths = [p]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

        return paths

    def _load_csv(
        self, data_file: Union[str, List[str]], author_role_filter: Optional[str]
    ) -> pd.DataFrame:
        paths = self._resolve_data_files(data_file)

        frames = []
        reference_cols = None
        for path in paths:
            df = pd.read_csv(path, encoding="utf-8")

            if reference_cols is None:
                reference_cols = set(df.columns)
            elif set(df.columns) != reference_cols:
                logger.warning(f"Column mismatch in {path}. Proceeding, but check schemas match.")

            df["_source_file"] = path.stem
            if "dialogueId" in df.columns:
                df["dialogueId"] = df["_source_file"] + "_" + df["dialogueId"].astype(str)

            frames.append(df)

        df = pd.concat(frames, ignore_index=True)

        required_cols = {"utterance", "emotions", "emoIntensity", "authorRole"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        if author_role_filter is not None:
            df = df[df["authorRole"] == author_role_filter].reset_index(drop=True)

        df = df.dropna(subset=["utterance"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows left after filtering authorRole == {author_role_filter!r}.")

        logger.info(f"Loaded {len(df)} rows from {len(paths)} file(s)")
        return df

    def _build_emotion_vocab(self, df: pd.DataFrame) -> List[str]:
        vocab = set()
        for emo_str in df["emotions"].fillna("neutral"):
            for label in str(emo_str).split(","):
                label = label.strip().lower()
                # Apply remapping to group sparse classes
                label = EMOTION_REMAPPING.get(label, label)
                if label:
                    vocab.add(label)
        return sorted(vocab)

    def _parse_emotions_and_intensity(
        self, emo_str: str, intensity_str: str
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Parse comma-separated emotion labels + comma-separated intensities.
        """
        # Safely handle unexpected NaN or float items in CSV string columns
        emo_str = str(emo_str) if pd.notna(emo_str) else "neutral"
        intensity_str = str(intensity_str) if pd.notna(intensity_str) else "1"

        labels = [l.strip().lower() for l in emo_str.split(",") if l.strip()]
        intens_raw = [i.strip() for i in intensity_str.split(",") if i.strip()]

        # If a single overall intensity number was provided for multiple emotions, duplicate it
        if len(intens_raw) == 1 and len(labels) > 1:
            intens_raw = intens_raw * len(labels)

        # Pad or truncate intensity list to precisely match label array length
        if len(intens_raw) != len(labels):
            logger.warning(
                f"Emotion/intensity length mismatch: {emo_str!r} vs {intensity_str!r}; standardizing lengths."
            )
            while len(intens_raw) < len(labels):
                intens_raw.append("1.0")
            intens_raw = intens_raw[:len(labels)]

        intens = []
        for val in intens_raw:
            try:
                intens.append(float(val))
            except ValueError:
                intens.append(1.0)

        emotion_vec = torch.zeros(len(self.emotion_vocab))
        weighted_polarity_sum = 0.0
        weight_sum = 0.0

        for label, intensity in zip(labels, intens):
            norm_intensity = min(intensity / self.max_intensity, 1.0)
            
            # Remap the class before matching vocabulary indices
            mapped_label = EMOTION_REMAPPING.get(label, label)

            if mapped_label in self.emotion_vocab:
                idx = self.emotion_vocab.index(mapped_label)
                # Use max() in case multiple raw labels collapse into the same target class
                emotion_vec[idx] = max(emotion_vec[idx], torch.tensor(norm_intensity))

            # Keep polarity lookups matching the original label
            polarity_sign = self.emotion_polarity_map.get(label, 0.0)
            if label not in self.emotion_polarity_map:
                logger.warning(f"Unmapped emotion label '{label}' — treated as neutral (0.0)")

            weighted_polarity_sum += polarity_sign * intensity
            weight_sum += intensity

        polarity = weighted_polarity_sum / weight_sum if weight_sum > 0 else 0.0
        polarity = max(-1.0, min(1.0, polarity))

        overall_intensity = max(
            (min(i / self.max_intensity, 1.0) for i in intens), default=0.0
        )

        return emotion_vec, polarity, overall_intensity

    def _parse_records(self, df: pd.DataFrame) -> List[Dict]:
        records = []
        for _, row in df.iterrows():
            emotion_vec, polarity, intensity = self._parse_emotions_and_intensity(
                row["emotions"], row["emoIntensity"]
            )
            records.append(
                {
                    "text": row["utterance"],
                    "emotion_labels": emotion_vec,
                    "polarity": polarity,
                    "intensity": intensity,
                }
            )
        return records

    def _split_data(
        self, data: List[Dict], split_ratio: Tuple[float, float, float], seed: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        train_ratio, val_ratio, test_ratio = split_ratio

        # Convert emotion labels into a 2D binary numpy matrix for multilabel stratification
        y = np.array([
            (sample["emotion_labels"].numpy() if torch.is_tensor(sample["emotion_labels"])
             else np.array(sample["emotion_labels"])) > 0
            for sample in data
        ], dtype=int)

        # Split train vs temp (val + test)
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - train_ratio, random_state=seed
        )
        train_idx, temp_idx = next(msss1.split(np.zeros(len(data)), y))

        # Split temp into val vs test
        val_test_split = val_ratio / (val_ratio + test_ratio)
        y_temp = y[temp_idx]
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - val_test_split, random_state=seed
        )
        val_sub_idx, test_sub_idx = next(msss2.split(np.zeros(len(temp_idx)), y_temp))

        val_idx = [temp_idx[i] for i in val_sub_idx]
        test_idx = [temp_idx[i] for i in test_sub_idx]

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        test_data = [data[i] for i in test_idx]

        return train_data, val_data, test_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        text = sample["text"]
        
        if not text:
            raise ValueError(f"Missing text in sample {idx}")

        encoding = self.tokenizer.encode(text)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "emotion_labels": sample["emotion_labels"],
            "polarity_labels": torch.tensor(sample["polarity"], dtype=torch.float),
            "intensity_labels": torch.tensor(sample["intensity"], dtype=torch.float),
        }