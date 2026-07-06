"""
data/dataset.py

Dataset class for Hindi sentiment analysis, loaded from dialogue-style CSV files.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split

from config import DataConfig, TrainingConfig
from models.model_utils import ModelTokenizer

logger = logging.getLogger(__name__)


# Default polarity assigned to each emotion label (-1 negative, +1 positive, 0 neutral/ambiguous).
# This is a starting guess — please review and adjust once you know the FULL set of
# unique labels in your `emotions` column, e.g.:
#   pd.read_csv(...)["emotions"].str.split(",").explode().str.strip().unique()
DEFAULT_EMOTION_POLARITY = {
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
    "anticipation": 0.0,   # context-dependent; treated as neutral by default
    "surprise": 0.0,       # ambiguous polarity by default
    "neutral": 0.0,
}


class HindiSentimentCSVDataset(Dataset):
    """
    Dataset for Hindi sentiment analysis loaded from dialogue-style CSV files.

    Expected CSV columns:
        dialogueId, utterance_no, authorRole, utterance, emotions, emoIntensity

    - `emotions`     : comma-separated emotion labels, e.g. "sad,anger"
    - `emoIntensity` : comma-separated intensities, positionally aligned with
                       `emotions`, e.g. "3,3" (raw scale, e.g. 0-3; normalized
                       internally using `max_intensity`)

    `data_file` can be:
      - a path to a single CSV
      - a list of CSV paths
      - a directory (e.g. "data/raw" or "data/raw/emotion_hindi") — in which
        case every "*.csv" found recursively under it is loaded and combined,
        assuming they all share the same schema. `dialogueId` is prefixed
        per source file to avoid collisions across combined datasets.

    Design choices baked in here:
    - Only rows where authorRole == "user" are kept (bot turns dropped).
    - Each row is treated as an INDEPENDENT sample — no dialogue/context
      from previous turns is used.
    - There is no `polarity` column in the source data, so polarity is
      DERIVED per-sample as an intensity-weighted average of each present
      emotion's polarity sign (see DEFAULT_EMOTION_POLARITY).
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

        # Build a stable emotion vocabulary from the full (filtered) dataset,
        # so the vector index for e.g. "anger" is consistent across train/val/test.
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

    # ------------------------------------------------------------------ #
    # Loading & parsing
    # ------------------------------------------------------------------ #

    def _resolve_data_files(self, data_file: Union[str, List[str]]) -> List[Path]:
        """
        Resolve `data_file` into a concrete list of CSV paths.

        Accepts:
          - a single CSV path, e.g. "data/raw/emotion_hindi/dialogues.csv"
          - a list of CSV paths
          - a directory, e.g. "data/raw" or "data/raw/emotion_hindi" — in which
            case ALL "*.csv" files found recursively under it are used
        """
        if isinstance(data_file, (list, tuple)):
            paths = [Path(p) for p in data_file]
        else:
            p = Path(data_file)
            if p.is_dir():
                paths = sorted(p.rglob("*.csv"))
                if not paths:
                    raise FileNotFoundError(f"No CSV files found under directory: {data_file}")
                logger.info(
                    f"Auto-discovered {len(paths)} CSV file(s) under {data_file}: "
                    f"{[str(x) for x in paths]}"
                )
            else:
                paths = [p]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

        return paths

    def _load_csv(
        self, data_file: Union[str, List[str]], author_role_filter: Optional[str]
    ) -> pd.DataFrame:
        """
        Load one or more CSV files and concatenate them into a single DataFrame.
        Assumes all files share the same schema. A `_source_file` column is added,
        and `dialogueId` (if present) is prefixed per-source-file to avoid
        collisions when combining multiple raw datasets that each start their
        own dialogueId numbering from 0.
        """
        paths = self._resolve_data_files(data_file)

        frames = []
        reference_cols = None
        for path in paths:
            df = pd.read_csv(path, encoding="utf-8")

            if reference_cols is None:
                reference_cols = set(df.columns)
            elif set(df.columns) != reference_cols:
                logger.warning(
                    f"Column mismatch in {path}: expected {reference_cols}, "
                    f"got {set(df.columns)}. Proceeding, but check schemas match."
                )

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

        # Drop rows with missing utterance text
        df = df.dropna(subset=["utterance"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows left after filtering authorRole == {author_role_filter!r}. "
                f"Check the CSV's authorRole values."
            )

        logger.info(f"Loaded {len(df)} rows from {len(paths)} file(s)")

        return df

    def _build_emotion_vocab(self, df: pd.DataFrame) -> List[str]:
        """Scan the full dataset once to build a stable, sorted emotion vocabulary."""
        vocab = set()
        for emo_str in df["emotions"].fillna("neutral"):
            for label in str(emo_str).split(","):
                label = label.strip().lower()
                if label:
                    vocab.add(label)
        return sorted(vocab)

    def _parse_emotions_and_intensity(
        self, emo_str: str, intensity_str: str
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Parse comma-separated emotion labels + comma-separated intensities into:
          - a multi-label emotion vector (sized to self.emotion_vocab), filled
            with normalized intensity (0-1) for each emotion present
          - a derived polarity scalar (-1 to 1): intensity-weighted average of
            each present emotion's polarity sign
          - an overall intensity scalar (0-1): the max normalized intensity
            across the emotions present in this sample
        """
        labels = [l.strip().lower() for l in str(emo_str).split(",") if l.strip()]
        intens_raw = [i.strip() for i in str(intensity_str).split(",") if i.strip()]

        if len(intens_raw) != len(labels):
            logger.warning(
                f"Emotion/intensity length mismatch: {emo_str!r} vs {intensity_str!r}; "
                f"defaulting missing intensities to 1.0"
            )
            intens = [1.0] * len(labels)
            for i, val in enumerate(intens_raw[: len(labels)]):
                try:
                    intens[i] = float(val)
                except ValueError:
                    pass
        else:
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

            if label in self.emotion_vocab:
                idx = self.emotion_vocab.index(label)
                emotion_vec[idx] = norm_intensity

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

        train_data, temp_data = train_test_split(
            data, test_size=1 - train_ratio, random_state=seed
        )
        val_test_split = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, test_size=1 - val_test_split, random_state=seed
        )
        return train_data, val_data, test_data

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #

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
            "polarity_labels": torch.tensor([sample["polarity"]], dtype=torch.float),
            "intensity_labels": torch.tensor([sample["intensity"]], dtype=torch.float),
        }