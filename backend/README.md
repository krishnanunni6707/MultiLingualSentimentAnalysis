# Multilingual Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E.svg)](https://huggingface.co/)

A production-ready, dual-language sentiment analysis system for Hindi and English text. Two transformer-based models run in parallel at inference time, their outputs are cross-validated, and an optional Llama summarizer produces emotion-aware summaries.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Summarization](#summarization)
- [Training Data](#training-data)
- [Output Schema](#output-schema)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Training](#training)
- [REST API](#rest-api)
- [Python API](#python-api)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The system performs **three interconnected sentiment tasks** from a single forward pass:

| Task | Type | Output Range | Description |
|---|---|---|---|
| **Emotion Classification** | Multi-class / multi-label | 0.0 – 1.0 per class | Probability per emotion label |
| **Polarity** | Derived regression | −1.0 to +1.0 | Negative ↔ Neutral ↔ Positive |
| **Intensity** | Derived regression | 0.0 to 1.0 | Weak ↔ Very strong |

Default inference models use 7 labels including `neutral`: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`.

---

## Pipeline Architecture

### Dual-Path Inference (core flow)

```
                  Raw Input Text
                        │
              LanguageDetector (langdetect)
                        │
          ┌─────────────┴─────────────┐
          │   PARALLEL EXECUTION      │
          │                           │
    Hindi Branch               English Branch
    (vashuag/HindiEmotion)    (j-hartmann/distilroberta)
          │                           │
          │  [Hindi input only]       │
          │  MarianMT translation ────┘
          │                           │
          └────────────┬──────────────┘
                       │
                       ▼
              ResultComparator
              (per-segment reconcile)
                       │
                       ▼
                OutputAssembler
                       │
                       ▼
              BilingualSummarizer (optional)
                       │
                       ▼
               Final JSON Output
```

### Hindi input (`language="hi"`)

1. `HindiPreprocessor` cleans and normalizes Devanagari text.
2. `DocumentSegmenter` splits on sentence boundaries (Devanagari danda `।`).
3. **Branch A** — Hindi model runs on raw Hindi segments (primary).
4. **Branch B** — Segments are translated via `Helsinki-NLP/opus-mt-hi-en`, then the English model runs (cross-check).
5. `ResultComparator` reconciles with `primary_source: "hindi_direct"`.

### English input (`language="en"`)

1. `EnglishPreprocessor` cleans Latin text.
2. `DocumentSegmenter` splits on English sentence boundaries.
3. **Branch A** — English model runs natively (primary, `english_direct`).
4. **Branch B** — Hindi multilingual model runs on the same English segments (cross-check, no translation).
5. `ResultComparator` reconciles with `primary_source: "english_direct"`.

### Reconciliation Strategies

| Strategy | Behaviour |
|---|---|
| `agreement_flag` *(default)* | Use the language-appropriate primary branch; flag `agreement: false` on valence mismatch |
| `confidence_weighted_pick` | Pick the branch with higher confidence |
| `average` | Average emotion probabilities, polarity, intensity, and confidence |

Agreement is evaluated on **valence groups** (positive / negative / neutral), not exact label match. `surprise` is treated as neutral-valence because it is polarity-ambiguous.

### Per-branch processing

```
Raw Text
    │
    ▼
Preprocessor (Hindi or English)
    │
    ▼
DocumentSegmenter
    │
    ▼
ModelTokenizer (max_length=512)
    │
    ▼
AutoSequenceClassifierWrapper  OR  MultiTaskSentimentModel
    │
    ├── emotion_probabilities  (softmax over labels)
    ├── polarity_scores        (weighted sum of emotion valences)
    └── intensity_scores       (1 − P(neutral), or max non-neutral prob)
```

---

## Models

### Inference (default)

When no custom checkpoint is provided, `models/model_utils.py` loads pretrained HuggingFace classifiers from the **local cache** (`local_files_only=True`):

| Path | Model | Labels |
|---|---|---|
| Hindi | `vashuag/HindiEmotion` | anger, disgust, fear, joy, neutral, sadness, surprise |
| English | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise |
| Translation | `Helsinki-NLP/opus-mt-hi-en` | MarianMT seq2seq, beam search, CPU by default |

Both classifiers are wrapped in `AutoSequenceClassifierWrapper`:

- **Logit temperature smoothing** (`temperature=1.2`) prevents over-confident predictions on isolated emotion words.
- **Polarity** is derived as a weighted dot product of emotion probabilities and per-label valence signs.
- **Intensity** is `1.0 − P(neutral)` when a neutral label exists; otherwise the max non-neutral probability.
- **Ambiguity gate** — when max class confidence &lt; 0.45, polarity is pulled 70% toward neutral.

### Training (custom checkpoints)

Training uses `MultiTaskSentimentModel` — a shared transformer encoder with three dedicated heads:

| Language | Base Encoder | Training Data |
|---|---|---|
| Hindi | `distilbert-base-multilingual-cased` | EmoInHindi CSV |
| English | `bert-base-uncased` | GoEmotions (HuggingFace `datasets`) |

```
Input: CLS pooled representation [batch, 768]
                           │
    ┌──────────────────────|────────────────────┐
    │                      │                    │
    ▼                      ▼                    ▼
EmotionHead          PolarityHead         IntensityHead
  Linear(768→64)      Linear(768→16)       Linear(768→16)
  ReLU + Dropout      ReLU + Dropout       ReLU + Dropout
  Linear(64→N)        Linear(16→1)         Linear(16→1)
  Sigmoid per class   Tanh → [-1, 1]       Sigmoid → [0, 1]
  Loss : BCE          Loss : MSE           Loss : MSE
```

### Why BCE and MSE 
In Emotion:  Emotions are classified not predicted.
That is each emotions are classified "true/false" or "yes/no"
soo we need to find the Binary loss(BCE).

In Polarity and Intensity Head: These values are preddicted for each segments.
So the regression is used so we need to get the prediction loss
there for w using the Mean squared Error(MSE).

**Loss function** (`MultiTaskLoss`):

```
L_total = w_emo × BCE(emotion_logits, emotion_labels)
        + w_pol × MSE(polarity_scores, polarity_labels)
        + w_int × MSE(intensity_scores, intensity_labels)

Default weights: w_emo=1.0, w_pol=0.8, w_int=0.8
```

`load_model()` auto-detects checkpoint type: `MultiTaskSentimentModel` state dicts (keys like `encoder.*`, `emotion_head.*`) vs. plain sequence-classification weights.

Checkpoint search order for English: `checkpoints/en/best_model.pt` → `EngModel/model/final_model.pt`.

---

## Summarization

After sentiment analysis, `BilingualSummarizer` (`multilingual_summarizer/summarizer.py`) optionally generates an abstractive summary.

| Property | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Auth | `HF_TOKEN` in `.env` |
| Quantization | 4-bit NF4 on CUDA (via `bitsandbytes`) |
| Languages | English and Hindi (Devanagari ratio heuristic) |
| Context injection | Dominant emotion + intensity passed as tone context |

The pipeline adapts sentiment JSON via `adapt_sentiment_output_to_summarizer()` before calling the summarizer. Long texts (&gt; 3500 tokens) use hierarchical chunk-and-merge summarization.

Response fields:

| Field | Values |
|---|---|
| `summary` | Generated text, or `null` if summarizer unavailable |
| `summarizer_status` | `active` \| `disabled` |

Summarizer initialization failure is non-fatal — sentiment analysis continues without it.

---

## Training Data

### Hindi — EmoInHindi (LREC)

| Property | Detail |
|---|---|
| **Default file** | `data/EmoInHindi/LREC_EmoInHindi_reduced.csv` |
| **Format** | Dialogue CSV (`dialogueId`, `utterance_no`, `authorRole`, `utterance`, `emotions`, `emoIntensity`) |
| **Filter** | Only `authorRole == "user"` rows |
| **Emotion vocab** | Dynamically built from unique labels in the `emotions` column |
| **Intensity scale** | Raw 0–3, normalized to 0–1 by dividing by `max_intensity=3.0` |
| **Split** | 80% train / 10% val / 10% test |
| **Multi-file** | Pass a directory; all `*.csv` files are combined |

Polarity is derived per sample as an intensity-weighted average of each emotion's signed polarity (joy/hope → +1, anger/sadness/fear → −1, surprise/neutral → 0).

### English — GoEmotions (Google Research)

| Property | Detail |
|---|---|
| **Source** | `google-research-datasets/go_emotions` via HuggingFace `datasets` |
| **Size** | ~58,000 Reddit comments |
| **Emotions** | 27 fine-grained labels (model head size matches dataset vocabulary) |
| **Split** | Official HuggingFace train/validation/test |

---

## Output Schema

Every analysis returns a unified document:

```json
{
  "overall_sentiment": {
    "emotions": { "anger": 0.12, "joy": 0.64, "sadness": 0.08, "fear": 0.03, "surprise": 0.10, "disgust": 0.02, "neutral": 0.01 },
    "dominant_emotion": "joy",
    "polarity_score": 0.71,
    "polarity_description": "Positive",
    "intensity": 0.78,
    "intensity_description": "Strong",
    "confidence": 0.89,
    "num_segments": 4,
    "cross_check_summary": {
      "strategy_used": "agreement_flag",
      "num_segments": 4,
      "num_agree": 3,
      "num_disagree": 1,
      "agreement_rate": 0.75
    }
  },
  "metadata": {
    "language": "hi",
    "language_confidence": 0.98,
    "original_length": 1240,
    "cleaned_length": 1188,
    "num_segments": 4,
    "segment_strategy": "sentence"
  },
  "segments": [
    {
      "segment_id": "doc_001_seg_000",
      "text_span": "char_span: 0-312",
      "sentiment": {
        "label": "joy",
        "polarity_score": 0.82,
        "intensity": 0.74,
        "confidence": 0.91
      },
      "emotion_tags": ["joy", "surprise"],
      "language_detected": "hi",
      "translation_used": true,
      "cross_check": {
        "hindi_direct": { "label": "joy", "polarity_score": 0.80, "intensity": 0.73, "confidence": 0.90 },
        "english_translated": { "label": "joy", "polarity_score": 0.84, "intensity": 0.75, "confidence": 0.89 },
        "agreement": true,
        "strategy_used": "agreement_flag",
        "primary_source": "hindi_direct"
      }
    }
  ],
  "summary": "The text expresses strong positive sentiment about...",
  "summarizer_status": "active"
}
```

---

## Project Structure

```
backend/
├── config.py                     # Model, training, inference, API, translation config
├── main.py                       # CLI entry point (sentiment + summary)
├── train.py                      # Hindi model training CLI
├── train_en.py                   # English model training CLI
├── requirements.txt
│
├── models/
│   ├── sentiment_model.py        # MultiTaskSentimentModel (encoder + 3 heads)
│   └── model_utils.py            # AutoSequenceClassifierWrapper, ModelTokenizer, load_model()
│
├── data/
│   ├── dataset.py                # HindiSentimentCSVDataset
│   ├── dataset_en.py             # GoEmotionsDataset
│   ├── dataloader.py             # Hindi data loaders
│   ├── dataloader_en.py          # English data loaders
│   ├── total-emotion-splitter.py # Utility to extract unique emotions from CSV
│   └── EmoInHindi/               # Hindi training data (not in git)
│
├── training/
│   ├── trainer.py                # Training loop, eval, checkpointing, early stopping
│   ├── losses.py                 # MultiTaskLoss
│   └── metrics.py                # F1, MSE, Pearson correlation
│
├── inference/
│   ├── pipeline.py               # SentimentInferencePipeline + RealTimeInference
│   ├── pipeline_en.py            # English standalone pipeline
│   ├── segmentation.py           # DocumentSegmenter
│   ├── comparator.py             # ResultComparator
│   └── output_assembler.py       # OutputAssembler
│
├── multilingual_summarizer/
│   ├── summarizer.py             # BilingualSummarizer (Llama-3.2-3B)
│   └── config.py                 # Summarizer prompts, generation params
│
├── utils/
│   ├── preprocessing.py          # HindiPreprocessor, LanguageDetector, PolarityPivotAdjuster
│   ├── preprocessing_en.py       # English preprocessing
│   ├── tokenizer.py                # HindiWordTokenizer (Indic-NLP)
│   ├── translator.py               # HindiToEnglishTranslator (MarianMT)
│   └── logger.py                   # Centralized logging
│
├── api/
│   ├── routes.py                   # FastAPI endpoints
│   ├── schemas.py                  # Pydantic request/response models
│   └── middleware.py               # Request logging, timing, size limits
│
├── EngModel/                       # Standalone English training utilities
│   ├── train.py
│   ├── data_processing.py
│   ├── clean-data.py
│   └── train-test-split.py
│
├── checkpoints/                  # Model checkpoints (gitignored)
│   ├── best_model.pt
│   └── en/best_model.pt
└── logs/
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, recommended for inference and summarization)
- HuggingFace account + token (for Llama summarizer)
- Pre-downloaded model weights in local HuggingFace cache

### Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### Environment Variables

Create `.env` in `backend/`:

```env
HF_TOKEN=your_huggingface_token
MODEL_PATH=checkpoints/best_model.pt
TRANSLATION_DEVICE=cpu
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### Pre-download Models

Inference uses `local_files_only=True`. Download models before first run:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

for name in [
    "vashuag/HindiEmotion",
    "j-hartmann/emotion-english-distilroberta-base",
    "Helsinki-NLP/opus-mt-hi-en",
]:
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(name)
```

For summarization, also cache `meta-llama/Llama-3.2-3B-Instruct` with your `HF_TOKEN`.

---

## CLI Usage

`main.py` is the command-line entry point. It loads all models (sentiment, translator, summarizer) and prints a summary.

```bash
cd backend

# Analyze a text file
python main.py --file test-input_hindi.txt

# Analyze inline text
python main.py --text "यह फिल्म बहुत अच्छी है!"

# Force language
python main.py --file test-input_english.txt --language en
```

| Flag | Description |
|---|---|
| `--file`, `-f` | Path to a `.txt` file |
| `--text`, `-t` | Text string to analyze |
| `--language`, `-l` | Force language: `hi` or `en` |
| `--length` | Summary length hint: `short`, `medium` (default), `long` |

Example output:

```
[~] Reading input...
[~] Loading all models...
[~] Running sentiment analysis...

==================================================
ANALYSIS SUMMARY
==================================================
Language: hi
Dominant Emotion: joy
Polarity: Positive

-----------------------SUMMARY-----------------------
The text conveys a strongly positive tone...
==================================================
```

---

## Training

### Train the Hindi Model

```bash
python train.py train data/EmoInHindi/LREC_EmoInHindi_reduced.csv

python train.py train data/EmoInHindi/LREC_EmoInHindi_reduced.csv \
    --epochs 20 --batch-size 16 --lr 2e-5 --output-dir ./checkpoints

python train.py evaluate data/EmoInHindi/LREC_EmoInHindi_reduced.csv checkpoints/best_model.pt
```
To Continue Training (Fine-tune from previous best)
Use the --checkpoint flag to resume from your saved model:

```bash
python train.py train --checkpoint checkpoints/best_model.pt
```

### Train the English Model

```bash
python train_en.py
python train_en.py --epochs 10 --batch-size 32 --output-dir checkpoints/en
```

GoEmotions is downloaded automatically from HuggingFace on first run.

### Checkpoint Format

```python
{
    "model_state_dict": ...,
    "emotion_labels": ["anger", "joy", ...],
    "num_emotions": N,
    "model_name": "distilbert-base-multilingual-cased"
}
```

Each epoch saves `checkpoints/checkpoint_epoch_{N}.pt`. The best validation-loss epoch writes `checkpoints/best_model.pt`.

---

## REST API

API routes are defined in `api/routes.py` with Pydantic schemas in `api/schemas.py` and middleware in `api/middleware.py`.

To start a server, create a small app module (e.g. `server.py`) in `backend/`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.middleware import configure_api_middleware
from config import APIConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Multilingual Sentiment Analysis", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
configure_api_middleware(app)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=APIConfig.HOST, port=APIConfig.PORT, reload=APIConfig.RELOAD)
```

```bash
cd backend
python server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/analyze` | Aggregate sentiment + metadata + summary |
| `POST` | `/api/v1/analyze_detailed` | Full output with per-segment breakdown |
| `POST` | `/api/v1/analyze_json` | Same as `analyze_detailed` |
| `POST` | `/api/v1/batch_analyze` | Batch of up to 100 texts |
| `POST` | `/api/v1/analyze_stream` | Server-Sent Events for long documents |
| `GET` | `/api/v1/emotions` | Supported emotion labels |
| `GET` | `/api/v1/stats` | Model and API statistics |

#### Example

```bash
curl -X POST http://localhost:8000/api/v1/analyze_json \
  -H "Content-Type: application/json" \
  -d '{"text": "यह फिल्म बहुत अच्छी है!"}'
```

---

## Python API

```python
from inference.pipeline import SentimentInferencePipeline, RealTimeInference

# Full pipeline with optional custom checkpoints
pipeline = SentimentInferencePipeline(
    model_path="checkpoints/best_model.pt",
    english_model_path="checkpoints/en/best_model.pt",
    comparison_strategy="agreement_flag",
)

result = pipeline.process_document(
    text="मुझे यह फिल्म बहुत पसंद है।",
    return_segments=True,
    segment_id="doc_001",
)

print(result["overall_sentiment"]["dominant_emotion"])
print(result["summary"])
print(result["summarizer_status"])

# Lightweight wrapper (used by API routes)
inference = RealTimeInference()
result = inference.predict("यह एक परीक्षण वाक्य है।")
results = inference.predict_batch(["text 1", "text 2"])
```

---

## Configuration Reference

| Config Class | Key | Default | Description |
|---|---|---|---|
| `ModelConfig` | `MODEL_NAME` | `distilbert-base-multilingual-cased` | Hindi training encoder |
| `ModelConfigEN` | `MODEL_NAME` | `bert-base-uncased` | English training encoder |
| `ModelConfig` | `MAX_SEQUENCE_LENGTH` | `512` | Max subword tokens |
| `InferenceConfig` | `SEGMENTATION_STRATEGY` | `"sentence"` | `sentence` / `paragraph` / `fixed_length` |
| `InferenceConfig` | `BATCH_SIZE` | `32` | Inference batch size |
| `TranslationConfig` | `MODEL_NAME` | `Helsinki-NLP/opus-mt-hi-en` | MarianMT model |
| `TranslationConfig` | `DEVICE` | `"cpu"` | Translation device |
| `APIConfig` | `MAX_TEXT_LENGTH` | `50000` | Max request text length (chars) |
| `TrainingConfig` | `BATCH_SIZE` | `1` | Training batch size |
| `TrainingConfig` | `PATIENCE` | `3` | Early stopping patience |

---

## Troubleshooting

### Models not found (`local_files_only=True`)

Pre-download HuggingFace models to your cache. See [Pre-download Models](#pre-download-models).

### `HF_TOKEN` / summarizer disabled

The summarizer requires a valid HuggingFace token with Llama access and a locally cached model. Sentiment analysis works without it.

### CUDA out of memory

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}
```

Reduce `TrainingConfig.BATCH_SIZE` and `InferenceConfig.BATCH_SIZE` in `config.py`.

### Slow inference on CPU

- Translation runs on CPU by default and is the main bottleneck for Hindi inputs.
- Set `TRANSLATION_DEVICE=cuda` if GPU memory allows.
- Increase `InferenceConfig.BATCH_SIZE` to amortize tokenization.

### `indic-nlp-library` not installed

```bash
pip install indic-nlp-library
```

Optional but recommended for Hindi Unicode normalization. The system falls back gracefully without it.

### Clear `__pycache__`

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

## License

MIT License — see [LICENSE](../LICENSE).

**Version:** 1.1.0 · **Last Updated:** July 2026

If cuda cache full do this : nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}

to clear the __Pycache__ : find . -type d -name "__pycache__" -exec rm -rf {} +