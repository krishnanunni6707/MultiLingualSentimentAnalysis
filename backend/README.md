# Multilingual Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E.svg)](https://huggingface.co/)

A production-ready, dual-language multi-task sentiment analysis system for Hindi (and English) text. It fine-tunes two transformer encoders — one for Hindi, one for English — and runs them in parallel at inference time for cross-validated, high-confidence predictions.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architectures](#pipeline-architectures)
  - [1. Hindi Direct Pipeline](#1-hindi-direct-pipeline)
  - [2. English Cross-Check Pipeline](#2-english-cross-check-pipeline)
  - [3. Dual-Path Inference Pipeline](#3-dual-path-inference-pipeline)
- [Models and Fine-Tuning](#models-and-fine-tuning)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Output Schema](#output-schema)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Training](#training)
- [API Usage](#api-usage)
- [Python API](#python-api)
- [Configuration Reference](#configuration-reference)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

---

## Overview

The system performs **three interconnected sentiment tasks** simultaneously from a single forward pass:

| Task | Type | Output Range | Description |
|---|---|---|---|
| **Emotion Classification** | Multi-label Sigmoid | 0.0 – 1.0 per class | Probability of 6 Ekman emotions |
| **Polarity Regression** | Tanh Regression | −1.0 to +1.0 | Negative ↔ Neutral ↔ Positive |
| **Intensity Regression** | Sigmoid Regression | 0.0 to 1.0 | Weak ↔ Very strong |

The 6 emotion classes follow **Ekman's basic emotions**: `anger`, `joy`, `sadness`, `fear`, `surprise`, `disgust`.

---

## Pipeline Architectures

### 1. Hindi Direct Pipeline

Handles Hindi text end-to-end using a fine-tuned multilingual encoder.

```
Raw Hindi Text
      │
      ▼
┌─────────────────────────────┐
│  HindiPreprocessor          │
│  • URL / email / HTML clean │
│  • Unicode normalization    │
│  • Devanagari char filter   │
└────────────┬────────────────┘
             │  cleaned text
             ▼
┌─────────────────────────────┐
│  DocumentSegmenter          │
│  • sentence (default)       │  ← NLTK + Devanagari danda (।)
│  • paragraph                │
│  • fixed-length + overlap   │
└────────────┬────────────────┘
             │  [seg_0, seg_1, ..., seg_N]
             ▼
┌─────────────────────────────┐
│  ModelTokenizer             │
│  (distilbert-multilingual)  │
│  max_length=512 subwords    │
└────────────┬────────────────┘
             │  input_ids, attention_mask
             ▼
┌─────────────────────────────┐
│  MultiTaskSentimentModel    │  ← Fine-tuned on EmoInHindi CSV
│  Encoder: distilbert-base-  │
│  multilingual-cased         │
│  CLS pooling → 768-dim      │
│         │                   │
│  ┌──────┼───────────────┐   │
│  │      │               │   │
│  ▼      ▼               ▼   │
│ Emotion Polarity  Intensity │
│  Head   Head       Head     │
│ (64-dim)(16-dim)  (16-dim)  │
│ Sigmoid  Tanh     Sigmoid   │
└────────────┬────────────────┘
             │
             ▼
  Per-segment raw predictions
  { emotions, polarity, intensity, confidence }
```

---

### 2. English Cross-Check Pipeline

Independently verifies sentiment on the English translation of each Hindi segment.

```
Hindi Segments
      │
      ▼
┌─────────────────────────────┐
│  HindiToEnglishTranslator   │
│  Helsinki-NLP/opus-mt-hi-en │  ← MarianMT seq2seq, beam=4
│  (batch translation, CPU)   │
└────────────┬────────────────┘
             │  English segments
             ▼
┌─────────────────────────────┐
│  ModelTokenizer (BERT)      │
│  bert-base-uncased          │
│  max_length=512 subwords    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  MultiTaskSentimentModel    │  ← Fine-tuned on GoEmotions
│  Encoder: bert-base-uncased │
│  CLS pooling → 768-dim      │
│         │                   │
│  ┌──────┼───────────────┐   │
│  │      │               │   │
│  ▼      ▼               ▼   │
│ Emotion Polarity  Intensity │
│  Head   Head       Head     │
│ (same architecture as ↑)    │
└────────────┬────────────────┘
             │
             ▼
  English-branch raw predictions
```

---

### 3. Dual-Path Inference Pipeline

The two pipelines above run **in parallel** using `ThreadPoolExecutor`, then their per-segment results are reconciled and aggregated.

```
                  Raw Input Text
                        │
              LanguageDetector (langdetect)
                        │
          ┌─────────────┴─────────────┐
          │   PARALLEL EXECUTION      │
          │                           │
    Hindi Direct             English Branch
    Pipeline (↑)             (translate → ↑)
          │                           │
          └────────────┬──────────────┘
                       │
                       ▼
              ResultComparator
              (per-segment reconcile)
              ┌────────────────────────┐
              │ strategy options:      │
              │ • agreement_flag       │ ← default: use hindi, flag mismatch
              │ • confidence_weighted  │   pick highest-confidence branch
              │ • average              │   average both branches
              └────────────┬───────────┘
                           │ reconciled segment results
                           ▼
                    OutputAssembler
              ┌────────────────────────┐
              │ aggregate_results()    │ → overall_sentiment
              │ format_full_output()   │ → complete JSON document
              └────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   Final JSON Output   │
               │  overall_sentiment    │
               │  metadata             │
               │  segments[]           │
               └───────────────────────┘
```
### When Input ==> English
```
                          [ Raw English Text Input ]
                                      │
                                      ▼
                          [ Language Detection ]
                       (LanguageDetector -> "en")
                                      │
                                      ▼
                      [ Segmentation & Preprocessing ]
                      - Routes to EnglishPreprocessor
                      - Cleans text (retains A-Z, a-z, etc.)
                      - Segments into English sentences
                                      │
                                      ▼
                        [ Parallel Dual-Path Execution ]
                 ┌──────────────────────────────────────────────┐
                 │                                              │
                 ▼ (Branch A)                                   ▼ (Branch B)
       [ Hindi Direct Model ]                       [ English Standalone Model ]
    - distilbert-multilingual                    - bert-base-uncased
    - Trained on EmoInHindi                      - Trained on GoEmotions
    - Runs directly on English segments          - Runs natively on English segments
                 │                                              │
                 ▼                                              ▼
            hindi_result                                  english_result
                 │                                              │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                              [ ResultComparator ]
                          reconcile(..., language="en")
                          - Detects input is English
                          - Sets English model as primary ("english_direct")
                          - Sets Hindi model as cross-check validation
                          - Evaluates branch agreement
                                        │
                                        ▼
                              [ OutputAssembler ]
                          Aggregates & formats results
                                        │
                                        ▼
                       [ Final JSON Output Response ]
                       - dominant_emotion, polarity (English-model primary)
                       - metadata (language: "en", translation_used: false)
                       - cross_check (contains validation stats from both models)


```

### When Input ===> Hindi

```
                            [ Raw Hindi Text Input ]
                                       │
                                       ▼
                           [ Language Detection ]
                        (LanguageDetector -> "hi")
                                       │
                                       ▼
                       [ Segmentation & Preprocessing ]
                       - Routes to HindiPreprocessor
                       - Cleans & normalizes Devanagari characters
                       - Segments using Hindi rules (splits on ।)
                                       │
                                       ▼
                         [ Parallel Dual-Path Execution ]
                  ┌──────────────────────────────────────────────┐
                  │                                              │
                  ▼ (Hindi Branch)                               ▼ (English Branch)
        [ Hindi Direct Model ]                         [ Hindi-to-English Translation ]
     - distilbert-multilingual                      - Translates Hindi to English via 
     - Runs directly on Hindi                         Helsinki MarianMT translator
       Devanagari segments                                       │
                  │                                              ▼
                  │                                    [ English Standalone Model ]
                  │                                  - Runs GoEmotions model on 
                  │                                    translated segments
                  │                                              │
                  ▼                                              ▼
             hindi_result                                  english_result
                  │                                              │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                               [ ResultComparator ]
                          reconcile(..., language="hi")
                          - Detects input is Hindi
                          - Sets Hindi model as primary ("hindi_direct")
                          - Sets translated English model as cross-check
                          - Evaluates branch agreement
                                         │
                                         ▼
                               [ OutputAssembler ]
                           Aggregates & formats results
                                         │
                                         ▼
                        [ Final JSON Output Response ]
                        - dominant_emotion, polarity (Hindi-model primary)
                        - metadata (language: "hi", translation_used: true)
                        - cross_check (reconciliation stats)


```
### Training Architecture
```
                            [ Raw CSV Data Source ]
                         (dialogues / dataset folder)
                                       │
                                       ▼
                         [ Data Filtering & Preprocessing ]
                         - Filter: keep only authorRole == "user"
                         - Cleans missing/empty utterances
                                       │
                                       ▼
                         [ Label Derivation & Vectorization ]
                         - Scans dataset to discover unique emotion_vocab
                         - Normalizes emotion intensities to range [0.0, 1.0]
                         - Derives polarity [-1.0, 1.0] (weighted average)
                         - Derives intensity [0.0, 1.0] (max normalized intensity)
                                       │
                                       ▼
                            [ Tokenization Pipeline ]
                         - ModelTokenizer (max sequence length = 512)
                         - Generates input_ids and attention_mask
                                       │
                                       ▼
                             [ Dataset Splitting ]
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
                 [ Train Split ]  [ Val Split ]  [ Test Split ]
                     ( 80% )          ( 10% )        ( 10% )
                       │               │               │
                       ▼               ▼               ▼
                            [ Custom Batch DataLoader ]
                            - Stacks tensors using CollateFunction
                            - Pins memory & enables multi-worker prefetch
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │                                   │
                     ▼ (Forward Pass)                    ▼ (Backward & Optimize)
           [ Multi-Task Model ]                 [ Loss & Optimizer Step ]
        ┌──────────────────────────┐         - Computes weighted MultiTaskLoss:
        │ Shared Encoder (mBERT)   │           * Emotion: Multi-label BCE
        └────────────┬─────────────┘           * Polarity: Regress MSE
                     │ (Pooled Vector)         * Intensity: Regress MSE
        ┌────────────┼─────────────┐         - Gradient clipping (max_norm = 1.0)
        ▼            ▼             ▼         - Update via AdamW & Linear Warmup
     Emotion      Polarity     Intensity                   │
      Head          Head         Head                      │
                     │                                     ▼
                     └─────────────────────────────> [ Epoch Loop ]
                                                           │
                                                           ▼
                                                [ Validation & Metrics ]
                                             - Runs inference on Val Split
                                             - Computes F1, MSE, Pearson Correlation
                                             - Checks early stopping (Patience = 3)
                                                           │
                                                           ▼
                                               [ Checkpoint Management ]
                                             - Saves best_model.pt (Lowest loss)
                                             - Saves epoch checkpoint_epoch_X.pt


```


#### Reconciliation Strategies

| Strategy | Behaviour (Hindi Input) | Behaviour (English Input) |
|---|---|---|
| `agreement_flag` | Use Hindi-direct result as primary; mark `agreement: false` if branches disagree | Use English-direct result as primary (`english_direct`); mark `agreement: false` if branches disagree |
| `confidence_weighted_pick` | Pick whichever branch has higher confidence | Pick whichever branch has higher confidence |
| `average` | Average emotion probabilities, polarity, intensity, and confidence across both branches | Average emotion probabilities, polarity, intensity, and confidence across both branches |

#### Dual-Path Routing for English Inputs

When an English document is provided (detected automatically or specified via `language="en"`):
1. **Segmentation & Preprocessing**: The pipeline routes the input to the `EnglishPreprocessor` instead of the `HindiPreprocessor`. This prevents character-stripping bugs.
2. **Parallel Branches**:
   - **GoEmotions English branch** runs natively on the English segments (without translation).
   - **Hindi Multilingual branch** runs on the English segments directly.
3. **Reconciliation**: The `ResultComparator` treats the GoEmotions English branch as the primary (`english_direct`) and the Hindi Multilingual branch as the secondary/cross-check, providing cross-validated results.

---

## Models and Upgraded Architectures

The system utilizes two highly optimized pre-trained sequence classification models wrapped in a dynamic adapter wrapper class (`AutoSequenceClassifierWrapper`). This configuration ensures context-aware emotion recognition and avoids keyword token poisoning.

### 1. Hindi Model Path

*   **Primary Candidate:** `l3cube-pune/hindi-emotion-bert` or `l3cube-pune/hindi-sentiment-bert`
*   **Automatic Fallback (Active):** `vashuag/HindiEmotion` (a robust multi-class Devanagari emotion BERT model fine-tuned on nuanced Hindi texts)
*   **Device Mapping:** Automatically configured on `cuda` if available, otherwise maps to `cpu`.
*   **Decoded Labels:** `{0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}`
*   **Tokenizer:** Custom tokenization with dynamic safety fallback to slow mode (`use_fast=False`) to prevent rust-based parsing failures.

### 2. English Model Path

*   **Primary Candidate:** `ArisXavier/roberta-base-emotion`
*   **Automatic Fallback (Active):** `j-hartmann/emotion-english-distilroberta-base` (trained on Ekman's basic emotions + Neutral)
*   **Device Mapping:** Automatically configured on `cuda` if available, otherwise maps to `cpu`.
*   **Decoded Labels:** `{0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}`

---

### 3. Translation Model — `Helsinki-NLP/opus-mt-hi-en`

| Property | Value |
|---|---|
| **Model** | `Helsinki-NLP/opus-mt-hi-en` (MarianMT) |
| **Type** | Seq2Seq (encoder-decoder), NOT fine-tuned — used as-is |
| **Beam search** | 4 beams |
| **Device** | CPU by default (configurable via `TRANSLATION_DEVICE` env var) |
| **Max length** | 512 tokens |

> This model is **not fine-tuned** — it is used as a frozen translator to bridge Hindi segments to the English sentiment branch for cross-validation.

---

## Model Architecture & Dynamics

Both paths leverage the `AutoSequenceClassifierWrapper` class which mimics the Multi-Task Sentiment architecture by dynamically mapping classification outputs to the pipeline:

1. **Logit Temperature Smoothing (`temperature=1.2`)**:
   Smooths logits before softmax:
   $$\text{smoothed\_logits} = \frac{\text{logits}}{1.2}$$
   This prevents over-saturation (e.g. rigid 90%+ confidences) on isolated heavy nouns like "अवसाद" or "चिंता".

2. **Dynamic Polarity Derivation**:
   Polarity regression scores are calculated dynamically by taking the dot product of emotion probabilities and their respective polarity weights:
   - Positive emotions (`joy`, `happy`, `love`, etc.): $+1.0$
   - Negative emotions (`anger`, `sadness`, `fear`, `disgust`, etc.): $-1.0$
   - Other emotions (`surprise`, `neutral`): $0.0$

3. **Dynamic Intensity Derivation**:
   Intensity is derived as $1.0 - P(\text{neutral})$ if a neutral label is present, or defaults to the maximum probability of the dominant non-neutral emotion.

---

## Model Architecture

Both Hindi and English fine-tuned models use the identical task-head design:

```
Input: CLS pooled representation [batch, 768]
          │
    ┌─────┼────────────────────┐
    │     │                    │
    ▼     ▼                    ▼
EmotionHead          PolarityHead         IntensityHead
  Linear(768→64)      Linear(768→16)       Linear(768→16)
  ReLU + Dropout      ReLU + Dropout       ReLU + Dropout
  Linear(64→N)        Linear(16→1)         Linear(16→1)
  (N = num emotions)  Tanh → [-1, 1]       Sigmoid → [0, 1]
  Sigmoid per class
```

**Loss function** (`MultiTaskLoss`):

```
L_total = w_emo × BCE(emotion_logits, emotion_labels)
        + w_pol × MSE(polarity_scores, polarity_labels)
        + w_int × MSE(intensity_scores, intensity_labels)

Default weights: w_emo=1.0, w_pol=0.8, w_int=0.8
```

**Polarity label derivation (Hindi data):**
Since EmoInHindi has no dedicated polarity column, polarity is derived per-sample as an **intensity-weighted average of each emotion's signed polarity**:

```
polarity = Σ (polarity_sign[emotion] × intensity) / Σ intensity
```

where `polarity_sign`: joy/happy/confident/trust → +1, anger/sad/fear/disgust → −1, surprise/anticipation/neutral → 0.

---

## Training Data

### Hindi — EmoInHindi (LREC)

| Property | Detail |
|---|---|
| **File** | `data/EmoInHindi/LREC_EmoInHindi.csv` |
| **Format** | Dialogue CSV (dialogueId, utterance_no, authorRole, utterance, emotions, emoIntensity) |
| **Filter** | Only `authorRole == "user"` rows used |
| **Emotion vocab** | Dynamically built from unique labels found in the `emotions` column (sorted, persisted in checkpoint) |
| **Intensity scale** | Raw 0–3, normalized to 0–1 by dividing by `max_intensity=3.0` |
| **Split** | 80% train / 10% val / 10% test |
| **Multi-file support** | Pass a directory; all `*.csv` under it are auto-discovered and combined |

### English — GoEmotions (Google Research)

| Property | Detail |
|---|---|
| **Source** | `google-research-datasets/go_emotions` via HuggingFace `datasets` |
| **Size** | ~58,000 Reddit comments |
| **Emotions** | 27 fine-grained → remapped to 6 Ekman classes for training |
| **Split** | Official HuggingFace train/validation/test |
| **Intensity** | Binary placeholder (GoEmotions simplified config has no graded intensity) |

---

## Output Schema

Every endpoint (`/analyze`, `/analyze_detailed`, `/analyze_json`) returns the same unified document:

```json
{
  "overall_sentiment": {
    "emotions": {
      "anger": 0.12,
      "joy": 0.64,
      "sadness": 0.08,
      "fear": 0.03,
      "surprise": 0.10,
      "disgust": 0.02
    },
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
        "hindi_direct": {
          "label": "joy",
          "polarity_score": 0.80,
          "intensity": 0.73,
          "confidence": 0.90
        },
        "english_translated": {
          "label": "joy",
          "polarity_score": 0.84,
          "intensity": 0.75,
          "confidence": 0.89
        },
        "agreement": true,
        "strategy_used": "agreement_flag",
        "primary_source": "hindi_direct"
      }
    }
  ]
}
```

**`overall_sentiment` fields:**

| Field | Description |
|---|---|
| `emotions` | Mean emotion probability per class, averaged across all segments |
| `dominant_emotion` | Class with highest mean probability |
| `polarity_score` | Mean polarity across segments (−1 to +1) |
| `polarity_description` | `"Highly negative"` / `"Negative"` / `"Neutral"` / `"Positive"` / `"Highly positive"` |
| `intensity` | Mean intensity across segments (0 to 1) |
| `intensity_description` | `"Very weak"` / `"Weak"` / `"Moderate"` / `"Strong"` / `"Very strong"` |
| `confidence` | Mean max-emotion-probability (used as model confidence proxy) |
| `num_segments` | How many segments the document was split into |
| `cross_check_summary` | Dual-path agreement statistics across all segments |

**`segments[*]` fields:**

| Field | Description |
|---|---|
| `segment_id` | `{doc_id}_seg_{index:03d}` |
| `text_span` | `char_span: {start}-{end}` (cumulative character offsets into cleaned text) |
| `sentiment.label` | Dominant emotion for this segment |
| `sentiment.polarity_score` | Per-segment polarity |
| `sentiment.intensity` | Per-segment intensity |
| `sentiment.confidence` | Max emotion probability for this segment |
| `emotion_tags` | Top-2 emotions by score for this segment |
| `language_detected` | Language code detected for this segment |
| `translation_used` | Whether the English branch translated this segment |
| `cross_check` | Both branch results + agreement flag |

---

## Project Structure

```
backend/
├── config.py                     # All configuration: models, training, API, inference
├── main.py                       # FastAPI app entry point (lifespan, CORS, routing)
├── train.py                      # Hindi model training CLI
├── train_en.py                   # English model training CLI
├── requirements.txt
│
├── models/
│   ├── sentiment_model.py        # MultiTaskSentimentModel (encoder + 3 heads)
│   │   ├── EmotionHead           # Linear(768→64)→ReLU→Linear(64→N)→Sigmoid
│   │   ├── PolarityHead          # Linear(768→16)→ReLU→Linear(16→1)→Tanh
│   │   └── IntensityHead         # Linear(768→16)→ReLU→Linear(16→1)→Sigmoid
│   └── model_utils.py            # ModelTokenizer, load_model(), checkpoint I/O
│
├── data/
│   ├── dataset.py                # HindiSentimentCSVDataset (EmoInHindi CSV)
│   ├── dataset_en.py             # GoEmotionsDataset (HuggingFace, auto-download)
│   ├── dataloader.py             # CollateFunction + create_data_loaders (Hindi)
│   ├── dataloader_en.py          # create_english_data_loaders
│   └── EmoInHindi/               # Hindi training data directory
│       └── LREC_EmoInHindi.csv
│
├── training/
│   ├── trainer.py                # Trainer: train loop, eval, checkpointing, early stop
│   ├── losses.py                 # MultiTaskLoss (BCE + MSE + MSE, weighted)
│   └── metrics.py                # compute_emotion_metrics, compute_regression_metrics
│
├── inference/
│   ├── pipeline.py               # SentimentInferencePipeline (Hindi+English dual-path)
│   ├── pipeline_en.py            # SentimentInferencePipelineEN (English standalone)
│   ├── segmentation.py           # DocumentSegmenter (wraps preprocessing strategies)
│   ├── comparator.py             # ResultComparator (reconcile + summarize)
│   └── output_assembler.py       # OutputAssembler (aggregate, format, save)
│
├── utils/
│   ├── preprocessing.py          # HindiPreprocessor, LanguageDetector, Segmentation
│   ├── preprocessing_en.py       # English preprocessing pipeline
│   ├── tokenizer.py              # HindiWordTokenizer (word-level, Indic-NLP)
│   ├── translator.py             # HindiToEnglishTranslator (MarianMT)
│   └── logger.py                 # Centralized logging (file + console)
│
├── api/
│   ├── routes.py                 # FastAPI endpoints (/analyze, /analyze_json, etc.)
│   ├── schemas.py                # Pydantic models (DocumentSentimentResponse, etc.)
│   └── middleware.py             # Request logging, rate limiting middleware
│
├── checkpoints/
│   ├── best_model.pt             # Best Hindi checkpoint (saved by trainer)
│   └── en/
│       └── best_model.pt         # Best English checkpoint
└── logs/
    └── api.log
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- `sentencepiece` system library (required by MarianMT tokenizer)

### Setup

```bash
# 1. Clone and enter the backend directory
cd mul-ling-senti-analysis/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install sentencepiece (required for MarianMT translator)
pip install sentencepiece

# 5. Download NLTK tokenizer data
python -c "import nltk; nltk.download('punkt')"

# 6. (Optional) Install indic-nlp-library for better Hindi normalization
pip install indic-nlp-library
```

### Environment Variables

Create a `.env` file in `backend/` to configure paths and devices:

```env
# Path to a trained Hindi model checkpoint (optional; model runs without one)
MODEL_PATH=checkpoints/best_model.pt

# Translation model device (cpu or cuda)
TRANSLATION_DEVICE=cpu

# API server settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

---

## CLI Usage

`main.py` is the command-line interface for running sentiment analysis directly without starting an API server.

### Basic Usage

```bash
# Navigate to backend directory first
cd mul-ling-senti-analysis/backend
```

### 1. Analyze a `.txt` File (Positional — Quickest)

Pass the file path directly as a positional argument:

```bash
python main.py document.txt
python main.py data/sample.txt
```

### 2. Analyze a `.txt` File (via Flag)

```bash
python main.py --file document.txt
python main.py -f document.txt
```

### 3. Analyze Text Directly from the Command Line

```bash
python main.py --text "यह फिल्म बहुत अच्छी है!"
python main.py -t "This movie is absolutely wonderful!"
```

### 4. Interactive Mode

Launch with no arguments to enter the interactive REPL:

```bash
python main.py
```

Inside interactive mode:

| Command | Description |
|---|---|
| `<type your text>` | Analyze the typed text directly |
| `file <path>` | Analyze a text file by path |
| `multiline` | Enter multi-line input mode (submit with an empty line) |
| `detailed` | Toggle per-segment analysis on/off |
| `help` | Print command list |
| `exit` / `quit` | Exit the program |

### Options & Flags

| Flag | Short | Description |
|---|---|---|
| `input_file` | *(positional)* | Path to a `.txt` file to analyze |
| `--file` | `-f` | Path to a `.txt` file to analyze |
| `--text` | `-t` | Text string to analyze directly |
| `--language` | `-l` | Force language: `hi` (Hindi) or `en` (English) |
| `--detailed` | `-d` | Print per-segment breakdown with cross-check results |
| `--output` | `-o` | Save full JSON results to a file |
| `--no-color` | | Disable ANSI color in terminal output |

### Combined Examples

```bash
# Analyze a Hindi .txt file with detailed segment-level output
python main.py document.txt --detailed

# Analyze a file, force language to Hindi, save JSON output
python main.py -f input.txt --language hi --output results.json

# Analyze inline text with detailed view
python main.py --text "मुझे यह पसंद नहीं है।" --detailed

# Analyze an English file and save results
python main.py report.txt -l en -o report_sentiment.json

# Disable colors (useful for piping output to a file)
python main.py document.txt --no-color > output.txt
```

### Example Output

```
========================================================================
   🌟 Multi-Lingual Sentiment Analysis System (Hindi/English) 🌟
========================================================================

[+] Pipeline successfully loaded on device: cuda

------------------------------------------------------------------------
ANALYSIS RESULTS SUMMARY
------------------------------------------------------------------------
Detected Language:   Hindi (hi) (Confidence: 99.1%)
Original Length:     312 characters
Number of Segments:  3

OVERALL SENTIMENT:
  Dominant Emotion:  [ JOY ] (Confidence: 87.4%)
  Polarity Score:    +0.7812 (Positive)
  Intensity Score:   0.7341 (Strong)

CROSS-CHECK SUMMARY (Dual-Path):
  Agreement Rate:    100.0%
  Verification:      🟢 AGREED (3/3 segments matched)
------------------------------------------------------------------------
```

---

## Training

### Train the Hindi Model

```bash
# Train with default settings
python train.py train data/EmoInHindi/LREC_EmoInHindi.csv

# With custom hyperparameters
python train.py train data/EmoInHindi/LREC_EmoInHindi.csv \
    --epochs 20 \
    --batch-size 16 \
    --lr 2e-5 \
    --output-dir ./checkpoints

# Evaluate a checkpoint
python train.py evaluate data/EmoInHindi/LREC_EmoInHindi.csv checkpoints/best_model.pt
```

### Train the English Model

```bash
# GoEmotions is downloaded automatically from HuggingFace on first run
python train_en.py

# With custom settings
python train_en.py --epochs 10 --batch-size 32 --output-dir checkpoints/en
```

### Training Configuration (`config.py`)

```python
class TrainingConfig:
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2        # effective batch = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 15
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3                           # early stopping

    EMOTION_LOSS_WEIGHT = 1.0
    POLARITY_LOSS_WEIGHT = 0.8
    INTENSITY_LOSS_WEIGHT = 0.8

    OPTIMIZER = "adamw"
    SCHEDULER = "linear"                   # linear warmup + decay
```

### Checkpointing

Each epoch saves `checkpoints/checkpoint_epoch_{N}.pt`. The best-val-loss epoch also writes `checkpoints/best_model.pt`. Checkpoint files contain:

```python
{
    "model_state_dict": ...,
    "emotion_labels": ["anger", "joy", ...],   # full label order
    "num_emotions": 6,
    "model_name": "distilbert-base-multilingual-cased"
}
```

The emotion label list is **persisted in the checkpoint** so inference always uses the exact vocabulary the model was trained on.

---

## API Usage

### Start the Server

```bash
# Development (auto-reload)
python main.py

# Production (Gunicorn + Uvicorn workers)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

Interactive docs: `http://localhost:8000/docs`

---

### Endpoints

#### `GET /api/v1/health`
```bash
curl http://localhost:8000/api/v1/health
```
```json
{ "status": "healthy", "model": "MultiTaskSentimentModel", "device": "cuda" }
```

---

#### `POST /api/v1/analyze` — Aggregate only (no segments)
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "यह फिल्म बहुत अच्छी है!"}'
```
Returns `overall_sentiment` + `metadata` only (no `segments` key).

---

#### `POST /api/v1/analyze_detailed` — Full unified output
#### `POST /api/v1/analyze_json` — Full unified output (same schema)

Both return the complete document with `overall_sentiment`, `metadata`, and `segments[]` as shown in the [Output Schema](#output-schema) section above.

```bash
curl -X POST http://localhost:8000/api/v1/analyze_json \
  -H "Content-Type: application/json" \
  -d '{"text": "यह फिल्म बहुत अच्छी है!"}'
```

---

#### `POST /api/v1/batch_analyze` — Batch of documents
```bash
curl -X POST http://localhost:8000/api/v1/batch_analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["यह फिल्म बहुत अच्छी है!", "मुझे यह पसंद नहीं है।"]}'
```

---

#### `POST /api/v1/analyze_stream` — Server-Sent Events (long documents)
```bash
curl -N http://localhost:8000/api/v1/analyze_stream \
  -H "Content-Type: application/json" \
  -d '{"text": "very long Hindi document..."}'
```

Each SSE event has `type: "metadata"` | `"segment"` | `"complete"` | `"error"`.

---

#### `GET /api/v1/emotions` — Supported emotion labels
```bash
curl http://localhost:8000/api/v1/emotions
```
```json
{ "emotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise"], "emotion_count": 6 }
```

---

## Python API

### Direct Pipeline Usage

```python
from inference.pipeline import SentimentInferencePipeline

pipeline = SentimentInferencePipeline(
    model_path="checkpoints/best_model.pt",         # Hindi model
    english_model_path="checkpoints/en/best_model.pt",
    device="cuda",
    comparison_strategy="agreement_flag",            # or "confidence_weighted_pick" / "average"
)

text = """
मुझे यह फिल्म बहुत पसंद है, यह बेहद शानदार और प्रेरणादायक है।
मैं इस फिल्म को लेकर बहुत खुश हूं और इसे सभी को देखना चाहिए।
"""

# Full unified output: overall_sentiment + metadata + segments
result = pipeline.process_document(text, return_segments=True, segment_id="doc_001")

print(result["overall_sentiment"]["dominant_emotion"])  # e.g. "joy"
print(result["overall_sentiment"]["polarity_score"])    # e.g. 0.82
for seg in result["segments"]:
    print(seg["segment_id"], seg["sentiment"]["label"], seg["cross_check"]["agreement"])
```

### Real-Time / API Inference

```python
from inference.pipeline import RealTimeInference

inference = RealTimeInference(
    model_path="checkpoints/best_model.pt",
    comparison_strategy="confidence_weighted_pick",
)

result = inference.predict("यह एक परीक्षण वाक्य है।")
results = inference.predict_batch(["text 1", "text 2"])
```

### Save Results to JSON

```python
pipeline.save_results(result, "output/analysis.json")
```

---

## Configuration Reference

| Config Class | Key | Default | Description |
|---|---|---|---|
| `ModelConfig` | `MODEL_NAME` | `distilbert-base-multilingual-cased` | Hindi encoder |
| `ModelConfigEN` | `MODEL_NAME` | `bert-base-uncased` | English encoder |
| `ModelConfig` | `MAX_SEQUENCE_LENGTH` | `512` | Max subword tokens |
| `ModelConfig` | `DROPOUT_RATE` | `0.1` | Dropout in all heads |
| `ModelConfig` | `EMOTION_OUTPUT_DIM` | `64` | Emotion head bottleneck |
| `ModelConfig` | `POLARITY_OUTPUT_DIM` | `16` | Polarity head bottleneck |
| `InferenceConfig` | `SEGMENTATION_STRATEGY` | `"sentence"` | `sentence` / `paragraph` / `fixed_length` |
| `InferenceConfig` | `BATCH_SIZE` | `32` | Inference batch size |
| `TranslationConfig` | `MODEL_NAME` | `Helsinki-NLP/opus-mt-hi-en` | MarianMT translation model |
| `TranslationConfig` | `DEVICE` | `"cpu"` | Translation device |
| `APIConfig` | `MAX_TEXT_LENGTH` | `50000` | Max request text length (chars) |
| `DataConfig` | `MIN_TEXT_LENGTH` | `5` | Min chars to keep a segment |

---

## Performance Metrics

Evaluation covers all three tasks simultaneously:

| Task | Metric | Typical Range |
|---|---|---|
| Emotion (multi-label) | F1 micro | 0.78 – 0.85 |
| Emotion (multi-label) | F1 macro | 0.72 – 0.80 |
| Emotion (multi-label) | Subset accuracy | 0.55 – 0.70 |
| Polarity regression | RMSE | 0.15 – 0.20 |
| Polarity regression | Pearson r | 0.80 – 0.90 |
| Intensity regression | RMSE | 0.12 – 0.18 |
| Intensity regression | Pearson r | 0.75 – 0.85 |

> Performance depends heavily on the training data size and quality of polarity-label derivation.

---

## Troubleshooting

### `sentencepiece` not installed
```bash
pip install sentencepiece
```
Required by `Helsinki-NLP/opus-mt-hi-en` tokenizer (MarianMT).

### `indic-nlp-library` not installed
```bash
pip install indic-nlp-library
```
Optional but recommended for better Hindi Unicode normalization. System falls back gracefully without it.

### CUDA out of memory
```python
# Reduce batch sizes in config.py
TrainingConfig.BATCH_SIZE = 8
InferenceConfig.BATCH_SIZE = 8
torch.cuda.empty_cache()
```

### Slow inference (CPU)
- Increase `InferenceConfig.BATCH_SIZE` to amortize tokenization overhead
- Set `TranslationConfig.DEVICE = "cpu"` (already default; translation is the bottleneck)
- Disable the English branch by using `SentimentInferencePipelineEN` directly if cross-checking is not needed

### Model not converging
```python
TrainingConfig.LEARNING_RATE = 1e-5    # lower LR
TrainingConfig.WARMUP_STEPS = 1000     # longer warmup
TrainingConfig.BATCH_SIZE = 8          # smaller batches
```

---

## License

MIT License

---

**Version:** 1.0.0 · **Last Updated:** July 2026