# Backend — Multilingual Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E.svg)](https://huggingface.co/)

**Version:** 2.0.0 · **Last Updated:** July 2026

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Architecture](#model-architecture)
- [Reconciliation Engine](#reconciliation-engine)
- [Output JSON Schema](#output-json-schema)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [CLI Usage](#cli-usage)
- [REST API Reference](#rest-api-reference)
- [Python API](#python-api)
- [Training Guide](#training-guide)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The system performs three interconnected sentiment tasks from a **single forward pass** through a shared encoder:

| Task | Type | Output |
|---|---|---|
| **Emotion Classification** | Multi-label sigmoid | Probability per emotion ∈ [0, 1] |
| **Polarity Regression** | Continuous | Score ∈ [−1.0, +1.0] |
| **Intensity Regression** | Continuous | Score ∈ [0.0, +1.0] |

Every request goes through **two independent inference branches** that are then reconciled by `ResultComparator`. This dual-path design catches model-specific failures (e.g., keyword bias in the custom Hindi model) and surfaces them transparently via the `cross_check` field.

---

## Pipeline Architecture

### Dual-Path Flow

```
Raw Input Text
      │
      ▼
LanguageDetector (langdetect)
      │
      ├─── language = "hi" ──────────────────────────────────────────┐
      │                                                               │
      │    HindiPreprocessor + DocumentSegmenter (danda "।")         │
      │                                                               │
      │    Branch A (GPU)                  Branch B (CPU)            │
      │    ─────────────────               ─────────────────          │
      │    MultiTaskSentimentModel         Helsinki-NLP/opus-mt-hi-en │
      │    on raw Devanagari segments      translates → English       │
      │                                   j-hartmann/distilroberta   │
      │                                   classifies translated text  │
      │                                                               │
      └─── language = "en" ──────────────────────────────────────────┤
                                                                      │
           EnglishPreprocessor + DocumentSegmenter (NLTK punkt)      │
                                                                      │
           Branch A (CPU)                  Branch B (GPU)            │
           ─────────────────               ─────────────────          │
           j-hartmann/distilroberta        MultiTaskSentimentModel    │
           on raw English segments         on same English segments   │
           (primary, english_direct)       (cross-check, no translate)│
                                                                      │
      ┌───────────────────────────────────────────────────────────────┘
      │
      ▼
ResultComparator.reconcile() — per segment
      │  • exact_match, valence_match, polarity_proximity
      │  • Hindi override: if Hindi strongly-negative + EN neutral/pos → use EN
      ▼
OutputAssembler.format_full_output()
      │  • Aggregate segments → overall_sentiment
      │  • Attach cross_check_summary
      ▼
BilingualSummarizer (optional)
      │  • Llama-3.2-3B-Instruct, 4-bit NF4
      │  • Emotion-aware prompt injection
      ▼
Structured JSON Response
```

### Segment-Level Processing

```
Raw segment text
      │
      ▼
ModelTokenizer.batch_encode()  (max_length=512)
      │
      ▼
MultiTaskSentimentModel.forward()
      │
      ├── EmotionHead   → sigmoid probabilities per class
      ├── PolarityHead  → tanh score ∈ [−1, +1]
      └── IntensityHead → sigmoid score ∈ [0, +1]
      │
      ▼
Context-Aware Balance Gate (inside forward())
  If positive emotion signal > gate_threshold:
      polarity ← uplift toward positive
      (prevents keyword poisoning from negative words in positive text)
      │
      ▼
PolarityPivotAdjuster.adjust()
  Detects pivot conjunctions: "but", "however", "लेकिन", "परंतु"
  Shifts sentiment weight to post-pivot clause
      │
      ▼
AutoSequenceClassifierWrapper (for HuggingFace models only)
  • Temperature smoothing (T=1.2)
  • Ambiguity gate: if max_conf < 0.45 → pull polarity 70% to neutral
```

---

## Model Architecture

### MultiTaskSentimentModel

```
Input: CLS pooled representation [batch, 768]
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
EmotionHead      PolarityHead    IntensityHead
 Linear(768→64)   Linear(768→16)  Linear(768→16)
 ReLU             ReLU            ReLU
 Dropout(0.1)     Dropout(0.1)    Dropout(0.1)
 Linear(64→N)     Linear(16→1)    Linear(16→1)
 Sigmoid          Tanh → [-1,1]   Sigmoid → [0,1]
 Loss: BCE        Loss: MSE       Loss: MSE
```

**Why BCE for emotions?** Each emotion is an independent binary question (is this text angry? is this text joyful?). BCE treats each label independently rather than forcing one winner via softmax.

**Why MSE for polarity/intensity?** These are continuous regression targets. MSE penalizes deviation from the true score proportionally.

**Training loss:**
```
L_total = 5.0 × BCE(emotion_logits, emotion_labels)
        + 0.8 × MSE(polarity_scores, polarity_labels)
        + 0.8 × MSE(intensity_scores, intensity_labels)
```

The emotion weight is set high (5.0) because multi-label emotion prediction is sparse — most labels are zero, so the signal is weaker and needs amplification.

---

## Reconciliation Engine

`inference/comparator.py` — `ResultComparator.reconcile()`

### Agreement Criteria (any one = agreed)
| Criterion | Check |
|---|---|
| Exact match | `hindi_label == english_label` |
| Valence match | Both map to same group: positive/negative/neutral |
| Polarity proximity | `|hindi_polarity − english_polarity| ≤ 0.20` |

`surprise` is mapped to **neutral** valence — it is polarity-ambiguous (can be wonder or shock) and systematically caused false mismatches when treated as positive.

When secondary branch confidence < 0.50, the polarity proximity threshold widens to 0.35 (lenient gate).

### Strategies

| Strategy | Behaviour |
|---|---|
| `agreement_flag` *(default)* | Use language-appropriate primary branch; apply Hindi override rule when needed |
| `confidence_weighted_pick` | Pick whichever branch has higher confidence |
| `average` | Average emotion probs, polarity, intensity, confidence across branches |

### Hindi Override Rule (v2.0)

The custom Hindi model can misclassify positive/grateful text as strongly negative due to keyword bias (limited training data). When all of these are true simultaneously, the English translated branch is used as primary instead:

```
hindi_valence == "negative"
AND hindi polarity_score < −0.5
AND hindi confidence > 0.85
AND english_valence in ("positive", "neutral")
AND valence_match == False
→ primary_source = "english_translated"
```

---

## Output JSON Schema

```json
{
  "overall_sentiment": {
    "emotions": {
      "anger": 0.12,
      "joy": 0.64,
      "sadness": 0.08,
      "fear": 0.03,
      "surprise": 0.10,
      "disgust": 0.02,
      "neutral": 0.01
    },
    "dominant_emotion": "joy",
    "polarity_score": 0.71,
    "polarity_description": "Positive",
    "intensity": 0.78,
    "intensity_description": "Strong",
    "confidence": 0.89,
    "num_segments": 3,
    "cross_check_summary": {
      "strategy_used": "agreement_flag",
      "num_segments": 3,
      "num_agree": 3,
      "num_disagree": 0,
      "agreement_rate": 1.0
    }
  },
  "metadata": {
    "language": "hi",
    "language_confidence": 0.9999,
    "original_length": 349,
    "cleaned_length": 345,
    "num_segments": 3,
    "segment_strategy": "sentence"
  },
  "segments": [
    {
      "segment_id": "doc_001_seg_000",
      "text_span": "char_span: 0-120",
      "sentiment": {
        "label": "joy",
        "polarity_score": 0.75,
        "intensity": 0.82,
        "confidence": 0.91
      },
      "emotion_tags": ["joy", "surprise"],
      "language_detected": "hi",
      "translation_used": true,
      "cross_check": {
        "hindi_direct": {
          "label": "joy",
          "polarity_score": 0.73,
          "intensity": 0.80,
          "confidence": 0.90
        },
        "english_translated": {
          "label": "joy",
          "polarity_score": 0.77,
          "intensity": 0.84,
          "confidence": 0.88
        },
        "translated_text": "The film was excellent and I loved every moment of it.",
        "agreement": true,
        "agreement_detail": {
          "exact_label_match": true,
          "valence_match": true,
          "polarity_diff": 0.04,
          "polarity_proximate": true,
          "polarity_threshold_used": 0.2,
          "secondary_confidence": 0.88,
          "low_confidence_gate": 0.5
        },
        "strategy_used": "agreement_flag",
        "primary_source": "hindi_direct"
      }
    }
  ],
  "summary": "The text expresses strong positive sentiment about...",
  "summarizer_status": "active"
}
```

**`polarity_description` values:** Highly negative | Negative | Neutral | Positive | Highly positive  
**`intensity_description` values:** Very weak | Weak | Moderate | Strong | Very strong  
**`summarizer_status` values:** `active` | `disabled`

---

## Project Structure

```
backend/
├── server.py                     ← FastAPI app + CORS + lifespan
├── main.py                       ← CLI entry point
├── config.py                     ← ModelConfig, TrainingConfig, InferenceConfig,
│                                    TranslationConfig, APIConfig
├── train.py                      ← Hindi MultiTaskSentimentModel training CLI
├── train_en.py                   ← English model training CLI
├── requirements.txt
│
├── models/
│   ├── sentiment_model.py        ← MultiTaskSentimentModel
│   │                               (EmotionHead, PolarityHead, IntensityHead)
│   └── model_utils.py            ← AutoSequenceClassifierWrapper
│                                    ModelTokenizer, load_model()
│
├── inference/
│   ├── pipeline.py               ← SentimentInferencePipeline
│   │                               RealTimeInference
│   │                               adapt_sentiment_output_to_summarizer()
│   ├── pipeline_en.py            ← SentimentInferencePipelineEN
│   ├── comparator.py             ← ResultComparator (reconcile, summarize)
│   ├── output_assembler.py       ← OutputAssembler (aggregate, format)
│   └── segmentation.py           ← DocumentSegmenter (Hindi/English)
│
├── multilingual_summarizer/
│   ├── summarizer.py             ← BilingualSummarizer (Llama-3.2-3B)
│   └── config.py                 ← Prompt templates, generation params
│
├── utils/
│   ├── preprocessing.py          ← HindiPreprocessor, LanguageDetector,
│   │                               PolarityPivotAdjuster
│   ├── preprocessing_en.py       ← EnglishPreprocessor
│   ├── translator.py             ← HindiToEnglishTranslator (MarianMT)
│   ├── tokenizer.py              ← HindiWordTokenizer (Indic-NLP)
│   └── logger.py                 ← setup_logger(), get_logger()
│
├── api/
│   ├── routes.py                 ← All FastAPI endpoints
│   ├── schemas.py                ← Pydantic models (request/response)
│   └── middleware.py             ← Request logging, timing, size limits
│
├── data/
│   ├── dataset.py                ← HindiSentimentCSVDataset (EmoInHindi)
│   ├── dataset_en.py             ← GoEmotionsDataset
│   ├── dataloader.py             ← Hindi data loaders
│   └── dataloader_en.py          ← English data loaders
│
├── training/
│   ├── trainer.py                ← Training loop, eval, checkpointing,
│   │                               early stopping, custom Trainer class
│   ├── losses.py                 ← MultiTaskLoss
│   └── metrics.py                ← F1, MSE, Pearson correlation
│
├── EngModel/                     ← Standalone English training utilities
│   ├── train.py
│   ├── data_processing.py
│   ├── clean-data.py
│   └── train-test-split.py
│
├── checkpoints/                  ← Model weights (gitignored)
│   ├── best_model.pt             ← Hindi model checkpoint
│   └── en/best_model.pt          ← English model checkpoint (optional)
│
└── logs/
    ├── app.log                   ← Server logs with ms-precision timestamps
    └── server.out                ← Process stdout/stderr
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ with ≥ 4 GB VRAM (recommended; CPU-only is supported but slow)
- HuggingFace account + token (for Llama summarizer)

### Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### Environment Variables

Create `backend/.env`:

```env
HF_TOKEN=your_huggingface_token
MODEL_PATH=checkpoints/best_model.pt
TRANSLATION_DEVICE=cpu
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### Pre-download Models

Models load from the local HuggingFace cache (`local_files_only=True`). Download before first run:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

for name in [
    "j-hartmann/emotion-english-distilroberta-base",
    "Helsinki-NLP/opus-mt-hi-en",
    "distilbert-base-multilingual-cased",
]:
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(name)
```

For summarization, also cache Llama (requires HF_TOKEN with Llama access):
```bash
huggingface-cli login
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"
```

---

## Running the Server

```bash
cd backend
python server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8000

# Run in background (production-style):
nohup python server.py > logs/server.out 2>&1 &
```

Interactive docs: `http://localhost:8000/docs`  
Health check: `http://localhost:8000/api/v1/health`

The server loads all models **once at startup** and keeps them resident in memory for the lifetime of the process. Inference calls route through `run_in_threadpool` to avoid blocking FastAPI's async event loop.

---

## CLI Usage

```bash
cd backend

# Analyze a text file (auto language detection)
python main.py --file document.txt

# Analyze inline text
python main.py --text "यह फिल्म बहुत अच्छी है!"

# Force language
python main.py --file report.txt --language en

# Control summary length
python main.py --file report.txt --length short
python main.py --file report.txt --length long
```

| Flag | Short | Description |
|---|---|---|
| `--file` | `-f` | Path to a `.txt` file |
| `--text` | `-t` | Inline text string |
| `--language` | `-l` | Force: `hi` or `en` |
| `--length` | | Summary length hint: `short` / `medium` / `long` |

---

## REST API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/analyze` | Analyze + optional segments + optional summary |
| `POST` | `/api/v1/analyze_detailed` | Always includes per-segment breakdown |
| `POST` | `/api/v1/analyze_json` | Raw JSON — same as `analyze_detailed` |
| `POST` | `/api/v1/batch_analyze` | Batch up to 100 texts |
| `POST` | `/api/v1/analyze_stream` | SSE stream for long documents |
| `POST` | `/api/v1/speak` | Edge-TTS text-to-speech |
| `GET` | `/api/v1/emotions` | Supported emotion labels |
| `GET` | `/api/v1/stats` | Model + API statistics |

### Request Body (`/analyze`)

```json
{
  "text": "your text here",
  "language": "hi",
  "return_segments": true,
  "summarize": false
}
```

`language` accepts: `"hi"`, `"en"`, `"auto"`, `""`, `null` (all auto-detect).  
`summarize: true` adds ~20-40 seconds latency for the Llama model.

### cURL Example

```bash
# Quick analysis (no summary)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "यह फिल्म बहुत अच्छी थी।", "summarize": false}'

# With segments and summary
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "return_segments": true, "summarize": true}'
```

---

## Python API

```python
from inference.pipeline import SentimentInferencePipeline, RealTimeInference

# Full pipeline
pipeline = SentimentInferencePipeline(
    model_path="checkpoints/best_model.pt",          # Hindi model
    english_model_path=None,                          # auto-resolves j-hartmann
    comparison_strategy="agreement_flag",             # or "confidence_weighted_pick" / "average"
)

result = pipeline.process_document(
    text="मुझे यह फिल्म बहुत पसंद है।",
    return_segments=True,
    segment_id="doc_001",
    language="hi",
    summarize=False,
)

print(result["overall_sentiment"]["dominant_emotion"])
print(result["overall_sentiment"]["polarity_score"])
print(result["segments"][0]["cross_check"]["primary_source"])
print(result["segments"][0]["cross_check"]["translated_text"])

# Lightweight wrapper (used by API)
inference = RealTimeInference()
result = inference.predict("I love this product!", language="en")
results = inference.predict_batch(["text one", "text two"])
```

---

## Training Guide

### Train the Hindi Model

```bash
# Basic training
python train.py train data/EmoInHindi/LREC_EmoInHindi_reduced.csv

# Full options
python train.py train data/EmoInHindi/LREC_EmoInHindi_reduced.csv \
    --epochs 20 \
    --batch-size 16 \
    --lr 2e-5 \
    --output-dir ./checkpoints

# Resume from checkpoint
python train.py train data/EmoInHindi/ --checkpoint checkpoints/best_model.pt

# Evaluate
python train.py evaluate data/EmoInHindi/ checkpoints/best_model.pt
```

### Train the English Model

```bash
python train_en.py
python train_en.py --epochs 10 --batch-size 32 --output-dir checkpoints/en
```

GoEmotions is downloaded automatically from HuggingFace on first run.

### Training Data

| Language | Dataset | Size | Format |
|---|---|---|---|
| **Hindi** | EmoInHindi (LREC) | ~7,000 utterances | CSV: `utterance`, `emotions`, `emoIntensity` |
| **English** | GoEmotions (Google) | ~58,000 comments | HuggingFace `datasets`, 27 labels |

### Checkpoint Format

```python
{
    "model_state_dict": ...,       # PyTorch state dict
    "emotion_labels": ["anger", "joy", ...],
    "num_emotions": N,
    "model_name": "distilbert-base-multilingual-cased"
}
```

Best-val-loss checkpoint → `checkpoints/best_model.pt`  
Per-epoch → `checkpoints/checkpoint_epoch_{N}.pt`

---

## Configuration Reference

| Class | Key | Default | Description |
|---|---|---|---|
| `ModelConfig` | `MODEL_NAME` | `distilbert-base-multilingual-cased` | Hindi encoder |
| `ModelConfigEN` | `MODEL_NAME` | `bert-base-uncased` | English training encoder |
| `ModelConfig` | `MAX_SEQUENCE_LENGTH` | `512` | Token limit |
| `ModelConfig` | `DROPOUT_RATE` | `0.1` | Dropout across all heads |
| `InferenceConfig` | `SEGMENTATION_STRATEGY` | `"sentence"` | `sentence` / `paragraph` / `fixed_length` |
| `InferenceConfig` | `BATCH_SIZE` | `32` | Inference batch size |
| `InferenceConfig` | `CONFIDENCE_THRESHOLD` | `0.6` | Ambiguity gate threshold |
| `TranslationConfig` | `MODEL_NAME` | `Helsinki-NLP/opus-mt-hi-en` | MarianMT translation model |
| `TranslationConfig` | `DEVICE` | `"cpu"` | Translation device |
| `APIConfig` | `MAX_TEXT_LENGTH` | `50000` | Max request chars |
| `TrainingConfig` | `BATCH_SIZE` | `1` | Training batch size |
| `TrainingConfig` | `PATIENCE` | `3` | Early stopping patience |
| `TrainingConfig` | `EMOTION_LOSS_WEIGHT` | `5.0` | BCE weight in multi-task loss |

---

## Troubleshooting

### CUDA out of memory
```bash
# Kill all processes using GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}

# Or set CUDA_VISIBLE_DEVICES="" to force CPU
CUDA_VISIBLE_DEVICES="" python server.py
```

The English cross-check model already runs on CPU by design to protect VRAM for the Hindi model + Llama summarizer. If you have more than 6 GB VRAM, you can set `english_device = "cuda"` in `pipeline.py`.

### Models not found (local_files_only error)
Pre-download models to the HuggingFace cache before first run. See [Pre-download Models](#pre-download-models).

### Summarizer disabled
Set `HF_TOKEN` in `backend/.env` and ensure `meta-llama/Llama-3.2-3B-Instruct` is cached locally. Sentiment analysis works without the summarizer — `summarizer_status` will be `"disabled"`.

### Slow inference
- Translation is the biggest bottleneck for Hindi inputs (~1-2s on CPU).
- Set `TRANSLATION_DEVICE=cuda` if VRAM headroom allows.
- Use `summarize=false` for fast responses (skips ~20-40s Llama inference).

### Hindi positive text showing as "neutral"
The Helsinki-NLP translator produces imperfect English for complex Hindi text, causing the English cross-check model to return neutral. The Hindi override rule prevents the worst-case outcome (disgusted/angry classification), but the ideal fix is **retraining the Hindi model** on a larger, more diverse dataset.

### Clear Python cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### indic-nlp-library not installed
```bash
pip install indic-nlp-library
```
Optional but recommended for Hindi Unicode normalization. The system falls back gracefully without it.

---

## License

MIT License — see [LICENSE](../LICENSE).

**Version:** 2.0.0 · **Last Updated:** July 2026