# Multilingual Sentiment Analysis

A production-ready dual-language sentiment analysis system that fine-tunes transformer models on Hindi and English text, runs both in parallel at inference time, and cross-validates results for high-confidence predictions.

## What It Does

Given a Hindi or English document, the system performs dual-path inference:

1. **Language Detection & Segmentation**: The pipeline detects the document's language and segments it using the appropriate language preprocessor (Hindi or English).
2. **Parallel Dual-Path Inference**:
   - **For Hindi Input**:
     - **Hindi direct branch** runs on the raw Devanagari text.
     - **English branch** translates the text to English using MarianMT (`Helsinki-NLP/opus-mt-hi-en`) and runs the English-specific model.
   - **For English Input**:
     - **English branch** runs natively on the raw English text.
     - **Hindi branch** (multilingual) runs directly on the raw English text.
3. **Reconciliation**: Reconciles results from both branches per segment. For Hindi inputs, the Hindi branch acts as primary; for English inputs, the English branch acts as primary (`english_direct`).
4. **Unified Output**: Returns a structured JSON document with overall sentiment, metadata, and per-segment details including cross-check validation data.

## Models Fine-Tuned

| Language | Base Model | Training Dataset |
|---|---|---|
| **Hindi** | `distilbert-base-multilingual-cased` | EmoInHindi (dialogue CSV) |
| **English** | `bert-base-uncased` | GoEmotions (58k Reddit, HuggingFace) |
| **Translation** | `Helsinki-NLP/opus-mt-hi-en` | *(not fine-tuned — used as-is)* |

## Tasks

All models are trained simultaneously on three tasks:
- **Emotion classification** — 6 Ekman emotions (anger, joy, sadness, fear, surprise, disgust)
- **Polarity regression** — continuous −1.0 to +1.0
- **Intensity regression** — continuous 0.0 to 1.0

## Quick Start

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install sentencepiece
python train.py train data/EmoInHindi/LREC_EmoInHindi.csv
python main.py
```

API docs: `http://localhost:8000/docs`

## Repository Layout

```
mul-ling-senti-analysis/
├── backend/                 ← Main Python package (see backend/README.md)
│   ├── README.md            ← Full architecture docs, training guide, API reference
│   ├── config.py
│   ├── main.py              ← FastAPI server
│   ├── train.py             ← Hindi model training
│   ├── train_en.py          ← English model training
│   ├── models/              ← MultiTaskSentimentModel + utilities
│   ├── data/                ← HindiSentimentCSVDataset, GoEmotionsDataset
│   ├── training/            ← Trainer, MultiTaskLoss, metrics
│   ├── inference/           ← Dual-path pipeline, comparator, output assembler
│   ├── utils/               ← Preprocessing, tokenizer, translator, logger
│   └── api/                 ← FastAPI routes, Pydantic schemas, middleware
├── EmoInHindi.zip           ← Raw Hindi training data archive
└── hindi_sentiment_structure.txt
```

## Documentation

See **[`backend/README.md`](backend/README.md)** for:
- Full pipeline architecture diagrams (Hindi direct, English cross-check, dual-path)
- Model and fine-tuning details
- Training data description and format
- Complete output JSON schema
- API endpoint reference
- Configuration reference
- Troubleshooting guide
