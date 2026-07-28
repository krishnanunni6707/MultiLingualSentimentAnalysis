# Multilingual Sentiment Analysis & Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E.svg)](https://huggingface.co/)

A dual-language, dual-path sentiment analysis system for Hindi and English text. Two transformer models run in parallel for every request — their predictions are cross-validated by a reconciliation engine, and an optional Llama-3.2 summarizer generates bilingual, emotion-aware summaries.

---

## How It Works

```
User Input (Hindi or English)
        │
        ▼
 Language Detection (auto or forced)
        │
   ┌────┴────┐
   │         │
Branch A   Branch B       ← Run in parallel
(native)  (cross-check)
   │         │
   └────┬────┘
        │
  ResultComparator
  (reconcile + flag disagreements)
        │
  OutputAssembler
  (per-segment + overall)
        │
  BilingualSummarizer (optional)
        │
  Structured JSON Response
```

| Input | Branch A | Branch B |
|---|---|---|
| **Hindi** | `MultiTaskSentimentModel` on raw Devanagari | `j-hartmann/distilroberta` on MarianMT translation |
| **English** | `j-hartmann/distilroberta` natively | `MultiTaskSentimentModel` on same segments |

---

## Models

| Role | Model | Notes |
|---|---|---|
| **Hindi sentiment** | `MultiTaskSentimentModel` (custom) | `distilbert-base-multilingual-cased` + 3 task heads, trained on EmoInHindi |
| **English sentiment** | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | 7 Ekman emotions, highly robust |
| **Translation** | [`Helsinki-NLP/opus-mt-hi-en`](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) | MarianMT, Hindi→English, runs on CPU |
| **Summarization** | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 4-bit NF4 quantized, requires `HF_TOKEN` |

---

## Quick Start

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Copy and fill in environment variables
cp .env.example .env   # set HF_TOKEN for Llama summarizer

# Start the API server (loads all models on startup)
python server.py
# → Running on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --port 3001
# → http://localhost:3001
```

### CLI (no server needed)

```bash
cd backend
python main.py --text "यह फिल्म बहुत अच्छी है!" --language hi
python main.py --file my_document.txt --language en
```

---

## Repository Layout

```
mul-ling-senti-analysis/
├── README.md
├── PROJECT_DOCUMENTATION.md       ← Full architecture + design decisions
├── backend/
│   ├── server.py                  ← FastAPI server entry point
│   ├── main.py                    ← CLI entry point
│   ├── config.py                  ← All configuration classes
│   ├── train.py                   ← Hindi model training
│   ├── train_en.py                ← English model training
│   ├── requirements.txt
│   ├── README.md                  ← Full backend developer reference
│   │
│   ├── models/
│   │   ├── sentiment_model.py     ← MultiTaskSentimentModel (encoder + 3 heads)
│   │   └── model_utils.py         ← AutoSequenceClassifierWrapper, load_model()
│   │
│   ├── inference/
│   │   ├── pipeline.py            ← SentimentInferencePipeline + RealTimeInference
│   │   ├── pipeline_en.py         ← English standalone inference
│   │   ├── comparator.py          ← ResultComparator (reconciliation logic)
│   │   ├── output_assembler.py    ← Segment aggregation + JSON formatting
│   │   └── segmentation.py        ← DocumentSegmenter (Hindi/English)
│   │
│   ├── multilingual_summarizer/
│   │   ├── summarizer.py          ← BilingualSummarizer (Llama-3.2)
│   │   └── config.py              ← Prompts + generation params
│   │
│   ├── utils/
│   │   ├── preprocessing.py       ← HindiPreprocessor, PolarityPivotAdjuster
│   │   ├── preprocessing_en.py    ← English preprocessing
│   │   ├── translator.py          ← HindiToEnglishTranslator (MarianMT)
│   │   ├── tokenizer.py           ← HindiWordTokenizer (Indic-NLP)
│   │   └── logger.py              ← Centralized logging with ms timestamps
│   │
│   ├── api/
│   │   ├── routes.py              ← FastAPI router (all endpoints)
│   │   ├── schemas.py             ← Pydantic request/response models
│   │   └── middleware.py          ← Request logging, timing, CORS
│   │
│   ├── data/                      ← Dataset loaders (Hindi + English)
│   ├── training/                  ← Trainer, MultiTaskLoss, metrics
│   ├── EngModel/                  ← Standalone English training scripts
│   └── checkpoints/               ← Saved model weights (gitignored)
│
└── frontend/
    ├── app/
    │   ├── page.tsx               ← Main UI (analysis + results)
    │   ├── layout.tsx             ← Root layout
    │   └── globals.css            ← Global styles
    └── package.json
```

---

## API Endpoints

```bash
# Health check
GET  /api/v1/health

# Analyze text (auto language detection)
POST /api/v1/analyze
  Body: { "text": "...", "language": "hi|en|auto", "return_segments": bool, "summarize": bool }

# Full detailed output with per-segment cross-check
POST /api/v1/analyze_detailed

# Batch analysis (up to 100 texts)
POST /api/v1/batch_analyze
  Body: { "texts": ["...", "..."] }

# Text-to-Speech
POST /api/v1/speak
  Body: { "text": "...", "language": "hi|en" }

# Supported emotion labels
GET  /api/v1/emotions
```

Example response:
```json
{
  "overall_sentiment": {
    "dominant_emotion": "anger",
    "polarity_score": -0.72,
    "polarity_description": "Highly negative",
    "intensity": 0.84,
    "intensity_description": "Strong",
    "confidence": 0.91,
    "emotions": { "anger": 0.89, "disgust": 0.43, "joy": 0.02, ... },
    "cross_check_summary": {
      "strategy_used": "agreement_flag",
      "num_segments": 3,
      "num_agree": 3,
      "agreement_rate": 1.0
    }
  },
  "metadata": { "language": "en", "num_segments": 3, ... },
  "segments": [
    {
      "sentiment": { "label": "anger", "polarity_score": -0.75, ... },
      "cross_check": {
        "hindi_direct":      { "label": "angry", "polarity_score": -0.69, ... },
        "english_translated": { "label": "anger", "polarity_score": -0.77, ... },
        "translated_text":   null,
        "agreement":         true,
        "primary_source":    "english_direct"
      }
    }
  ],
  "summary": "The customer expresses strong anger...",
  "summarizer_status": "active"
}
```

---

## Environment Variables

Create `backend/.env`:

```env
HF_TOKEN=your_huggingface_token          # Required for Llama summarizer
MODEL_PATH=checkpoints/best_model.pt     # Optional: custom Hindi checkpoint
TRANSLATION_DEVICE=cpu                   # cpu or cuda
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

---

## Documentation

| File | Contents |
|---|---|
| [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) | Full architecture, design decisions, known limitations |
| [`backend/README.md`](backend/README.md) | Installation, training, API reference, Python API, troubleshooting |

---

## License

MIT — see [LICENSE](LICENSE).
