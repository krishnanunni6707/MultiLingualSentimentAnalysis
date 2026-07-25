# Multilingual Sentiment Analysis & Summarization

A dual-language sentiment analysis system for Hindi and English text. It runs two independent inference branches in parallel, cross-validates their results, and optionally generates a sentiment-aware abstractive summary.

## What It Does

Given a Hindi or English document, the system:

1. **Detects language** and segments the text with language-specific preprocessing.
2. **Runs dual-path inference in parallel**:
   - **Hindi input** — Hindi model on raw Devanagari text; English model on MarianMT translation (`Helsinki-NLP/opus-mt-hi-en`).
   - **English input** — English model on raw text; Hindi multilingual model on the same segments (cross-check, no translation).
3. **Reconciles branches** per segment using configurable strategies (`agreement_flag` by default).
4. **Summarizes** (optional) — feeds sentiment metadata into `meta-llama/Llama-3.2-3B-Instruct` for a bilingual, emotion-aware summary.

Output is a structured JSON document with overall sentiment, per-segment cross-check data, metadata, and an optional `summary` field.

## Models

| Role | Default Model | Notes |
|---|---|---|
| **Hindi sentiment** | [`vashuag/HindiEmotion`](https://huggingface.co/vashuag/HindiEmotion) | 7-class Devanagari emotion classifier |
| **English sentiment** | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | 7 Ekman + neutral labels |
| **Translation** | [`Helsinki-NLP/opus-mt-hi-en`](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) | Frozen MarianMT, used for Hindi → English cross-check |
| **Summarization** | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Requires `HF_TOKEN`; 4-bit quantized on CUDA |

Custom fine-tuned checkpoints (`MultiTaskSentimentModel`) are also supported. See [Training](#training) in [`backend/README.md`](backend/README.md).

At inference time, HuggingFace classifiers are wrapped by `AutoSequenceClassifierWrapper`, which derives polarity and intensity from emotion probabilities (logit temperature smoothing at 1.2).

## Quick Start

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Download inference models to local HuggingFace cache (first run)
# Hindi, English, translation, and optionally Llama for summarization

# CLI — analyze a file
python main.py --file test-input_hindi.txt

# CLI — analyze inline text
python main.py --text "This movie is absolutely wonderful!" --language en
```

Set `HF_TOKEN` in `backend/.env` if you want the summarizer enabled. Without it (or without a local Llama cache), sentiment analysis still works; `summarizer_status` will be `"disabled"`.

## Repository Layout

```
mul-ling-senti-analysis/
├── README.md
├── backend/                          ← Main Python package (see backend/README.md)
│   ├── main.py                       ← CLI entry point (sentiment + summary)
│   ├── config.py
│   ├── train.py                      ← Hindi MultiTaskSentimentModel training
│   ├── train_en.py                   ← English MultiTaskSentimentModel training
│   ├── models/                       ← MultiTaskSentimentModel + model loading utilities
│   ├── data/                         ← EmoInHindi & GoEmotions datasets
│   ├── training/                     ← Trainer, losses, metrics
│   ├── inference/                    ← Dual-path pipeline, comparator, output assembler
│   ├── multilingual_summarizer/      ← Llama-based bilingual summarizer
│   ├── utils/                        ← Preprocessing, tokenizer, translator, logger
│   ├── api/                          ← FastAPI routes, schemas, middleware
│   └── EngModel/                     ← Standalone English model training scripts
├── implementation plan.md            ← Original dual-path architecture plan
├── Summarizing_model_Architecture.txt
└── hindi_sentiment_structure.txt
```

## Documentation

See **[`backend/README.md`](backend/README.md)** for:

- Full pipeline architecture (Hindi direct, English cross-check, dual-path routing)
- Inference vs. training model details
- Output JSON schema (including `summary` and `summarizer_status`)
- CLI and Python API usage
- REST API endpoint reference
- Training guide
- Configuration and troubleshooting

## Environment Variables

Create `backend/.env`:

```env
HF_TOKEN=your_huggingface_token          # Required for Llama summarizer
MODEL_PATH=checkpoints/best_model.pt     # Optional custom Hindi checkpoint
TRANSLATION_DEVICE=cpu                   # cpu or cuda
API_HOST=0.0.0.0
API_PORT=8000
```

## Troubleshooting

**CUDA out of memory**

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}
```

Reduce batch sizes in `backend/config.py` (`TrainingConfig.BATCH_SIZE`, `InferenceConfig.BATCH_SIZE`).

**Clear Python cache**

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

**Models not found** — Inference loads from the local HuggingFace cache (`local_files_only=True`). Pre-download models before first use.

## License

MIT — see [LICENSE](LICENSE).
