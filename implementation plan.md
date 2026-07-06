# Implementation Plan: Dual-Path Hindi Sentiment Analysis

## 1. What this plan covers

Your diagram adds a **second analysis path** on top of what's already built. Today's
system (per `README.md` / `config.py`) does one thing: take Hindi text, run it through
one multi-task model, get JSON out. The target architecture runs **two independent
analyses per input and reconciles them**:

```
Input text (.txt / .pdf / .word)
        │
        ├──────────────────────────┐
        ▼                          ▼
Language detection            Raw Hindi text
  + translate to English            │
        │                           │
        ▼                           ▼
Sentiment LLM (English)     Sentiment LLM (Hindi, direct)
        │                           │
        └───────────┬───────────────┘
                     ▼
        Compare & reconcile outputs
                     ▼
              Final JSON result
```

Three new components don't exist yet in your codebase:
1. A **translation module** (Hindi → English)
2. A **second sentiment/emotion model** that runs on English text
3. A **comparator/merger** that reconciles the two results into one output

Everything else (segmentation, the existing Hindi `MultiTaskSentimentModel`, the CSV
data pipeline, checkpointing) stays as-is and becomes "Path B" in the diagram.

---

## 2. Current state — what already exists vs. what's missing

Based on the `README.md` and `config.py` you shared:

**Already implemented:**
- `models/sentiment_model.py` — multi-task model (Emotion/Polarity/Intensity heads) on `distilbert-base-multilingual-cased`
- `data/dataset.py` — `HindiSentimentCSVDataset`, loads EmoInHindi-style CSVs, derives polarity from an `emotions` + `emoIntensity` column via `DEFAULT_EMOTION_POLARITY`
- `training/` — trainer, losses, metrics
- `inference/pipeline.py`, `inference/segmentation.py` — single-path inference producing the target JSON schema
- `config.py` — `ModelConfig`, `TrainingConfig`, `DataConfig`, `InferenceConfig`, `APIConfig`

**Explicitly missing / blocking (per your own README notes):**
- `api/routes.py`, `api/schemas.py`, `api/middleware.py` — **`main.py` currently cannot start** without these
- No translation module anywhere in the structure
- No second model for English-side analysis
- No comparator/merge logic
- `utils/tokenizer.py` is described as "word-level Hindi tokenization" — separate from the subword tokenizer used by the model; clarify whether this is still needed once translation is added, or if it was for a pre-transformer approach

**Decision needed before writing code:** the two sentiment LLMs will almost certainly use
**different emotion label vocabularies** (your Hindi model's vocab is built dynamically
from whatever appears in the training CSV; an English model — pretrained or fine-tuned —
will have its own fixed label set). Section 5 below covers how to reconcile that; you'll
need to pick an approach before the comparator can do anything meaningful.

---

## 3. New component 1: Translation module (Hindi → English)

### Model choice
| Option | Trade-off |
|---|---|
| `Helsinki-NLP/opus-mt-hi-en` | Small, fast, good enough for MVP — **recommended starting point** |
| `ai4bharat/indictrans2-indic-en-1B` | Best quality for Hindi specifically, but heavier (1B params) — upgrade path once MVP works |
| `facebook/nllb-200-distilled-600M` | Multilingual, decent quality, useful if you later add other Indic languages beyond Hindi |

Start with `opus-mt-hi-en` to get the pipeline working end-to-end, then swap to
IndicTrans2 later — the module below is written so the model name is the only thing
that changes.

### New file: `utils/translator.py`

```python
class HindiToEnglishTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-hi-en", device: str = "cpu"):
        ...  # loads AutoTokenizer + AutoModelForSeq2SeqLM

    def translate(self, text: str) -> str:
        ...  # returns English translation of a single segment

    def translate_batch(self, texts: list[str]) -> list[str]:
        ...  # batched version for efficiency across segments
```

### Config addition (`config.py`)
```python
class TranslationConfig:
    MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    DEVICE = "cuda"
```

### Where this plugs in
Called once per segment (or batched across all segments of a document) right after
segmentation, before the English-path model runs. Its output text is **only** used for
Path A — Path B keeps using the original Hindi text untouched.

---

## 4. New component 2: English-side sentiment/emotion model

You have two real options here — this is the biggest open decision in the whole plan.

### Option A — Use a pretrained English emotion model off-the-shelf (recommended for MVP)
No training required. Candidates:
- `j-hartmann/emotion-english-distilroberta-base` — 7 labels (anger, disgust, fear, joy, neutral, sadness, surprise) — **closely matches your existing `DEFAULT_EMOTION_LABELS`**, which makes reconciliation in Section 5 much easier
- `SamLowe/roberta-base-go_emotions` — 28 fine-grained labels, richer but harder to align with your Hindi model's vocab

This gives you emotion classification only — no native polarity/intensity heads. You'd
derive polarity/intensity the same way `data/dataset.py` already derives them from
`DEFAULT_EMOTION_POLARITY` (intensity-weighted sign of the detected emotion), just
applied to this model's output instead of a CSV column.

### Option B — Fine-tune your own English multi-task model
Same three-head architecture as `models/sentiment_model.py`, swapped to an English base
encoder (e.g. `roberta-base` or `distilbert-base-uncased`), trained on an English
emotion+intensity dataset (e.g. a GoEmotions-derived set with synthetic intensity, or a
VAD-style affect dataset). More consistent with Path B's output shape, but real
additional training work.

**Recommendation:** ship Option A first so the full pipeline runs end-to-end, then
revisit Option B only if reconciliation quality in Section 5 turns out to need it.

### New file: `models/english_sentiment_model.py`
```python
class EnglishSentimentModel:
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base", device: str = "cpu"):
        ...  # wraps a transformers pipeline("text-classification", ..., top_k=None)

    def predict(self, text: str) -> dict:
        ...  # returns {label, confidence, emotion_tags, polarity_score, intensity}
             # polarity/intensity derived via the same DEFAULT_EMOTION_POLARITY-style mapping
```

### Config addition (`config.py`)
```python
class EnglishModelConfig:
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    CONFIDENCE_THRESHOLD = 0.35  # mirrors InferenceConfig.CONFIDENCE_THRESHOLD
```

---

## 5. New component 3: Comparator / merger

This is the box in your diagram labeled "Compare and contrast the final output." It
takes Path A's result (English, translated) and Path B's result (Hindi, direct) for the
**same segment** and produces one reconciled entry.

### The label-alignment problem (must be solved first)
Path B's emotion vocabulary is whatever appears in your training CSV (dynamic, per
`HindiSentimentCSVDataset.emotion_vocab`). Path A's vocabulary is fixed by whichever
English model you chose in Section 4. These will not match label-for-label out of the
box. You need a **label mapping table** before the comparator can compare anything
meaningfully:

```python
# utils/label_mapping.py
HINDI_TO_ENGLISH_EMOTION_MAP = {
    "sad": "sadness",
    "anger": "anger",
    "confident": "neutral",     # example only — you must build this from your actual
    "anticipation": "surprise", # Hindi emotion_vocab once it's known
    # ...
}
```
This mapping is a manual step you'll do once your Hindi model's `emotion_vocab` is
finalized from real training data — it can't be derived automatically.

### Merge strategy — pick one (recommend starting with the first)
1. **Confidence-weighted pick**: take whichever path has higher `confidence` for that
   segment as the primary result; keep both raw results nested under a `cross_check` key
   for transparency
2. **Averaging**: average `polarity_score` and `intensity` across both paths (only
   sensible once emotion labels are aligned via the mapping above)
3. **Agreement flag only**: keep Path B (direct Hindi) as the primary result always,
   and just report whether Path A agrees, as a confidence signal — simplest to implement,
   good MVP choice since it avoids needing the label mapping to be perfect

### New file: `inference/comparator.py`
```python
class ResultComparator:
    def __init__(self, strategy: str = "agreement_flag"):
        ...

    def reconcile(self, hindi_result: dict, english_result: dict) -> dict:
        ...  # returns the merged segment dict, shape defined in Section 6
```

---

## 6. Updated output schema

Extends the JSON shape already in your README with a `cross_check` block so you don't
lose either path's result:

```json
{
  "segment_id": "doc_001_seg_000",
  "text_span": "char_span: 0-30",
  "sentiment": {
    "label": "joy",
    "polarity_score": 0.81,
    "intensity": 0.77,
    "confidence": 0.92
  },
  "emotion_tags": ["joy", "surprise"],
  "language_detected": "hi",
  "translation_used": true,
  "cross_check": {
    "hindi_direct": {"label": "joy", "polarity_score": 0.85, "confidence": 0.90},
    "english_translated": {"label": "joy", "polarity_score": 0.77, "confidence": 0.88},
    "agreement": true,
    "strategy_used": "agreement_flag"
  }
}
```

---

## 7. Where this plugs into the existing pipeline

`inference/pipeline.py` currently does (roughly): segment → tokenize → model → format
JSON. It becomes:

```python
class SentimentInferencePipeline:
    def __init__(self, ...):
        self.hindi_model = ...          # existing MultiTaskSentimentModel
        self.translator = HindiToEnglishTranslator(...)     # new
        self.english_model = EnglishSentimentModel(...)     # new
        self.comparator = ResultComparator(...)             # new

    def process_document(self, text, ...):
        segments = self.segmenter.split(text)
        results = []
        for seg in segments:
            hindi_result = self._run_hindi_path(seg.text)
            english_text = self.translator.translate(seg.text)
            english_result = self._run_english_path(english_text)
            merged = self.comparator.reconcile(hindi_result, english_result)
            results.append(merged)
        return results
```

Both paths can run independently and in parallel (they don't depend on each other's
output) — worth batching translation calls and English-model calls separately across all
segments of a document rather than looping segment-by-segment, for speed.

---

## 8. Phased build order

Do these in order — each phase is independently testable before moving to the next.

- [ ] **Phase 0 — Unblock the server.** Build the missing `api/` package
  (`routes.py`, `schemas.py`, `middleware.py`) so `main.py` actually starts. Nothing
  else in this plan matters if the API can't boot.
- [ ] **Phase 1 — Translation module.** Build `utils/translator.py`, test it standalone
  on a handful of Hindi sentences, confirm translation quality is acceptable before
  building anything downstream of it.
- [ ] **Phase 2 — English sentiment model.** Build `models/english_sentiment_model.py`
  using Option A (pretrained). Test standalone on the translated outputs from Phase 1.
- [ ] **Phase 3 — Label mapping.** Once your Hindi model's actual `emotion_vocab` is
  known from real training data, build `utils/label_mapping.py` by hand.
- [ ] **Phase 4 — Comparator.** Build `inference/comparator.py` starting with the
  `agreement_flag` strategy (simplest, doesn't require the mapping to be perfect).
- [ ] **Phase 5 — Wire into pipeline.** Update `inference/pipeline.py` to run both paths
  and call the comparator. Update the API response schema (`api/schemas.py`) to include
  `cross_check`.
- [ ] **Phase 6 — End-to-end test.** Run a handful of documents through
  `/api/v1/analyze_json` and manually check whether `cross_check.agreement` correlates
  with cases where you'd intuitively expect the two paths to disagree (sarcasm,
  idioms, code-mixed slang — translation tends to fail on exactly these).

---

## 9. Open questions to resolve before/during implementation

1. Which merge strategy (Section 5) do you want as the default — agreement-flag,
   confidence-weighted pick, or averaging?
2. Is `utils/tokenizer.py` (word-level Hindi tokenization) still needed anywhere once
   translation is in place, or was it built for an earlier non-transformer approach?
3. Do you want Phase 0 (the missing `api/` package) done as part of this work, or is
   that being handled separately? It blocks server startup regardless of this feature.
4. For PDF/Word input (shown in your diagram) — is text extraction from `.pdf`/`.docx`
   already handled somewhere, or does that also need to be built as a preprocessing
   step before segmentation?