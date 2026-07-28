# Multilingual Sentiment Analysis & Summarization — Project Documentation

> **Version:** 2.0.0 · **Last Updated:** July 2026  
> A production-grade, dual-path sentiment analysis system for Hindi and English text with a bilingual LLM-based summarizer.

---

## What Makes This Project Unique

### 1. Dual-Path Cross-Verification Engine

Instead of trusting a single model, every request is routed through **two independent inference branches in parallel**, whose results are then reconciled by a custom `ResultComparator`:

| Input Language | Branch A (Primary) | Branch B (Cross-check) |
|---|---|---|
| **Hindi** | Native `MultiTaskSentimentModel` on raw Devanagari | `j-hartmann/emotion-english-distilroberta-base` on MarianMT-translated text |
| **English** | `j-hartmann/emotion-english-distilroberta-base` on raw text | `MultiTaskSentimentModel` on raw English segments (no translation) |

The comparator flags disagreements, reports agreement rate per document, and surfaces the translated text in the API response for full transparency.

---

### 2. Multi-Task Sentiment Formulation

A single transformer encoder backbone drives **three simultaneous task heads** — no separate pipelines:

| Head | Output | Loss |
|---|---|---|
| **EmotionHead** | Sigmoid probabilities per class (multi-label) | Binary Cross-Entropy (BCE) |
| **PolarityHead** | Continuous score ∈ [−1.0, +1.0] via Tanh | Mean Squared Error (MSE) |
| **IntensityHead** | Continuous score ∈ [0.0, +1.0] via Sigmoid | Mean Squared Error (MSE) |

**Why BCE for emotion?** Each emotion is independently true/false for a segment — it's a multi-label classification, not a softmax competition. BCE treats each label independently.

**Why MSE for polarity/intensity?** These are continuous regression targets. MSE minimizes the average squared deviation from the true polarity/intensity value.

---

### 3. Linguistic Bias Mitigation Gates (3 layers)

Standard transformers fail on contextually complex phrases. This system applies three algorithmic corrections:

#### Gate 1 — Context-Aware Balance Gate (in model forward pass)
Detects positive emotion signals dynamically (joy, hopeful, grateful, etc.) and **lifts the polarity score** when a strong positive signal is present alongside negative keywords. This prevents "keyword poisoning" — e.g., *"I struggled a lot but I'm so grateful"* being classified as negative just because of "struggled".

```python
uplift_mask = (positive_signal > gate_threshold).float()
polarity_scores = (1.0 - uplift_mask) * polarity_scores + uplift_mask * clamp(polarity_scores + positive_signal * 0.8)
```

#### Gate 2 — Polarity Pivot Adjuster (post-inference, `utils/preprocessing.py`)
Scans for linguistic pivot conjunctions (`but`, `however`, `लेकिन`, `परंतु`, `मगर`) and **shifts sentiment weight** to the clause following the pivot, where the true emotional intent usually resides.

#### Gate 3 — Ambiguity & Temperature Smoothing Gate (`models/model_utils.py`)
- **Logit temperature smoothing** (T = 1.2) softens overconfident predictions on short/ambiguous text.
- **Ambiguity gate**: if max confidence < 0.45, the polarity score is pulled 70% toward neutral, preventing spurious confident predictions.

---

### 4. Hindi Cross-Check Override (new in v2.0)

The custom Hindi model (`MultiTaskSentimentModel`) is known to suffer from keyword-bias on positive text due to limited training data. A smart reconciliation rule in `ResultComparator` detects this case and **defers to the English translated branch**:

```
IF language == "hi"
AND hindi_valence == "negative"
AND hindi polarity_score < -0.5
AND hindi confidence > 0.85
AND english_valence in ("positive", "neutral")
AND valence_match == False
→ use english_translated as primary_source
```

This prevents cases like a deeply grateful sentence being misclassified as "disgusted" purely because of Hindi negative keywords like "कठिन" (difficult) or "बंद" (closed).

---

### 5. Sentiment-Aware Bilingual Summarizer

The system integrates a locally-quantized `meta-llama/Llama-3.2-3B-Instruct` model (NF4 4-bit via `bitsandbytes`) that generates emotion-aware summaries in both Hindi and English:

- **Sentiment context injection**: The dominant emotion and intensity from classification are injected into the LLM system prompt (e.g., *"Tone Context: The input text conveys a strong anger tone."*).
- **Language-adaptive prompts**: Hindi inputs get Hindi-language prompt templates; English inputs get English templates.
- **Hierarchical summarization**: Long documents (>3500 tokens) are chunked with overlap and merged recursively.
- **Non-blocking execution**: Runs via `run_in_threadpool` so it never blocks FastAPI's async event loop.

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                              │
│          (Next.js Frontend  →  HTTP POST /api/v1/analyze)           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│  api/routes.py  →  run_in_threadpool(inference.predict(...))        │
│  api/middleware.py  →  request logging, size validation, CORS       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SentimentInferencePipeline.process_document()          │
│                                                                     │
│  1. LanguageDetector (langdetect)                                   │
│     → auto-detect "hi" or "en" (or use forced lang from request)   │
│                                                                     │
│  2. DocumentSegmenter                                               │
│     → Hindi: split on Devanagari danda "।"                         │
│     → English: split on sentence boundaries (NLTK punkt)           │
│                                                                     │
│  3. HindiPreprocessor / EnglishPreprocessor                        │
│     → Unicode normalization, noise removal                          │
│                                                                     │
│  4. DUAL-PATH PARALLEL INFERENCE                                    │
│     ┌────────────────────┐     ┌───────────────────────────────┐   │
│     │   Branch A (Hindi) │     │       Branch B (English)      │   │
│     │                    │     │                               │   │
│     │ MultiTaskSentiment │     │  [Hindi input]                │   │
│     │ Model on raw text  │     │  Helsinki-NLP MarianMT        │   │
│     │ (CUDA, GPU)        │     │  translates → English text    │   │
│     │                    │     │  j-hartmann distilroberta     │   │
│     │ EmotionHead  →     │     │  classifies (CPU)             │   │
│     │  sigmoid probs     │     │                               │   │
│     │ PolarityHead →     │     │  [English input]              │   │
│     │  tanh [-1,+1]      │     │  MultiTaskSentimentModel      │   │
│     │ IntensityHead →    │     │  runs on same segments        │   │
│     │  sigmoid [0,1]     │     │  (cross-check, no translate)  │   │
│     └────────┬───────────┘     └───────────────┬───────────────┘   │
│              │                                 │                   │
│              └─────────────┬───────────────────┘                   │
│                            ▼                                        │
│  5. ResultComparator.reconcile() — per segment                     │
│     • Computes: exact_match, valence_match, polarity_proximity      │
│     • Strategy: agreement_flag (default)                            │
│       - Hindi inputs: if strong negative Hindi vs neutral/pos EN    │
│         → override to english_translated (Hindi override v2.0)      │
│       - English inputs: always use english_direct as primary        │
│     • Returns: reconciled emotion/polarity/intensity + cross_check  │
│                                                                     │
│  6. OutputAssembler.format_full_output()                            │
│     • Aggregates segments → overall_sentiment                       │
│     • Builds per-segment formatted output with cross_check details  │
│     • Attaches cross_check_summary (agreement_rate, strategy)       │
│                                                                     │
│  7. BilingualSummarizer (optional, summarize=True)                  │
│     • Adapts sentiment JSON for LLM prompt                          │
│     • Injects emotion tone context into system prompt               │
│     • Llama-3.2-3B-Instruct (4-bit NF4 quantized, CUDA)            │
│     • Returns: summary text                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STRUCTURED JSON RESPONSE                        │
│  overall_sentiment: { dominant_emotion, polarity_score,             │
│    polarity_description, intensity, intensity_description,          │
│    confidence, emotions{}, cross_check_summary{} }                  │
│  metadata: { language, language_confidence, num_segments, ... }     │
│  segments: [ { sentiment, emotion_tags, cross_check{              │
│    hindi_direct, english_translated, translated_text,               │
│    agreement, agreement_detail, primary_source } } ]                │
│  summary: "..." | null                                              │
│  summarizer_status: "active" | "disabled"                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture: MultiTaskSentimentModel

```
Input Text (max 512 tokens)
        │
        ▼
 ModelTokenizer (DistilBERT tokenizer)
        │
        ▼
┌───────────────────────────────────┐
│  Encoder: distilbert-base-        │
│           multilingual-cased      │
│  (hidden_size = 768)              │
│  Shared weights across all tasks  │
└───────────┬───────────────────────┘
            │ pooler_output [batch, 768]
            │ (or last_hidden_state[:,0] as fallback)
     ┌──────┼──────────────┐
     │      │              │
     ▼      ▼              ▼
┌────────┐ ┌──────────┐ ┌───────────┐
│Emotion │ │ Polarity │ │ Intensity │
│  Head  │ │   Head   │ │   Head    │
│        │ │          │ │           │
│Linear  │ │Linear    │ │Linear     │
│768→64  │ │768→16    │ │768→16     │
│ReLU    │ │ReLU      │ │ReLU       │
│Dropout │ │Dropout   │ │Dropout    │
│Linear  │ │Linear    │ │Linear     │
│64→N    │ │16→1      │ │16→1       │
│Sigmoid │ │Tanh      │ │Sigmoid    │
└────────┘ └──────────┘ └───────────┘
     │           │              │
     ▼           ▼              ▼
 P(emotion_i)  score         score
 ∈ [0,1]       ∈ [-1,+1]     ∈ [0,1]
 per class      polarity      intensity

        ↓ Context-Aware Balance Gate (in forward pass)
        ↓ Polarity Pivot Adjuster (post-inference)
        ↓ Ambiguity & Smoothing Gate (AutoSequenceClassifierWrapper)
```

**Loss Function:**
```
L_total = 5.0 × BCE(emotion_logits, emotion_labels)
        + 0.8 × MSE(polarity_scores, polarity_labels)
        + 0.8 × MSE(intensity_scores, intensity_labels)
```

---

## Frontend Architecture (Next.js 16)

```
frontend/app/page.tsx
│
├── State Management
│   ├── inputText           — current textarea value
│   ├── fullSubmittedInput  — complete text submitted (un-truncated)
│   ├── isAnalyzing         — controls progress stepper visibility
│   ├── currentStep (0-2)   — drives dynamic step progress animation
│   ├── hasSubmitted        — toggles results pane
│   ├── inputExpanded       — controls expandable text bubble
│   └── analysisResult      — parsed backend JSON
│
├── UI Components
│   ├── Text Input Area      — auto-resize, Ctrl+Enter submit
│   ├── Expandable Input Bubble
│   │   ├── Shows first 2 lines of submitted text by default
│   │   ├── Down-chevron appears on hover
│   │   └── Click expands/collapses full text with CSS transition
│   ├── Progress Stepper (3 steps, state-driven setTimeout)
│   │   ├── Step 0: "Extracting & Segmenting Text"
│   │   ├── Step 1: "Running Dual Model Inference"
│   │   └── Step 2: "Generating Summary"
│   ├── Sentiment Results Panel
│   │   ├── Dominant Emotion Badge
│   │   ├── Polarity Score Ring
│   │   ├── Emotion Breakdown (from overall_sentiment.emotions)
│   │   └── Cross-Check Summary (agreement_rate, primary_source)
│   └── Summary Section (Llama output)
│
└── API Integration
    ├── POST /api/v1/analyze
    ├── Reads from overall_sentiment.emotions (not emotion_distribution)
    └── Displays translated_text from cross_check when available
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check — model name + device |
| `POST` | `/api/v1/analyze` | Full analysis (overall + optional segments) |
| `POST` | `/api/v1/analyze_detailed` | Same + always includes segments |
| `POST` | `/api/v1/analyze_json` | Raw JSON with full segment cross-check data |
| `POST` | `/api/v1/batch_analyze` | Batch of up to 100 texts |
| `POST` | `/api/v1/analyze_stream` | SSE stream for long documents |
| `POST` | `/api/v1/speak` | Edge-TTS text-to-speech (Hindi/English) |
| `GET` | `/api/v1/emotions` | Supported emotion label list |
| `GET` | `/api/v1/stats` | Model + API statistics |

**Request body** (`/analyze`):
```json
{
  "text": "your text here",
  "language": "hi",
  "return_segments": true,
  "summarize": false
}
```

---

## Reconciliation Strategy Detail

| Condition checked | How |
|---|---|
| Exact label match | `hindi_label == english_label` |
| Valence match | Both map to same group: positive/negative/neutral |
| Polarity proximity | `|hindi_polarity - english_polarity| ≤ 0.20` |
| Low-confidence leniency | Threshold widens to 0.35 if secondary confidence < 0.50 |
| **Hindi override (v2.0)** | If Hindi strongly negative + EN neutral/positive + valence mismatch → use EN result |

`agreement = exact_match OR valence_match OR polarity_proximate`

---

## Known Limitations

| Issue | Root Cause | Status |
|---|---|---|
| Hindi positive text sometimes classified as "neutral" | Helsinki-NLP translation quality is poor on complex gratitude/praise text | Partially mitigated by Hindi override; best fix = retrain Hindi model |
| High inference latency for Hindi (5-7s) | Translation runs on CPU; English cross-check also CPU | Acceptable on 4GB VRAM; use `TRANSLATION_DEVICE=cuda` if headroom exists |
| Llama summarization can be slow (20-40s) | 3B model, 4-bit quantized, CPU offload on small VRAM | Use `summarize=false` when speed matters |
| Hindi custom model undertrained | Small EmoInHindi dataset, limited epochs | Re-training with larger dataset is the long-term fix |

---

## Environment Variables

```env
# backend/.env
HF_TOKEN=your_huggingface_token          # Required for Llama summarizer
MODEL_PATH=checkpoints/best_model.pt     # Custom Hindi checkpoint path
TRANSLATION_DEVICE=cpu                   # cpu or cuda
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```