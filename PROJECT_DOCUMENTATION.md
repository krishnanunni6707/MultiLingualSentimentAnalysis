# what is the speciality or unique ness this project provides?

# 1. Dual-Path Cross-Verification (Reliability Engine)

   Instead of relying on a single model's prediction, the system routes inputs through a dual-path parallel verification architecture to validate results:

   For Hindi Inputs: The text is analyzed by a native Hindi model, while in parallel translating it to English (via MarianMT) to be analyzed by an English model.

   For English Inputs: The text is analyzed by a native English model, while in parallel running a multilingual sentiment model on the raw English text.

A custom ResultComparator reconciliation engine matches their outputs (checking for exact matches, valence alignment, or polarity proximity) to flags disagreements or choose the highest-confidence path.

# 2. Unified Multi-Task Sentiment Formulation

   Rather than running separate pipelines, the model shares a single transformer encoder backbone with three specialized task heads:

   ### Emotion Classification (Multi-Label): 
    Outputs sigmoid probabilities for multiple co-occurring emotions (e.g., detecting both surprise and joy).

   ### Polarity Regression: 
   Produces a continuous valence score from $[-1.0, 1.0]$ (Negative $\leftrightarrow$ Neutral $\leftrightarrow$ Positive).
   
   ### Intensity Regression: 
   Produces a continuous strength score from $[0.0, 1.0]$.

# 3. Linguistic & Contextual Bias Mitigation Gates

   Standard transformers often suffer from polarity misclassifications on complex phrases. This system solves this with three unique algorithmic gates:

   ### Valence Collapse Mitigation Gate: 
   Prevents hopeful or resilient text containing negative words (e.g., "I have struggled but I am hopeful") from being classified as negative. It detects positive signals and dynamically lifts the polarity score.

   ### Polarity Pivot Adjuster: 
   Explicitly scans for pivot conjunctions (like "but", "however", "लेकिन", "परंतु") and shifts the sentiment weight to contextually relevant clauses.

   ### Ambiguity & Smoothing Gates: 
    Uses logit temperature smoothing ($T = 1.2$) to prevent over-confidence on short inputs and drags polarity to $0.0$ (neutral) if the dominant emotion confidence falls below $0.45$.

# 4. Sentiment-Aware Bilingual Summarizer
   The system integrates a local, quantized (NF4 4-bit) Llama-3.2-3B-Instruct model to generate document summaries:

   ### Sentiment Context Injection: 
    The dominant emotion and intensity scores from the classification step are dynamically injected into the LLM system prompt (e.g., “Tone Context: The input text conveys a strong joy tone.”). This ensures the summary matches the emotional tone of the source material.

   ### Hierarchical Summarization: 
    Seamlessly summarizes long documents by using overlap-aware chunking and a recursive multi-stage summary combine.