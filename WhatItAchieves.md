## What It Achieves

**Extraction**
- Converts NCRB's PDF tables into a clean 36×12 numeric feature matrix (36 states, 12 crime/judicial metrics) — something that doesn't exist anywhere in machine-readable form
- Does this without LLM hallucination: RAG retrieves the right chunk, regex pulls the number deterministically

**Retrieval**
- Dual-index RAG separates numeric lookups ("Kerala's conviction rate") from explanatory queries ("why did cybercrime rise") — most RAG systems conflate these and fail on both
- Every output is traceable to a source chunk

**Prediction + Explanation**
- XGBoost identifies which judicial metrics (conviction rate, charge-sheeting rate, police disposal) most predict a state's overall crime rate
- SHAP makes this interpretable — not just "what's the prediction" but "what drove it"

**Unified interface**
- One natural language query returns a grounded factual answer + a prediction + a SHAP explanation together

---

## Real-World Relevance

- **Policy analysts** currently extract NCRB figures manually — this automates that entirely
- **Judiciary/administration** can see quantitatively whether their bottleneck is policing, charge-sheeting, or conviction — different problems needing different interventions
- **Reproducible methodology** applicable to any Indian government statistical PDF (NFHS, Economic Survey, SRS)

---

## Honest Limitations

- 36 data points makes predictions directional signals, not precise forecasts
- SHAP shows correlation, not causation — improving conviction rates won't necessarily reduce crime
- Static snapshot — not real-time