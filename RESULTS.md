# Naive RAG Baseline (Section 5.2)

Evaluated with RAGAS on 5 test questions (`testset.py`) using cosine vector search (k=5) over openFDA drug labels.

**Run:** `python eval_ragas.py` → `eval_baseline.csv`

## Aggregate scores

| Metric | Naive RAG | 2026 "good" threshold |
|---|---|---|
| Faithfulness | **0.75** | ≥ 0.75 |
| Context precision (LLM, no reference) | **0.80** | ≥ 0.70 |
| Context recall | **1.00** | ≥ 0.80 |
| Answer relevancy | *not measured* | ≥ 0.80 |

## Per-question notes

| Question | Faithfulness | Precision | Recall | Notes |
|---|---|---|---|---|
| Metoprolol adverse reactions | 1.00 | ~1.00 | 1.00 | Strong |
| Atorvastatin indications | 1.00 | ~1.00 | 1.00 | Strong |
| Warfarin drug interactions | — | ~1.00 | 1.00 | Faithfulness job hit max_tokens |
| Metformin indications | 0.00 | 0.00 | 1.00 | Retriever pulled adverse-reaction chunks; model said "I don't know" |
| Warfarin monitoring | 1.00 | ~1.00 | 1.00 | Strong |

## Takeaway

Aggregate scores meet the baseline thresholds, but the naive retriever has clear failure modes (e.g. metformin). Section 5.3 goal: beat these numbers with hybrid retrieval + RRF re-ranking.

---

# Hybrid RAG (Section 5.3)

Evaluated with RAGAS on the same testset using vector + keyword search fused with RRF (pool=20, k=5).

**Run:** `python eval_ragas.py` → `eval_hybrid.csv`

## Aggregate scores

| Metric | Naive RAG | Hybrid RAG | 2026 "good" threshold |
|---|---|---|---|
| Faithfulness | 0.75 | **0.78** | ≥ 0.75 |
| Context precision (LLM, no reference) | 0.80 | **0.64** | ≥ 0.70 |
| Context recall | 1.00 | **1.00** | ≥ 0.80 |

## Takeaway

Hybrid search improved faithfulness but lowered context precision on this small testset. See `eval_hybrid.csv` for per-question breakdown.
