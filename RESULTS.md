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
