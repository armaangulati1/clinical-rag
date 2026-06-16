from sentence_transformers import CrossEncoder

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # standard small reranker
    return _model

def rerank(question, rows, top_k=5):
    # rows: list of (id, drug, section, content)
    pairs = [(question, r[3]) for r in rows]      # (query, chunk_text) pairs
    scores = _get_model().predict(pairs)          # relevance score per pair
    ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
    return [row for _s, row in ranked[:top_k]]