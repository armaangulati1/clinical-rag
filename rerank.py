import os
import instructor
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
RERANKER = os.environ.get("RERANKER", "cross-encoder")   # set RERANKER=llm on Render

_ce_model = None
_client_inst = None


class RankedChunks(BaseModel):
    ranked: list[int] = Field(description="source numbers ordered best-first")


def _get_cross_encoder():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _ce_model


def _get_client():
    global _client_inst
    if _client_inst is None:
        _client_inst = instructor.from_openai(OpenAI(timeout=60.0, max_retries=3))
    return _client_inst


def _rerank_cross_encoder(question, rows, top_k=5):
    pairs = [(question, r[3]) for r in rows]
    scores = _get_cross_encoder().predict(pairs)
    ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
    return [row for _s, row in ranked[:top_k]]


def _rerank_llm(question, rows, top_k=5):
    if not rows:
        return []
    context = "\n\n".join(
        f"[{n}] ({d} — {s}) {c[:400]}" for n, (_id, d, s, c) in enumerate(rows, start=1))
    result = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        response_model=RankedChunks,
        messages=[
            {"role": "system", "content":
                "Rank the numbered sources by relevance to the question. "
                "Return only valid source numbers in 'ranked', best first."},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
        ],
    )
    out, seen = [], set()
    for n in result.ranked:
        if 1 <= n <= len(rows):
            row = rows[n - 1]
            if row[0] not in seen:
                out.append(row)
                seen.add(row[0])
        if len(out) >= top_k:
            return out[:top_k]
    for row in rows:
        if row[0] not in seen:
            out.append(row)
            seen.add(row[0])
        if len(out) >= top_k:
            break
    return out[:top_k]


def rerank(question, rows, top_k=5):
    if RERANKER == "llm":
        return _rerank_llm(question, rows, top_k)
    try:
        return _rerank_cross_encoder(question, rows, top_k)
    except ImportError:
        return _rerank_llm(question, rows, top_k)
