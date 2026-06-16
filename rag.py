import sys, requests
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from db import init_db, add_chunks, search_chunks, add_fts, vector_search, keyword_search
from rerank import rerank

load_dotenv()
client = OpenAI(timeout=60.0, max_retries=3)
client_inst = instructor.from_openai(client)
EMBED_MODEL, CHAT_MODEL = "text-embedding-3-small", "gpt-4o-mini"
SECTIONS = ["indications_and_usage", "dosage_and_administration", "adverse_reactions",
            "warnings", "contraindications", "drug_interactions"]

def embed(texts):                                   # texts: list[str] -> list[vector]
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def reciprocal_rank_fusion(result_lists, k=60):
    scores, rows = {}, {}
    for results in result_lists:
        for rank, row in enumerate(results):
            cid = row[0]                       # chunk id
            rows[cid] = row
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [rows[cid] for cid in ranked]

def candidate_pool(question, pool=20):
    qvec = embed([question])[0]
    vec = vector_search(qvec, pool)
    kw  = keyword_search(question, pool)
    return reciprocal_rank_fusion([vec, kw])[:pool]   # wide, fused candidate set

def retrieve(question, k=5, pool=20):
    candidates = candidate_pool(question, pool)       # ~20 candidates
    return rerank(question, candidates, top_k=k)      # cross-encoder picks the true best k

def chunk_text(text, size=800, overlap=100):        # NAIVE chunking: fixed-size with overlap
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def fetch_label(drug):
    r = requests.get("https://api.fda.gov/drug/label.json",
                     params={"search": f'openfda.generic_name:"{drug}"', "limit": 1}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise ValueError(f"No openFDA label for '{drug}'")
    return results[0]

def ingest(drugs):
    rows = []
    for drug in drugs:
        print(f"Fetching {drug}...")
        label = fetch_label(drug)
        for section in SECTIONS:
            raw = label.get(section)
            if not raw:
                continue
            text = " ".join(raw) if isinstance(raw, list) else raw
            pieces = chunk_text(text)
            vectors = embed(pieces)                 # batch-embed this section's chunks
            rows += [(drug, section, p, v) for p, v in zip(pieces, vectors)]
    add_chunks(rows)
    print(f"Ingested {len(rows)} chunks from {len(drugs)} drugs.")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="answer grounded ONLY in the provided sources")
    citations: list[int] = Field(default_factory=list,
                                 description="the source numbers [n] actually used to answer")

def _answer_core(question, k=5):
    hits = retrieve(question, k)                              # (id, drug, section, content) x k
    context = "\n\n".join(
        f"[{n}] ({d} — {s}) {c}" for n, (_id, d, s, c) in enumerate(hits, start=1))

    result = client_inst.chat.completions.create(
        model=CHAT_MODEL,
        response_model=AnswerWithCitations,
        messages=[
            {"role": "system", "content":
                "Answer using ONLY the numbered sources below. In 'citations', list the source "
                "numbers you actually used. If the answer isn't in the sources, set answer to a "
                "brief 'I don't know based on the available sources' and citations to []."},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ],
    )

    cites = []
    for n in result.citations:
        if 1 <= n <= len(hits):
            _id, d, s, c = hits[n - 1]
            cites.append({
                "n": n, "drug": d, "section": s, "snippet": c[:200],
                "source": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={d}",
            })
    contexts = [f"[{d} — {s}] {c}" for _id, d, s, c in hits]   # for the eval harness
    return result.answer, contexts, cites

def answer_with_contexts(question, k=5):      # used by your RAGAS/simple eval — unchanged signature
    ans, contexts, _ = _answer_core(question, k)
    return ans, contexts

def answer_cited(question, k=5):              # used by the app/CLI — returns real citations
    ans, _, cites = _answer_core(question, k)
    return ans, cites

def answer(question, k=5):
    text, _ = answer_cited(question, k)
    return text

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        init_db()
        ingest(["metoprolol", "lisinopril", "metformin", "atorvastatin", "warfarin"])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ans, cites = answer_cited(" ".join(sys.argv[2:]))
        print(ans, "\n\nSources:")
        for c in cites:
            print(f"  [{c['n']}] {c['drug']} — {c['section']}  →  {c['source']}")
    else:
        print('Usage: python rag.py ingest   |   python rag.py ask "your question"')