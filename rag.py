import sys, requests
from openai import OpenAI
from dotenv import load_dotenv
from db import init_db, add_chunks, search_chunks, add_fts, vector_search, keyword_search

load_dotenv()
client = OpenAI(timeout=60.0, max_retries=3)
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

def hybrid_search(question, k=5, pool=20):
    qvec = embed([question])[0]
    vec = vector_search(qvec, pool)            # top 20 by meaning
    kw  = keyword_search(question, pool)       # top 20 by keywords
    return reciprocal_rank_fusion([vec, kw])[:k]   # merge, take best k

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

def answer_with_contexts(question, k=5):
    hits = hybrid_search(question, k)
    contexts = [f"[{d} — {s}] {c}" for _id, d, s, c in hits]
    context = "\n\n".join(contexts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content":
                "Answer using ONLY the context provided. If the answer isn't in the context, "
                "say you don't know. Cite the [drug — section] tags you used."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, contexts

def answer(question, k=5):
    text, _ = answer_with_contexts(question, k)
    return text

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        init_db()
        ingest(["metoprolol", "lisinopril", "metformin", "atorvastatin", "warfarin"])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        print(answer(" ".join(sys.argv[2:])))
    else:
        print('Usage: python rag.py ingest   |   python rag.py ask "your question"')