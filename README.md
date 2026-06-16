# Clinical Drug-Safety RAG

RAG over FDA drug labels with hybrid retrieval, cross-encoder/LLM re-ranking, structured citations, FastAPI + Streamlit UI, and RAGAS evaluation.

Observability via Langfuse; Level-1 assertions run in CI on every commit.

![evals](https://github.com/armaangulati1/clinical-rag/actions/workflows/evals.yml/badge.svg)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pip install -r requirements-dev.txt for eval + cross-encoder
# set OPENAI_API_KEY, DATABASE_URL, LANGFUSE_* in .env
python rag.py ingest
python rag.py ask "What are metoprolol side effects?"
uvicorn api:app --reload
streamlit run app.py
pytest -q
```

See `RESULTS.md` for naive vs hybrid vs rerank RAGAS scores.
