from fastapi import FastAPI
from pydantic import BaseModel
from langfuse import get_client
from rag import answer_cited

app = FastAPI(title="Clinical Drug-Safety RAG API")

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    answer, citations = answer_cited(req.question)
    get_client().flush()
    return {"answer": answer, "citations": citations}