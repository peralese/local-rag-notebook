# app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os, os.path, re
from vectorstore import VectorStore
from ingest import ingest_file

app = FastAPI(title="Local RAG (fastembed + Chroma)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

VS = VectorStore()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/reset")
def reset():
    VS.reset()
    return {"status": "reset", "count": VS.count()}

@app.get("/count")
def count():
    return {"count": VS.count()}

@app.post("/ingest_all")
def ingest_all(root: str = "/data/uploads"):
    processed = 0
    results = []
    if os.path.isdir(root):
        for r, _, files in os.walk(root):
            for name in files:
                path = os.path.join(r, name)
                try:
                    res = ingest_file(path, VS)
                    results.append(res)
                    processed += res.get("added", 0)
                except Exception as e:
                    results.append({"path": path, "status": "error", "error": str(e)})
    return {"root": root, "processed": processed, "details": results, "count": VS.count()}

@app.post("/ingest_file")
def ingest_one(path: str):
    res = ingest_file(path, VS)
    return {"result": res, "count": VS.count()}

@app.get("/query")
def query(q: str, k: int = 5, include_embeddings: bool = False):
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")
    return VS.query(q, k=k, include=include)

def _extract_answer(text: str, query: str) -> str:
    m = re.search(r'code[^\n:]*:\s*([\w\-]+)', text, flags=re.I)
    if m:
        return m.group(1)
    import re as _re
    sents = _re.split(r'(?<=[.!?])\s+', text.strip())
    q_terms = {w.lower() for w in _re.findall(r"[\w\-]+", query) if len(w) > 2}
    best = ""
    best_score = -1
    for s in sents:
        terms = {w.lower() for w in _re.findall(r"[\w\-]+", s)}
        score = len(q_terms & terms)
        if score > best_score:
            best_score = score
            best = s
    return best

@app.get("/answer")
def answer(q: str, k: int = 5):
    res = VS.query(q, k=k, include=["documents", "metadatas", "distances"])
    # IDs still come back in res["ids"] even though not in include
    docs = res.get("documents") or []
    if not docs or not docs[0]:
        return {"answer": "", "evidence": [], "note": "no hits"}
    top_text = docs[0][0]
    ans = _extract_answer(top_text, q)
    return {
        "answer": ans,
        "evidence": [
            {
                "id": (res.get("ids") or [[None]])[0][0],
                "distance": (res.get("distances") or [[None]])[0][0],
                "metadata": (res.get("metadatas") or [[{}]])[0][0],
                "snippet": top_text[:400],
            }
        ],
    }
