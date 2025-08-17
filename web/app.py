# app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os, os.path
from vectorstore import VectorStore
from ingest import ingest_file

app = FastAPI(title="Local RAG (fastembed + Chroma)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Single shared VectorStore instance
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
def query(q: str = Query(..., description="Your question"), k: int = 5):
    out = VS.query(q, k=k)
    return out
