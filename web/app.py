# app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os, os.path, re
from vectorstore import VectorStore
from ingest import ingest_file

app = FastAPI(title="Local RAG (fastembed + Chroma)", version="0.3.0")

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

@app.post("/delete_by_path")
def delete_by_path(path: str, exact: bool = False):
    res = VS.delete_by_path(path, exact=exact)
    return {"path": path, **res, "count": VS.count()}

@app.get("/query")
def query(
    q: str = Query(..., description="Your question"),
    k: int = 5,
    include_embeddings: bool = False,
    path_substr: str | None = None,
    mtime_gt: int | None = None,
    mtime_lt: int | None = None,
    max_distance: float | None = None,
    min_score: float | None = None,
):
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")

    # Build 'where' ONLY for fields Chroma 0.5.x supports on metadata
    where = {}
    if mtime_gt is not None or mtime_lt is not None:
        where.setdefault("mtime", {})
        if mtime_gt is not None:
            where["mtime"]["$gt"] = int(mtime_gt)
        if mtime_lt is not None:
            where["mtime"]["$lt"] = int(mtime_lt)
    if not where:
        where = None

    # distance threshold or score threshold (score = 1 - distance)
    distance_threshold = None
    if max_distance is not None:
        distance_threshold = float(max_distance)
    elif min_score is not None:
        distance_threshold = 1.0 - float(min_score)

    res = VS.query_advanced(
        q, k=k, include=include, where=where,
        distance_threshold=distance_threshold, add_scores=True
    )

    # Client-side path substring filter (Chroma metadata doesn't support $contains)
    if path_substr:
        path_substr = str(path_substr)
        metas_rows = res.get("metadatas") or []
        mask_rows = []
        for row in metas_rows:
            mask_rows.append([(m and path_substr in (m.get("path") or "")) for m in row] if row else [])
        # apply the mask to all nested arrays if present
        for key in ["documents", "metadatas", "distances", "embeddings", "uris", "data", "ids", "scores"]:
            if key in res and res[key] is not None:
                new_rows = []
                for row, mask in zip(res[key], mask_rows):
                    if row is None:
                        new_rows.append(row)
                        continue
                    new_rows.append([v for v, keep in zip(row, mask) if keep])
                res[key] = new_rows
        res["applied_path_substr"] = path_substr

    return res


def _extract_answer(text: str, query: str) -> str:
    m = re.search(r'code[^\n:]*:\s*([\w\-]+)', text, flags=re.I)
    if m:
        return m.group(1)
    import re as _re
    sents = _re.split(r'(?<=[.!?])\s+', text.strip())
    q_terms = {w.lower() for w in _re.findall(r"[\w\-]+", query) if len(w) > 2}
    best, best_score = "", -1
    for s in sents:
        terms = {w.lower() for w in _re.findall(r"[\w\-]+", s)}
        score = len(q_terms & terms)
        if score > best_score:
            best_score, best = score, s
    return best

@app.get("/answer")
def answer(q: str, k: int = 5, path_substr: str | None = None):
    res = VS.query_advanced(q, k=k, include=["documents", "metadatas", "distances"], where=None, add_scores=True)

    # optional client-side path filter
    if path_substr:
        metas_rows = res.get("metadatas") or []
        mask_rows = []
        for row in metas_rows:
            mask_rows.append([(m and path_substr in (m.get("path") or "")) for m in row] if row else [])
        for key in ["documents", "metadatas", "distances", "ids", "scores"]:
            if key in res and res[key] is not None:
                new_rows = []
                for row, mask in zip(res[key], mask_rows):
                    if row is None:
                        new_rows.append(row)
                        continue
                    new_rows.append([v for v, keep in zip(row, mask) if keep])
                res[key] = new_rows

    docs = res.get("documents") or []
    if not docs or not docs[0]:
        return {"answer": "", "evidence": [], "note": "no hits"}
    top_text = docs[0][0]
    ans = _extract_answer(top_text, q)
    return {
        "answer": ans,
        "evidence": [{
            "id": (res.get("ids") or [[None]])[0][0],
            "distance": (res.get("distances") or [[None]])[0][0],
            "score": (res.get("scores") or [[None]])[0][0],
            "metadata": (res.get("metadatas") or [[{}]])[0][0],
            "snippet": top_text[:400],
        }],
    }

