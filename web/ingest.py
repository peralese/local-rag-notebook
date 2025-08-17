# ingest.py
import os, hashlib
from typing import Optional, Dict, Any
from vectorstore import VectorStore

def _read_text(path: str) -> str:
    # Minimal reader: TXT/MD/PY/etc. (you can add PDF later)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def ingest_file(path: str, vs: Optional[VectorStore] = None) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    text = _read_text(path)
    if not text.strip():
        return {"path": path, "status": "empty", "added": 0}

    # Deterministic id per file content
    file_bytes = text.encode("utf-8", errors="ignore")
    doc_id = hashlib.sha1(file_bytes).hexdigest()

    vs = vs or VectorStore()
    vs.add(ids=[doc_id], documents=[text], metadatas=[{"path": path, "size": os.path.getsize(path)}])
    return {"path": path, "status": "ok", "added": 1}
