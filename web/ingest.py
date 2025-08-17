# ingest.py
import os, hashlib
from typing import Optional, Dict, Any, List
from vectorstore import VectorStore
from chunker import chunk_markdown

TEXT_EXTS = {'.txt','.md','.py','.json','.csv','.yaml','.yml','.toml'}

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _chunk_and_ids(text: str, path: str) -> tuple[list[str], list[str]]:
    chunks = chunk_markdown(text, max_chars=1200, overlap=150)
    ids: List[str] = []
    for i, c in enumerate(chunks):
        h = hashlib.sha1((path + '::' + str(i) + '::' + c).encode('utf-8', errors='ignore')).hexdigest()
        ids.append(h)
    return chunks, ids

def _file_metadata(path: str, i: int, total: int) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "chunk_index": i,
        "chunk_count": total,
    }

def ingest_file(path: str, vs: Optional[VectorStore] = None) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in TEXT_EXTS:
        return {"path": path, "status": "skipped", "reason": f"unsupported extension {ext}"}

    text = _read_text(path)
    if not text.strip():
        return {"path": path, "status": "empty", "added": 0}

    docs, ids = _chunk_and_ids(text, path)
    metas = [_file_metadata(path, i, len(docs)) for i in range(len(docs))]

    vs = vs or VectorStore()
    vs.add(ids=ids, documents=docs, metadatas=metas)
    return {"path": path, "status": "ok", "added": len(docs)}
