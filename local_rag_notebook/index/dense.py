from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .schema import Chunk, Hit

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # noqa: F401

class DenseIndexer:
    def __init__(self, index_dir: Path, model_name: str):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.faiss = None
        self.chunk_id_order: list[str] = []
        self._emb_matrix: Optional[np.ndarray] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype="float32")

    def build(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        self.chunk_id_order = [c.chunk_id for c in chunks]
        X = self._embed(texts)  # [N, D]
        np.save(self.index_dir / "embeddings.npy", X)
        (self.index_dir / "chunk_ids.json").write_text(json.dumps(self.chunk_id_order), encoding="utf-8")

        if faiss is not None:
            dim = X.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(X)
            faiss.write_index(index, str(self.index_dir / "faiss.index"))

    def load(self) -> "DenseIndexer":
        self.chunk_id_order = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        if faiss is not None and (self.index_dir / "faiss.index").exists():
            self.faiss = faiss.read_index(str(self.index_dir / "faiss.index"))
            self._emb_matrix = None
        else:
            self.faiss = None
            self._emb_matrix = np.load(self.index_dir / "embeddings.npy")
        return self

    def search(self, query: str, top_k: int = 40) -> List[Hit]:
        q = self._embed([query])
        if self.faiss is not None:
            scores, idxs = self.faiss.search(q, top_k)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()
        else:
            M = self._emb_matrix
            assert M is not None, "Embeddings matrix not loaded"
            sims = (q @ M.T)[0]
            if top_k >= sims.shape[0]:
                top_idx = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, top_k)[:top_k]
                top_idx = part[np.argsort(-sims[part])]
            idxs = top_idx.tolist()
            scores = sims[idxs].tolist()

        hits = []
        for i, s in zip(idxs, scores):
            if i == -1:
                continue
            cid = self.chunk_id_order[i]
            hits.append(Hit(chunk_id=cid, score=float(s)))
        return hits
