
from __future__ import annotations

from typing import List, Optional, Tuple

from ..index.schema import Chunk


class Reranker:
    """
    Cross-encoder reranker (CPU-friendly). If torch / sentence-transformers is
    unavailable, this class gracefully disables itself and becomes a no-op.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self.enabled = False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._CrossEncoder = CrossEncoder
            self.enabled = True
        except Exception:
            self._CrossEncoder = None
            self.enabled = False

    def _ensure_model(self):
        if not self.enabled:
            return
        if self._model is None:
            self._model = self._CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> Tuple[List[Chunk], Optional[List[float]]]:
        if not self.enabled or not chunks:
            return chunks, None
        self._ensure_model()
        pairs = [(query, c.text) for c in chunks[:top_k]]
        try:
            scores = self._model.predict(pairs)  # higher is better
        except Exception:
            return chunks, None
        scored = list(zip(chunks[:top_k], scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        sorted_chunks = [c for c, s in scored] + chunks[top_k:]
        return sorted_chunks, [float(s) for s in scores]
