from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..index.schema import Chunk


@dataclass
class RerankedChunk:
    """A chunk paired with its own rerank score at all times — never two
    separately-ordered lists that a caller has to zip together by position."""

    chunk: Chunk
    score: Optional[float]  # None when this chunk wasn't actually scored
    # (reranker disabled/unavailable, or beyond the top_k that got reranked)


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

    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[RerankedChunk]:
        if not self.enabled or not chunks:
            return [RerankedChunk(chunk=c, score=None) for c in chunks]
        self._ensure_model()
        pairs = [(query, c.text) for c in chunks[:top_k]]
        try:
            scores = self._model.predict(pairs)  # higher is better
        except Exception:
            return [RerankedChunk(chunk=c, score=None) for c in chunks]

        # Build (chunk, score) pairs and sort THAT — score and chunk move
        # together, so there is no separate list that can drift out of sync.
        scored = sorted(
            zip(chunks[:top_k], (float(s) for s in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        reranked = [RerankedChunk(chunk=c, score=s) for c, s in scored]
        # Candidates beyond top_k were never scored; pass them through
        # untouched rather than silently dropping them.
        reranked.extend(RerankedChunk(chunk=c, score=None) for c in chunks[top_k:])
        return reranked
