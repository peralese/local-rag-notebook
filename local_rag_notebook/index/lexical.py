from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from .schema import Chunk, Hit


def _tok(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


class LexicalIndexer:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.bm25: BM25Okapi | None = None
        self.chunk_id_order: list[str] = []
        self._chunk_map: dict[str, Chunk] = {}

    def build(self, chunks: List[Chunk]):
        self.chunk_id_order = [c.chunk_id for c in chunks]
        docs = [_tok(c.text) for c in chunks]
        bm25 = BM25Okapi(docs)
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        (self.index_dir / "chunk_ids_lex.json").write_text(
            json.dumps(self.chunk_id_order), encoding="utf-8"
        )
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as out:
            for c in chunks:
                out.write(c.model_dump_json() + "\n")

    def load(self) -> "LexicalIndexer":
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.chunk_id_order = json.loads(
            (self.index_dir / "chunk_ids_lex.json").read_text(encoding="utf-8")
        )
        return self

    def _ensure_chunk_map(self):
        if self._chunk_map:
            return
        mp: dict[str, Chunk] = {}
        path = self.index_dir / "chunks.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f, start=1):
                s = ln.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                except json.JSONDecodeError as e:
                    # Recovery for "trailing characters" or accidental noise:
                    end = s.rfind("}")
                    if end != -1:
                        try:
                            data = json.loads(s[: end + 1])
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to parse JSONL line {i} in {path}: {e2}"
                            ) from e
                    else:
                        raise RuntimeError(f"Failed to parse JSONL line {i} in {path}: {e}") from e
                c = Chunk(**data)
                mp[c.chunk_id] = c
        self._chunk_map = mp

    def search(self, query: str, top_k: int = 40) -> List[Hit]:
        assert self.bm25 is not None, "BM25 not loaded"
        scores = self.bm25.get_scores(_tok(query))
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        hits = []
        for i, s in pairs[:top_k]:
            hits.append(Hit(chunk_id=self.chunk_id_order[i], score=float(s)))
        return hits

    def load_chunks_by_ids(self, ids: List[str]) -> List[Chunk]:
        self._ensure_chunk_map()
        return [self._chunk_map[i] for i in ids if i in self._chunk_map]
