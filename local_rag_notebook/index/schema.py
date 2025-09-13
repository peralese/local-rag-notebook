from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    level: str                 # "doc" | "section" | "chunk"
    heading_path: Optional[str]
    page_no: Optional[int]
    text: str
    meta: dict                 # e.g., {"file_path": "...", "order": int}

class Hit(BaseModel):
    chunk_id: str
    score: float

class CorpusIndex(BaseModel):
    chunk_id_order: List[str]
