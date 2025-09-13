import re
from typing import Dict, List

from ..index.schema import Chunk


def _simple_tokenize(t: str) -> list[str]:
    return re.findall(r"\w+|\S", t)


def _detok(tokens: list[str]) -> str:
    # Simple detokenizer: join with spaces, then compact spaces before punctuation
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s


def chunk_sections(section: Dict, window_tokens: int, overlap_tokens: int) -> List[Chunk]:
    doc_id = section["doc_id"]
    heading_path = section["heading_path"]
    page_no = section.get("page_no")
    text = section["text"]
    meta = section.get("meta", {})

    chunks: List[Chunk] = []

    # Section-level chunk
    sec_id = f"{doc_id}::sec::{hash(heading_path)}::{page_no}"
    chunks.append(
        Chunk(
            doc_id=doc_id,
            chunk_id=sec_id,
            level="section",
            heading_path=heading_path,
            page_no=page_no,
            text=text,
            meta=meta | {"order": 0},
        )
    )

    toks = _simple_tokenize(text)
    stride = max(1, window_tokens - overlap_tokens)
    idx = 0
    order = 1
    while idx < len(toks):
        w = toks[idx : idx + window_tokens]
        if not w:
            break
        ch_text = _detok(w)
        ch_id = f"{doc_id}::chunk::{hash(heading_path)}::{page_no}::{order}"
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=ch_id,
                level="chunk",
                heading_path=heading_path,
                page_no=page_no,
                text=ch_text,
                meta=meta | {"order": order},
            )
        )
        order += 1
        idx += stride
    return chunks
