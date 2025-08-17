# chunker.py
import re
from typing import List

def _split_headers(text: str) -> List[str]:
    parts = re.split(r'(?m)^(?=\s{0,3}#{1,6} )', text)
    return [p for p in parts if p.strip()]

def _split_paragraphs(block: str) -> List[str]:
    paras = re.split(r'\n\s*\n+', block)
    return [p.strip() for p in paras if p.strip()]

def _split_sentences_piecewise(text: str, max_chars: int, overlap: int):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    buf = ""
    for s in sents:
        if not s:
            continue
        if len(s) > max_chars:
            for i in range(0, len(s), max_chars):
                frag = s[i:i+max_chars]
                if buf:
                    out.append(buf); buf = ""
                out.append(frag)
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = buf + " " + s
        else:
            out.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + " " + s).strip()
    if buf:
        out.append(buf)
    return out

def _merge_with_overlap(pieces: List[str], max_chars: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    buf = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            for sent_chunk in _split_sentences_piecewise(piece, max_chars=max_chars, overlap=overlap):
                chunks.append(sent_chunk)
            continue
        if not buf:
            buf = piece
        elif len(buf) + 2 + len(piece) <= max_chars:
            buf = buf + "\n\n" + piece
        else:
            if buf:
                chunks.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n\n" + piece).strip()
            if len(buf) > max_chars:
                for sent_chunk in _split_sentences_piecewise(buf, max_chars=max_chars, overlap=overlap):
                    chunks.append(sent_chunk)
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks

def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    blocks = _split_headers(text)
    pieces = []
    for b in blocks:
        pieces.extend(_split_paragraphs(b))
    return _merge_with_overlap(pieces, max_chars=max_chars, overlap=overlap)
