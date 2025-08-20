
from __future__ import annotations
from typing import List, Tuple, Union
import re
from ..index.schema import Chunk

def _dehyphenate(text: str) -> str:
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

def _format_heading(hp: Union[str, List[str], None]) -> str:
    if not hp:
        return ""
    if isinstance(hp, (list, tuple)):
        parts = [str(h) for h in hp if h]
        return " > ".join(parts)
    return str(hp)

def extract_answer(
    question: str,
    contexts: List[Chunk],
    max_chars: int = 1500,
    join_with: str = "\n\n",
    include_headings: bool = True,
    dehyphenate: bool = True,
) -> Tuple[str, List[Chunk]]:
    pieces: List[str] = []
    used: List[Chunk] = []
    for ch in contexts:
        txt = ch.text or ""
        if dehyphenate:
            txt = _dehyphenate(txt)

        if include_headings:
            heading = _format_heading(getattr(ch, "heading_path", None))
            if heading:
                txt = f"{heading}:\n{txt}"

        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        if not txt:
            continue

        budget = max_chars - sum(len(p) for p in pieces) - (len(join_with) if pieces else 0)
        if budget <= 0:
            break

        if len(txt) > budget:
            txt = txt[:budget].rstrip()

        pieces.append(txt)
        used.append(ch)

        if sum(len(p) for p in pieces) >= max_chars:
            break

    answer = join_with.join(pieces).strip()
    return answer, used
