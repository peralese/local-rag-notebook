import re
from typing import List, Tuple
from .classify import classify
from ..index.schema import Chunk

BULLET_RE = re.compile(r'^\s*[-*]\s+(.*)')

def _sentences(text: str) -> list[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _window_around_query(text: str, query: str, width: int = 2) -> str:
    sents = _sentences(text)
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2]
    hit_idx = None
    for i, s in enumerate(sents):
        sl = s.lower()
        if any(t in sl for t in toks):
            hit_idx = i
            break
    if hit_idx is None:
        return " ".join(sents[: min(3, len(sents))])
    lo = max(0, hit_idx - width)
    hi = min(len(sents), hit_idx + width + 1)
    return " ".join(sents[lo:hi])

def _extract_bullets(text: str, max_items: int = 10) -> List[str]:
    bullets = []
    for ln in text.splitlines():
        m = BULLET_RE.match(ln)
        if m:
            item = m.group(1).strip()
            if item and item not in bullets:
                bullets.append(item)
                if len(bullets) >= max_items:
                    break
    return bullets

def extract_answer(query: str, contexts: List[Chunk]) -> Tuple[str, List[Chunk]]:
    mode = classify(query)
    used = []
    if not contexts:
        return "No matching content found.", used

    if mode == "list":
        # Try to pull real bullets from top contexts
        agg = []
        for ch in contexts:
            bullets = _extract_bullets(ch.text, max_items=10)
            if not bullets:
                # fallback to sentence window
                bullets = [_window_around_query(ch.text, query, width=1)]
            for b in bullets:
                if b not in agg:
                    agg.append(b)
            used.append(ch)
            if len(agg) >= 10:
                break
        answer = "Here are relevant items:\n" + "\n".join(f"- {b}" for b in agg[:10])
        return answer, used

    if mode == "compare":
        parts = []
        for i, ch in enumerate(contexts[:4], start=1):
            snippet = _window_around_query(ch.text, query, width=1)
            parts.append(f"[{i}] {snippet}")
            used.append(ch)
        answer = "Comparison contexts:\n" + "\n".join(parts)
        return answer, used

    if mode == "compute":
        parts = []
        for i, ch in enumerate(contexts[:6], start=1):
            snippet = _window_around_query(ch.text, query, width=1)
            parts.append(f"[{i}] {snippet}")
            used.append(ch)
        answer = "Relevant data for computation:\n" + "\n".join(parts)
        return answer, used

    best = contexts[0]
    snippet = _window_around_query(best.text, query, width=2)
    used.append(best)
    return snippet, used
