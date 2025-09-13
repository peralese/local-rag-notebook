from __future__ import annotations

from typing import Dict, List

from ..index.schema import Hit


def rrf_merge(bm25_hits: List[Hit], dense_hits: List[Hit], k: int = 60) -> List[Hit]:
    scores: Dict[str, float] = {}
    for rank, h in enumerate(bm25_hits, start=1):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, h in enumerate(dense_hits, start=1):
        scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
    merged = [Hit(chunk_id=cid, score=s) for cid, s in scores.items()]
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged

def expand_neighbors(ids: List[str], order_list: List[str], radius: int = 1) -> List[str]:
    index_map = {cid: idx for idx, cid in enumerate(order_list)}
    out = []
    seen = set()
    for cid in ids:
        base = index_map.get(cid)
        if base is None:
            continue
        for j in range(base - radius, base + radius + 1):
            if 0 <= j < len(order_list):
                cand = order_list[j]
                if cand not in seen:
                    out.append(cand)
                    seen.add(cand)
    return out
