from pathlib import Path
from typing import List, Optional, Tuple
from copy import deepcopy
import time
import yaml

from .index.schema import Chunk, Hit
from .ingest.pdf import parse_pdf
from .ingest.md_txt import parse_md_or_txt
from .ingest.csv_tsv import parse_csv_tsv
from .ingest.normalize import chunk_sections
from .index.dense import DenseIndexer
from .index.lexical import LexicalIndexer
from .retrieve.fuse import rrf_merge, expand_neighbors
from .retrieve.rerank import Reranker
from .answer.extract import extract_answer
from .utils.log import Logger

SUPPORTED = {".pdf", ".md", ".txt", ".csv", ".tsv"}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest_path(data_dir: Path, cfg: dict):
    data_dir = data_dir.resolve()
    index_dir = Path(cfg["app"]["index_dir"]).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(Path("logs") / "ingest.log.jsonl")

    sections = []
    for f in list(data_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED:
            continue
        try:
            if f.suffix.lower() == ".pdf":
                secs = parse_pdf(f)
            elif f.suffix.lower() in (".md", ".txt"):
                secs = parse_md_or_txt(f)
            elif f.suffix.lower() in (".csv", ".tsv"):
                secs = parse_csv_tsv(f)
            else:
                continue
            sections.extend(secs)
        except Exception as e:
            log.write({"event": "parse_error", "file": str(f), "error": str(e)})

    chunks: list[Chunk] = []
    for s in sections:
        chunks.extend(
            chunk_sections(
                s, cfg["ingest"]["chunk_tokens"], cfg["ingest"]["overlap_tokens"]
            )
        )

    meta_path = index_dir / "chunks.jsonl"
    with open(meta_path, "w", encoding="utf-8") as out:
        for ch in chunks:
            out.write(ch.model_dump_json() + "\n")

    LexicalIndexer(index_dir).build(chunks)
    DenseIndexer(index_dir, cfg["models"]["embedding"]).build(chunks)

    print(f"Ingest complete. Chunks: {len(chunks)}")
    print(f"Index dir: {index_dir}")


def _file_match(path: str, filters: List[str]) -> bool:
    p = (path or "").lower()
    for f in filters:
        f = f.strip().lower()
        if not f:
            continue
        if f in p:
            return True
    return False


def query_text(
    question: str,
    cfg: dict,
    final_k: int | None = None,
    file_filters: Optional[List[str]] = None,
    show_contexts: bool = False,
    page_range: Optional[Tuple[int, int]] = None,
    max_answer_chars: Optional[int] = None,
    # ---- New optional knobs (threaded from CLI) ----
    recall_topk: Optional[int] = None,
    rerank_topk: Optional[int] = None,
    min_rerank_score: Optional[float] = None,
    no_rerank: bool = False,
    neighbor_window: Optional[int] = None,
) -> dict:
    """
    Retrieval (BM25 + dense + RRF) -> neighbor expansion -> optional rerank -> extractive answer.
    Adds safe fallbacks so reranker can't zero-out contexts.
    """
    # Work on a copy so we never mutate the loaded config
    cfg_run = deepcopy(cfg)

    timers = {}
    t0 = time.perf_counter()
    index_dir = Path(cfg_run["app"]["index_dir"]).resolve()
    lex = LexicalIndexer(index_dir).load()
    dense = DenseIndexer(index_dir, cfg_run["models"]["embedding"]).load()
    timers["load_ms"] = int((time.perf_counter() - t0) * 1000)

    # ---- Apply per-call overrides (recall / neighbor / rerank) ----
    # Recall overrides: if provided, apply to BOTH lexical and dense budgets.
    top_k_lex = int(cfg_run["retrieval"].get("top_k_lexical", 40))
    top_k_den = int(cfg_run["retrieval"].get("top_k_dense", 40))
    if recall_topk is not None:
        top_k_lex = int(recall_topk)
        top_k_den = int(recall_topk)

    # Neighbor radius override
    neigh_radius = int(cfg_run["retrieval"].get("neighborhood", 1))
    if neighbor_window is not None:
        neigh_radius = int(neighbor_window)

    # Reranker defaults
    rer_cfg = (cfg_run.get("retrieval", {}).get("reranker", {}) or {})
    use_reranker_cfg = bool(rer_cfg.get("enabled", False))
    rr_enabled = use_reranker_cfg and not no_rerank

    rr_topk_to_rerank = int(rer_cfg.get("top_k_to_rerank", 50))
    rr_final_k = int(rer_cfg.get("final_k", 8))
    if rerank_topk is not None:
        rr_final_k = int(rerank_topk)
        # If top_k_to_rerank isn't large enough, bump it to at least rr_final_k
        if rr_topk_to_rerank < rr_final_k:
            rr_topk_to_rerank = rr_final_k

    rr_min_score = float(min_rerank_score) if min_rerank_score is not None else float(rer_cfg.get("min_score", 0.0))

    # If '--k' was provided, prefer that as final_k after rerank
    if final_k is not None:
        rr_final_k = int(final_k)

    # ---- Recall
    t1 = time.perf_counter()
    bm25_hits = lex.search(question, top_k=top_k_lex)
    dense_hits = dense.search(question, top_k=top_k_den)
    timers["recall_ms"] = int((time.perf_counter() - t1) * 1000)

    # ---- RRF fuse
    t2 = time.perf_counter()
    fused = rrf_merge(bm25_hits, dense_hits, k=int(cfg_run["retrieval"].get("rrf_k", 60)))
    fused_ids = [h.chunk_id for h in fused]
    timers["fuse_ms"] = int((time.perf_counter() - t2) * 1000)

    # ---- Neighbor expansion (overrides allowed)
    t3 = time.perf_counter()
    ctx_ids = expand_neighbors(fused_ids, lex.chunk_id_order, radius=neigh_radius)
    timers["neighbor_ms"] = int((time.perf_counter() - t3) * 1000)

    # Load expanded chunks
    base_chunks: List[Chunk] = lex.load_chunks_by_ids(ctx_ids)

    # Optional filters
    if file_filters:
        base_chunks = [
            ch for ch in base_chunks if _file_match(ch.meta.get("file_path", ""), file_filters)
        ]
    if page_range:
        lo, hi = page_range
        base_chunks = [
            ch for ch in base_chunks if (ch.page_no is not None and lo <= int(ch.page_no) <= hi)
        ]

    # ---- Optional rerank with safe fallback ----
    reranked_ids: List[str] = []
    chunks: List[Chunk] = base_chunks  # default path = no rerank

    t4 = time.perf_counter()
    if rr_enabled:
        rr_model = rer_cfg.get("model", "BAAI/bge-reranker-base")
        rr = Reranker(rr_model)
        if rr.enabled and base_chunks:
            # candidates to rerank
            candidates = base_chunks[: rr_topk_to_rerank]
            # Expect: (ranked_chunks, scores_map_or_list)
            rer_chunks, scores_obj = rr.rerank(question, candidates, top_k=rr_topk_to_rerank)

            # Apply score cutoff if we can read scores
            if rer_chunks:
                filtered: List[Chunk] = []
                if scores_obj:
                    # scores may be a dict keyed by chunk_id, or a list aligned with rer_chunks
                    if isinstance(scores_obj, dict):
                        for ch in rer_chunks:
                            sc = float(scores_obj.get(ch.chunk_id, 0.0))
                            if sc >= rr_min_score:
                                filtered.append(ch)
                    elif isinstance(scores_obj, list):
                        for ch, sc in zip(rer_chunks, scores_obj):
                            if float(sc) >= rr_min_score:
                                filtered.append(ch)
                    else:
                        # unknown structure; skip thresholding
                        filtered = rer_chunks
                else:
                    filtered = rer_chunks

                chunks = filtered[: rr_final_k] if filtered else rer_chunks[: rr_final_k]
                reranked_ids = [c.chunk_id for c in rer_chunks[: rr_topk_to_rerank]]

            # Safe fallback: if nothing survived, fall back to pre-rerank
            if not chunks:
                chunks = base_chunks[: rr_final_k]
        else:
            chunks = base_chunks[: rr_final_k]
        timers["rerank_ms"] = int((time.perf_counter() - t4) * 1000)
    else:
        # No rerank path
        timers["rerank_ms"] = 0
        chunks = base_chunks[: rr_final_k]

    # ---- Answer formatting / extraction
    ans_cfg = cfg_run.get("answer", {}) or {}
    max_chars = int(max_answer_chars or ans_cfg.get("max_chars", 1500))
    join_with = str(ans_cfg.get("join_with", "\n\n"))
    include_headings = bool(ans_cfg.get("include_headings", True))
    dehyphenate = bool(ans_cfg.get("dehyphenate", True))

    t5 = time.perf_counter()
    answer, used = extract_answer(
        question,
        chunks,
        max_chars=max_chars,
        join_with=join_with,
        include_headings=include_headings,
        dehyphenate=dehyphenate,
    )
    timers["answer_ms"] = int((time.perf_counter() - t5) * 1000)
    timers["total_ms"] = int((time.perf_counter() - t0) * 1000)

    citations = [
        {
            "file": ch.meta.get("file_path"),
            "heading_path": ch.heading_path,
            "page_no": ch.page_no,
        }
        for ch in used
    ]

    trace = {
        "bm25_ids": [h.chunk_id for h in bm25_hits],
        "dense_ids": [h.chunk_id for h in dense_hits],
        "fused_ids": fused_ids,
        "neighbor_ids": ctx_ids,
        "reranked_ids": reranked_ids,
        "top_context_ids": [c.chunk_id for c in chunks],
        "timers_ms": timers,
        "reranker_enabled": rr_enabled,
        "recall_topk_effective": {"lexical": top_k_lex, "dense": top_k_den},
        "neighbor_radius_effective": neigh_radius,
        "rerank_controls": {
            "top_k_to_rerank": rr_topk_to_rerank,
            "final_k": rr_final_k,
            "min_score": rr_min_score,
        },
    }

    Logger(Path("logs") / "queries.log.jsonl").write(
        {
            "question": question,
            "file_filters": file_filters,
            "trace": trace,
            "citations": citations,
        }
    )

    top_k = rr_final_k if isinstance(rr_final_k, int) and rr_final_k > 0 else 8
    
    resp = {
        "answer": answer,
        "citations": citations,
        "trace": trace,
        # ALWAYS include contexts so the synthesizer has material
        "contexts": [
            {
                "id": c.chunk_id,
                "file": c.meta.get("file_path"),
                "page": c.page_no,
                "text": c.text[:1000],
                # Make score numeric if available; otherwise omit or default
                "score": (float(getattr(c, "score")) if getattr(c, "score", None) is not None else 0.7),
            }
            for c in chunks[:top_k]
        ],
    }
    return resp

