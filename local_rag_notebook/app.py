
from pathlib import Path
from typing import List, Optional, Tuple
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
        chunks.extend(chunk_sections(s, cfg["ingest"]["chunk_tokens"], cfg["ingest"]["overlap_tokens"]))

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
    page_range: Optional[Tuple[int,int]] = None,
    max_answer_chars: Optional[int] = None,
) -> dict:
    timers = {}
    t0 = time.perf_counter()
    index_dir = Path(cfg["app"]["index_dir"]).resolve()
    lex = LexicalIndexer(index_dir).load()
    dense = DenseIndexer(index_dir, cfg["models"]["embedding"]).load()
    timers["load_ms"] = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    bm25_hits = lex.search(question, top_k=cfg["retrieval"]["top_k_lexical"])
    dense_hits = dense.search(question, top_k=cfg["retrieval"]["top_k_dense"])
    timers["recall_ms"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    fused = rrf_merge(bm25_hits, dense_hits, k=cfg["retrieval"]["rrf_k"])
    fused_ids = [h.chunk_id for h in fused]
    timers["fuse_ms"] = int((time.perf_counter() - t2) * 1000)

    t3 = time.perf_counter()
    ctx_ids = expand_neighbors(fused_ids, lex.chunk_id_order, radius=cfg["retrieval"]["neighborhood"])
    timers["neighbor_ms"] = int((time.perf_counter() - t3) * 1000)

    chunks = lex.load_chunks_by_ids(ctx_ids)

    if file_filters:
        chunks = [ch for ch in chunks if _file_match(ch.meta.get("file_path",""), file_filters)]

    if page_range:
        lo, hi = page_range
        chunks = [ch for ch in chunks if (ch.page_no is not None and lo <= int(ch.page_no) <= hi)]

    rer_cfg = (cfg.get("retrieval", {}).get("reranker", {}) or {})
    use_reranker = bool(rer_cfg.get("enabled", False))
    reranked_ids: List[str] = []
    if use_reranker:
        t4 = time.perf_counter()
        top_k_to_rerank = int(rer_cfg.get("top_k_to_rerank", 50))
        rr_model = rer_cfg.get("model", "BAAI/bge-reranker-base")
        rr = Reranker(rr_model)
        if rr.enabled:
            rer_chunks, _ = rr.rerank(question, chunks, top_k=top_k_to_rerank)
            chunks = rer_chunks
            reranked_ids = [c.chunk_id for c in chunks[:top_k_to_rerank]]
        timers["rerank_ms"] = int((time.perf_counter() - t4) * 1000)
    else:
        timers["rerank_ms"] = 0

    top_k = final_k or int(rer_cfg.get("final_k", 8)) if use_reranker else (final_k or 8)

    ans_cfg = cfg.get("answer", {}) or {}
    max_chars = int(max_answer_chars or ans_cfg.get("max_chars", 1500))
    join_with = str(ans_cfg.get("join_with", "\n\n"))
    include_headings = bool(ans_cfg.get("include_headings", True))
    dehyphenate = bool(ans_cfg.get("dehyphenate", True))

    t5 = time.perf_counter()
    answer, used = extract_answer(
        question,
        chunks[:top_k],
        max_chars=max_chars,
        join_with=join_with,
        include_headings=include_headings,
        dehyphenate=dehyphenate,
    )
    timers["answer_ms"] = int((time.perf_counter() - t5) * 1000)
    timers["total_ms"] = int((time.perf_counter() - t0) * 1000)

    citations = [{
        "file": ch.meta.get("file_path"),
        "heading_path": ch.heading_path,
        "page_no": ch.page_no
    } for ch in used]

    trace = {
        "bm25_ids": [h.chunk_id for h in bm25_hits],
        "dense_ids": [h.chunk_id for h in dense_hits],
        "fused_ids": fused_ids,
        "neighbor_ids": ctx_ids,
        "reranked_ids": reranked_ids,
        "top_context_ids": [c.chunk_id for c in chunks[:top_k]],
        "timers_ms": timers,
        "reranker_enabled": use_reranker
    }

    Logger(Path("logs") / "queries.log.jsonl").write({
        "question": question,
        "file_filters": file_filters,
        "trace": trace,
        "citations": citations
    })

    resp = {"answer": answer, "citations": citations, "trace": trace}
    if show_contexts:
        resp["contexts"] = [{"id": c.chunk_id, "file": c.meta.get("file_path"), "page": c.page_no, "text": c.text[:1000]} for c in chunks[:top_k]]
    return resp
