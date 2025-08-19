from pathlib import Path
from typing import List, Optional
import yaml
from tqdm import tqdm

from .index.schema import Chunk, Hit
from .ingest.pdf import parse_pdf
from .ingest.md_txt import parse_md_or_txt
from .ingest.csv_tsv import parse_csv_tsv
from .ingest.normalize import chunk_sections
from .index.dense import DenseIndexer
from .index.lexical import LexicalIndexer
from .retrieve.fuse import rrf_merge, expand_neighbors
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
    for f in tqdm(list(data_dir.rglob("*")), desc="Scanning files"):
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

    lex = LexicalIndexer(index_dir)
    lex.build(chunks)

    dense = DenseIndexer(index_dir, cfg["models"]["embedding"])
    dense.build(chunks)

    print(f"Ingest complete. Chunks: {len(chunks)}")
    print(f"Index dir: {index_dir}")

def _file_match(path: str, filters: List[str]) -> bool:
    p = path.lower()
    for f in filters:
        f = f.strip().lower()
        if not f:
            continue
        # substring match; could be extended to glob
        if f in p:
            return True
    return False

def query_text(question: str, cfg: dict, final_k: int | None = None, file_filters: Optional[List[str]] = None) -> dict:
    index_dir = Path(cfg["app"]["index_dir"]).resolve()
    lex = LexicalIndexer(index_dir).load()
    dense = DenseIndexer(index_dir, cfg["models"]["embedding"]).load()

    bm25_hits = lex.search(question, top_k=cfg["retrieval"]["top_k_lexical"])
    dense_hits = dense.search(question, top_k=cfg["retrieval"]["top_k_dense"])

    fused = rrf_merge(bm25_hits, dense_hits, k=cfg["retrieval"]["rrf_k"])

    ctx_ids = expand_neighbors([h.chunk_id for h in fused], lex.chunk_id_order, radius=cfg["retrieval"]["neighborhood"])

    chunks = lex.load_chunks_by_ids(ctx_ids)

    # Optional file filter
    if file_filters:
        chunks = [ch for ch in chunks if _file_match(ch.meta.get("file_path",""), file_filters)]

    top_k = final_k or 8
    answer, used = extract_answer(question, chunks[:top_k])

    citations = [{
        "file": ch.meta.get("file_path"),
        "heading_path": ch.heading_path,
        "page_no": ch.page_no
    } for ch in used]

    trace = {
        "top_context_ids": [c.chunk_id for c in chunks[:top_k]],
        "bm25_count": len(bm25_hits),
        "dense_count": len(dense_hits)
    }
    return {"answer": answer, "citations": citations, "trace": trace}
