import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from local_rag_notebook.app import load_config, query_text
from local_rag_notebook.index.dense import DenseIndexer
from local_rag_notebook.index.lexical import LexicalIndexer
from local_rag_notebook.retrieve.fuse import rrf_merge


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cases.append(json.loads(ln))
    return cases


def _match_citation(ch_meta: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    f = ch_meta.get("file_path") or ch_meta.get("file") or ch_meta.get("filePath") or ""
    p = (
        ch_meta.get("page_no")
        if "page_no" in ch_meta
        else ch_meta.get("page") if "page" in ch_meta else None
    )
    ok_file = (spec.get("file_contains") or "").lower() in f.lower()
    if not ok_file:
        return False
    if p is None:
        return True if (spec.get("page_min") is None and spec.get("page_max") is None) else False
    lo = spec.get("page_min", -(10**9))
    hi = spec.get("page_max", 10**9)
    return lo <= int(p) <= hi


def _compute_retrieval_metrics(
    lex: LexicalIndexer, dense: DenseIndexer, q: str, qk: int, expected: List[Dict[str, Any]]
) -> Dict[str, Any]:
    bm25_hits = lex.search(q, top_k=qk)
    dense_hits = dense.search(q, top_k=qk)
    fused = rrf_merge(bm25_hits, dense_hits, k=60)
    fused_ids = [h.chunk_id for h in fused][:qk]
    chunks = lex.load_chunks_by_ids(fused_ids)
    rank = None
    for i, ch in enumerate(chunks, start=1):
        meta = {"file_path": ch.meta.get("file_path"), "page_no": ch.page_no}
        if any(_match_citation(meta, spec) for spec in expected):
            rank = i
            break
    return {"hit": rank is not None, "mrr": 0.0 if rank is None else 1.0 / rank}


def _string_checks(
    answer: str, must_inc: List[str], any_of: List[str], must_not: List[str]
) -> Dict[str, Any]:
    a = answer.lower()
    ok = True
    reasons = []
    for s in must_inc or []:
        if s.lower() not in a:
            ok = False
            reasons.append(f"missing:{s}")
    if any_of:
        if not any(s.lower() in a for s in any_of):
            ok = False
            reasons.append("any_of_failed")
    for s in must_not or []:
        if s.lower() in a:
            ok = False
            reasons.append(f"must_not:{s}")
    return {"ok": ok, "reasons": reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold.jsonl")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--qk", type=int, default=10, help="Top-K for retrieval metrics")
    ap.add_argument("--tags", type=str, default="", help="Comma-separated tag filter (optional)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    index_dir = Path(cfg["app"]["index_dir"]).resolve()
    lex = LexicalIndexer(index_dir).load()
    dense = DenseIndexer(index_dir, cfg["models"]["embedding"]).load()

    gold = _load_gold(Path(args.gold))
    if args.tags:
        tagset = set(t.strip().lower() for t in args.tags.split(",") if t.strip())
        gold = [g for g in gold if set((g.get("tags") or [])).intersection(tagset)]

    results = []
    latencies = []
    hit_flags = []
    mrr_vals = []
    ans_ok_flags = []

    for g in gold:
        q = g["question"]
        file_filters = g.get("file_filters")
        must_inc = g.get("must_include", [])
        any_of = g.get("any_of", [])
        must_not = g.get("must_not_include", [])
        expected = g.get("expected_citations", [])

        t0 = time.perf_counter()
        ans = query_text(q, cfg, final_k=None, file_filters=file_filters, show_contexts=False)
        dt = int((time.perf_counter() - t0) * 1000)
        latencies.append(dt)

        ret = _compute_retrieval_metrics(lex, dense, q, args.qk, expected)
        hit_flags.append(1 if ret["hit"] else 0)
        mrr_vals.append(ret["mrr"])

        checks = _string_checks(ans["answer"], must_inc, any_of, must_not)
        ans_ok_flags.append(1 if checks["ok"] else 0)

        results.append(
            {
                "qid": g.get("qid"),
                "question": q,
                "retrieval_hit": ret["hit"],
                "mrr": ret["mrr"],
                "answer_ok": checks["ok"],
                "answer_reasons": checks["reasons"],
                "latency_ms": dt,
            }
        )

    recall_at_k = sum(hit_flags) / max(1, len(hit_flags))
    mrr_at_k = sum(mrr_vals) / max(1, len(mrr_vals))
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0

    print("=== EVAL SUMMARY ===")
    print(f"Cases: {len(gold)}")
    print(f"Recall@{args.qk}: {recall_at_k:.3f}")
    print(f"MRR@{args.qk}:    {mrr_at_k:.3f}")
    print(f"Latency p50:      {p50} ms")
    print(f"Latency p95:      {p95} ms")
    print("\nFailed cases:")
    for r in results:
        if not (r["retrieval_hit"] and r["answer_ok"]):
            print(
                f"- {r['qid']}: hit={r['retrieval_hit']} answer_ok={r['answer_ok']} reasons={r['answer_reasons']} (lat {r['latency_ms']} ms)"
            )


if __name__ == "__main__":
    main()
