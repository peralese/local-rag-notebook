#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from statistics import mean
import os
import re

import requests  # <-- added for warm-up HTTP call

from local_rag_notebook.app import ingest_path, load_config, query_text
from local_rag_notebook.logging_utils import setup_logging
from local_rag_notebook.utils.output import (  # infer_format kept for compatibility
    infer_format,
    write_output,
)

# Phase 2 (local-only synthesis)
from synthesizer import synthesize_answer

logger = logging.getLogger(__name__)

# NOTE: We no longer instantiate OllamaLLM here for warm-up; we do a direct HTTP warm-up instead.
# from llm.ollama import OllamaLLM


def _normalize_endpoint(ep: str | None) -> str:
    """CLI --endpoint > OLLAMA_HOST env > default; ensure scheme; strip trailing slash."""
    cand = (ep or os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
    if not re.match(r"^https?://", cand):
        cand = "http://" + cand
    return cand.rstrip("/")


def _ensure_localhost(endpoint: str):
    """Offline guard: refuse non-local endpoints when --offline is set."""
    if not (endpoint.startswith("http://localhost") or endpoint.startswith("http://127.0.0.1")):
        raise SystemExit(f"Offline mode: refusing non-local endpoint: {endpoint}")


def main():
    parser = argparse.ArgumentParser(
        prog="local-rag-notebook",
        description="Local, NotebookLM-style retrieval and (optional) LLM synthesis over your docs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global logging flags
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logs (to stderr)."
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce logs to WARN and above.")
    parser.add_argument("--log-json", action="store_true", help="Emit logs as JSON lines (stderr).")

    # -----------------------
    # ingest
    # -----------------------
    p_ing = sub.add_parser("ingest", help="Ingest a folder of documents")
    p_ing.add_argument("path", type=str, help="Path to folder with documents")
    p_ing.add_argument("--config", type=str, default="config.yaml")

    # -----------------------
    # query
    # -----------------------
    p_q = sub.add_parser("query", help="Query the indexed corpus")
    p_q.add_argument("question", type=str, help="Your question string")
    p_q.add_argument("--config", type=str, default="config.yaml")

    # Existing knobs
    p_q.add_argument(
        "--k", type=int, default=None, help="Override final top-k contexts (default 8)"
    )
    p_q.add_argument(
        "--files", type=str, default="", help="Comma-separated file name filters (substring match)"
    )
    p_q.add_argument("--pages", type=str, default="", help="Optional page range like 16-20")
    p_q.add_argument(
        "--max-ans-chars", type=int, default=None, help="Answer budget in characters (default 1500)"
    )
    p_q.add_argument(
        "--show-contexts", action="store_true", help="Print the final contexts used for answering"
    )
    p_q.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write result to a file (infers format from extension)",
    )
    p_q.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["json", "md", "txt", "html"],
        help="Output format (overrides --out extension)",
    )
    p_q.add_argument(
        "--save", type=str, default=None, help="Directory to auto-save result (default outputs/)"
    )
    p_q.add_argument(
        "--quiet", action="store_true", help="Suppress console output (use with --out/--save)"
    )

    # NEW: retrieval / rerank controls (threaded into query_text)
    p_q.add_argument(
        "--recall-topk", type=int, default=None, help="Initial recall size before rerank (e.g., 12)"
    )
    p_q.add_argument(
        "--rerank-topk", type=int, default=None, help="Top-k to keep after rerank (e.g., 12)"
    )
    p_q.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="Minimum rerank score cutoff (e.g., 0.15)",
    )
    p_q.add_argument(
        "--no-rerank",
        action="store_true",
        help="Bypass reranker and use recall top-k directly (debug/fallback)",
    )
    p_q.add_argument(
        "--neighbor-window",
        type=int,
        default=None,
        help="Also include ±N neighbor chunks per selected hit (section stitching)",
    )

    # Phase 2 (local-only synthesis) flags
    p_q.add_argument(
        "--synthesize",
        action="store_true",
        help="Enable grounded LLM synthesis (local only; answers strictly from retrieved context)",
    )
    p_q.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama"],
        help="Local LLM backend (default: ollama)",
    )
    p_q.add_argument(
        "--endpoint",
        default=None,  # ← let env var take precedence if flag is omitted
        help="Backend endpoint. Precedence: --endpoint > OLLAMA_HOST env > http://localhost:11434",
    )
    p_q.add_argument("--model", default="llama3.1:8b", help="Local model name (e.g., llama3.1:8b)")
    p_q.add_argument(
        "--keep-alive",
        default="30m",
        help="Keep the model loaded on the server (e.g., 30m, 2h) for faster subsequent calls",
    )

    # Offline guard + explicit override
    p_q.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Disallow non-local endpoints (default: on)",
    )
    p_q.add_argument(
        "--allow-remote",
        action="store_true",
        help="Override offline guard to allow non-local endpoints",
    )

    p_q.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.70,
        help="Min blended support to answer; else ABSTAIN (default: 0.70)",
    )
    p_q.add_argument(
        "--cite-n",
        type=int,
        default=3,
        help="Trim citations to at most N distinct sources (default: 3)",
    )
    p_q.add_argument(
        "--max-context-chars",
        type=int,
        default=24000,
        help="Max characters of context sent to the LLM (default: 24000)",
    )
    p_q.add_argument(
        "--strict-citations",
        action="store_true",
        help="Require each sentence to include at least one [C#] tag",
    )
    p_q.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip Ollama warmup even when using --synthesize",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    # ----- logging setup -----
    if args.verbose and args.quiet:
        print("Cannot use --verbose and --quiet together.", file=sys.stderr)
        sys.exit(2)
    if args.verbose:
        setup_logging(level="DEBUG", json_logs=args.log_json)
    elif args.quiet:
        setup_logging(level="WARNING", json_logs=args.log_json)
    else:
        setup_logging(level="INFO", json_logs=args.log_json)

    logger.debug("CLI args parsed: %s", vars(args))

    if args.cmd == "ingest":

        try:
            data_path = Path(args.path)
            logger.info("Starting ingest: %s", data_path.resolve())
            ingest_path(data_path, cfg)
            logger.info("Ingest completed.")
            sys.exit(0)  # ← ensure we don't fall through to query
        except Exception as e:
            logger.exception("Ingest failed: %s", e)
            sys.exit(2)

    elif args.cmd == "query":
        # Resolve endpoint from CLI/env/default and enforce offline guard if needed
        endpoint = _normalize_endpoint(args.endpoint)
        # Only warm up when synthesizing with ollama and not explicitly skipped
        if getattr(args, "synthesize", False) and args.backend == "ollama" and not getattr(args, "no_warmup", False):
            logger.info("Warming up Ollama model (once) ...")
            try:
                warmup_payload = {
                    "model": args.model,
                    "prompt": "ping",
                    "stream": False,
                    "keep_alive": args.keep_alive,
                }
                r = requests.post(f"{endpoint}/api/generate", json=warmup_payload, timeout=(10, 120))
                if r.status_code >= 400:
                    logger.warning("warmup skipped: %s returned HTTP %s", f"{endpoint}/api/generate", r.status_code)
                else:
                    logger.debug("warmup ok: %s model=%s", endpoint, args.model)
            except requests.exceptions.RequestException as e:
                logger.warning("warmup skipped: cannot reach Ollama at %s (%s)", endpoint, e.__class__.__name__)
                
        if args.offline and not args.allow_remote:
            _ensure_localhost(endpoint)

        # Parse filters and page range
        filters = [s.strip() for s in args.files.split(",")] if args.files else None
        page_range = None
        if args.pages:
            try:
                lo, hi = [int(x) for x in args.pages.split("-", 1)]
                page_range = (lo, hi)
            except Exception:
                print("Invalid --pages format; expected something like 16-20")

        # Build kwargs for query_text without breaking your current signature
        query_kwargs = dict(
            final_k=args.k,
            file_filters=filters,
            show_contexts=args.show_contexts,
            page_range=page_range,
            max_answer_chars=args.max_ans_chars,
        )

        # Thread the NEW knobs through if your query_text supports them
        if args.recall_topk is not None:
            query_kwargs["recall_topk"] = args.recall_topk
        if args.rerank_topk is not None:
            query_kwargs["rerank_topk"] = args.rerank_topk
        if args.min_rerank_score is not None:
            query_kwargs["min_rerank_score"] = args.min_rerank_score
        if args.no_rerank:
            query_kwargs["no_rerank"] = True
        if args.neighbor_window is not None:
            query_kwargs["neighbor_window"] = args.neighbor_window

        # Run retrieval-based pipeline
        ans = query_text(args.question, cfg, **query_kwargs)

        # If Phase-2 synthesis is requested, build a local-only synthesized answer
        if args.synthesize:
            contexts = ans.get("contexts", []) or []

            # Adapt retrieved chunks for synthesizer
            retrieved_chunks = []
            for ctx in contexts:
                retrieved_chunks.append(
                    {
                        "text": ctx.get("text", ""),
                        "title": ctx.get("file") or ctx.get("title") or "Source",
                        "path": ctx.get("path") or ctx.get("file") or "",
                        "source": ctx.get("file") or "",
                    }
                )

            # Average similarity if available; otherwise use a conservative default
            if contexts and any("score" in c for c in contexts):

                def _to_float_score(x, default=0.7):
                    try:
                        return default if x is None else float(x)
                    except (TypeError, ValueError):
                        return default

                sims = [_to_float_score(c.get("score"), 0.7) for c in contexts]
                avg_similarity = mean(sims) if sims else 0.7
            else:
                avg_similarity = 0.7  # sane default if scores not provided

            synth = synthesize_answer(
                query=args.question,
                retrieved=retrieved_chunks,
                avg_sim=avg_similarity,
                backend=args.backend,
                model=args.model,
                endpoint=endpoint,
                max_context_chars=args.max_context_chars,
                cite_n=args.cite_n,
                abstain_threshold=args.abstain_threshold,
                strict_citations=args.strict_citations,
            )

            # Render into the same structure the rest of the CLI expects
            if synth.get("abstain"):
                ans["answer"] = f"**ABSTAINED**: {synth.get('why', 'insufficient support')}"
                # Show top snippets as pseudo-citations so printing doesn't break
                snips = synth.get("snippets", []) or []
                ans["citations"] = [
                    {
                        "file": s.get("title", f"Source {s.get('id','?')}"),
                        "heading_path": "",
                        "page_no": "?",
                        "id": s.get("id", ""),
                    }
                    for s in snips
                ]
            else:
                ans["answer"] = synth.get("answer_markdown", "").strip()
                # Map synthesizer citations to existing citation print format
                cits = synth.get("citations", []) or []
                ans["citations"] = [
                    {
                        "file": c.get("title", f"Source {c.get('id','?')}"),
                        "heading_path": c.get("uri_or_path", ""),
                        "page_no": "?",
                        "id": c.get("id", ""),
                    }
                    for c in cits
                ]

        # Write outputs if requested
        if args.out or args.save:
            target = write_output(
                args.question, ans, out_path=args.out, fmt=args.format, save_dir=args.save
            )
            if not args.quiet:
                print(f"[saved] {target}")

        if not args.quiet:
            print("\n=== ANSWER ===")
            print((ans.get("answer") or "").strip())

            print("\n=== CITATIONS ===")
            for c in ans.get("citations", []):
                hp = c.get("heading_path") or ""
                pg = c.get("page_no") or "?"
                print(f"- {c.get('file','?')} | {hp} | Page {pg}")

            print("\n=== TRACE ===")
            top_ids = ans.get("trace", {}).get("top_context_ids")
            print(f"top_context_ids: {top_ids}")
            timers = ans.get("trace", {}).get("timers_ms")
            if timers:
                print(f"timers_ms: {timers}")

            if args.show_contexts:
                print("\n=== CONTEXTS ===")
                for i, ctx in enumerate(ans.get("contexts", []), start=1):
                    print(
                        f"[{i}] {ctx.get('file','?')} | Page {ctx.get('page','?')} | {ctx.get('id','?')}"
                    )
                    print(ctx.get("text", ""))
                    print("---")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
