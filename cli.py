
import argparse
from pathlib import Path
from local_rag_notebook.app import ingest_path, query_text, load_config
from local_rag_notebook.utils.output import write_output, infer_format

def main():
    parser = argparse.ArgumentParser(prog="local-rag-notebook")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest a folder of documents")
    p_ing.add_argument("path", type=str, help="Path to folder with documents")
    p_ing.add_argument("--config", type=str, default="config.yaml")

    p_q = sub.add_parser("query", help="Query the indexed corpus")
    p_q.add_argument("question", type=str, help="Your question string")
    p_q.add_argument("--config", type=str, default="config.yaml")
    p_q.add_argument("--k", type=int, default=None, help="Override final top-k contexts (default 8)")
    p_q.add_argument("--files", type=str, default="", help="Comma-separated file name filters (substring match)")
    p_q.add_argument("--pages", type=str, default="", help="Optional page range like 16-20")
    p_q.add_argument("--max-ans-chars", type=int, default=None, help="Answer budget in characters (default 1500)")
    p_q.add_argument("--show-contexts", action="store_true", help="Print the final contexts used for answering")
    p_q.add_argument("--out", type=str, default=None, help="Write result to a file (infers format from extension)")
    p_q.add_argument("--format", type=str, default=None, choices=["json","md","txt","html"], help="Output format (overrides --out extension)")
    p_q.add_argument("--save", type=str, default=None, help="Directory to auto-save result (default outputs/)")
    p_q.add_argument("--quiet", action="store_true", help="Suppress console output (use with --out/--save)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "ingest":
        ingest_path(Path(args.path), cfg)
    elif args.cmd == "query":
        filters = [s.strip() for s in args.files.split(",")] if args.files else None
        page_range = None
        if args.pages:
            try:
                lo, hi = [int(x) for x in args.pages.split("-", 1)]
                page_range = (lo, hi)
            except Exception:
                print("Invalid --pages format; expected something like 16-20")
        ans = query_text(
            args.question,
            cfg,
            final_k=args.k,
            file_filters=filters,
            show_contexts=args.show_contexts,
            page_range=page_range,
            max_answer_chars=args.max_ans_chars,
        )

        # Write outputs if requested
        if args.out or args.save:
            target = write_output(args.question, ans, out_path=args.out, fmt=args.format, save_dir=args.save)
            if not args.quiet:
                print(f"[saved] {target}")

        if not args.quiet:
            print("\n=== ANSWER ===")
            print(ans["answer"].strip())
            print("\n=== CITATIONS ===")
            for c in ans["citations"]:
                hp = c.get("heading_path") or ""
                pg = c.get("page_no") or "?"
                print(f"- {c['file']} | {hp} | Page {pg}")
            print("\n=== TRACE ===")
            top_ids = ans.get("trace", {}).get("top_context_ids")
            print(f"top_context_ids: {top_ids}")
            timers = ans.get("trace", {}).get("timers_ms")
            if timers:
                print(f"timers_ms: {timers}")
            if args.show_contexts:
                print("\n=== CONTEXTS ===")
                for i, ctx in enumerate(ans.get("contexts", []), start=1):
                    print(f"[{i}] {ctx['file']} | Page {ctx['page']} | {ctx['id']}")
                    print(ctx["text"])
                    print("---")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
