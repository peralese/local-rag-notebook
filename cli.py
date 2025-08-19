import argparse
from pathlib import Path
from local_rag_notebook.app import ingest_path, query_text, load_config

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

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "ingest":
        ingest_path(Path(args.path), cfg)
    elif args.cmd == "query":
        filters = [s.strip() for s in args.files.split(",")] if args.files else None
        ans = query_text(args.question, cfg, final_k=args.k, file_filters=filters)
        print("\n=== ANSWER ===")
        print(ans["answer"].strip())
        print("\n=== CITATIONS ===")
        for c in ans["citations"]:
            hp = c.get("heading_path") or ""
            pg = c.get("page_no") or "?"
            print(f"- {c['file']} | {hp} | Page {pg}")
        print("\n=== TRACE ===")
        print(f"top_context_ids: {ans['trace'].get('top_context_ids')}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
