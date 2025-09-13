# local-rag-notebook
Local, **LLM-optional** “NotebookLM-style” search and Q&A over your own documents.

This repo now includes **Phase 1 / 1.1 / 1.2** upgrades:
- Hybrid retrieval **+ reranker** (config-toggable).
- **Eval harness** with `gold.jsonl` (Recall@K, MRR@K, latency).
- **Answer length controls** (`--max-ans-chars`, headings on/off, de-hyphenation).
- **File outputs** (JSON/Markdown/TXT/HTML) via `--out` / `--save` / `--format`.
- Small fixes (heading path rendering, page-range filter for debugging).
- **Phase 2 warm-up and keep-alive support** for Ollama (avoids cold-start timeouts).
- **DX improvements (Sept 2025):** Makefile, `pyproject.toml`, `requirements-dev.txt`, `.env.example`, and a test scaffold for smoother development.

---

## ✨ Features (current)
- **Offline-first.** No external APIs required (LLM synthesis optional & off by default).
- **Hybrid retrieval**: BM25 (lexical) + dense embeddings → **RRF** fusion + **neighbor expansion**.
- **Optional reranker**: cross-encoder boosts precision on the top candidates.
- **Extractive answers**: concatenated, normalized snippets with **(File, Heading, Page)** citations.
- **PDF normalization**: de-hyphenation, bullet cleanup.
- **Table-aware (MVP)**: CSV/TSV rows rendered as key:value lines for strong lexical matches.
- **Target by file** with `--files` (substring match).
- **LLM warm-up**: on first `query` run, a lightweight warm-up call is made to pre-load the Ollama model.
- **Keep-alive flag**: models stay loaded on the Ollama server (`--keep-alive 30m`) so subsequent queries are fast.
- **DX tooling**: `make env`, `make lint`, `make fmt`, `make test`, `make run`.

---

## 📦 Requirements
- Python **3.11+** recommended (works on **Windows + Python 3.13** with minor notes).
- No GPU required.
- Disk: a few hundred MB for embeddings & indexes at small scale.

**Runtime deps (core)**: `sentence-transformers`, `rank-bm25`, `numpy`, `pypdf`, `pyyaml`, `tqdm`, `pydantic`, `requests`

**Dev deps (new)**: `ruff`, `black`, `isort`, `pytest`, `python-dotenv`

---

## 🚀 Install & Ingest
```powershell
# 1) Create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # (macOS/Linux: source .venv/bin/activate)

# 2) Install runtime + dev deps
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3) Put docs in ./data  (PDF / MD / TXT / CSV / TSV)
# 4) Build indexes
python cli.py ingest .\data
```

---

## 🔎 Query (CLI)
Now supports **new retrieval/rerank controls** (added recently) **plus warm-up/keep-alive**:
- `--recall-topk N` — control the number of passages recalled before rerank.
- `--rerank-topk N` — control the number of passages kept after rerank.
- `--min-rerank-score F` — set a cutoff score; passages below are dropped unless fallback triggers.
- `--no-rerank` — bypass reranker entirely, using recall only (debug/fallback).
- `--neighbor-window N` — include ±N neighboring chunks around each selected passage for context stitching.
- `--allow-remote` — override offline guard to allow non-local endpoints.
- `--keep-alive DURATION` — keep the Ollama model in memory (default `30m`; supports `m`/`h` values).

### Warm-up behavior
- On the first `query` run, `cli.py` automatically sends a small `"warmup"` prompt to Ollama.
- Combined with `--keep-alive`, this ensures large models (e.g., `llama3.1:8b`) don’t unload between queries.

### Examples
Example (bypass reranker, recall 12 passages, keep 6 contexts):
```powershell
python cli.py query "Provide an overview of the exam" `
  --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435 `
  --recall-topk 12 --no-rerank --k 6 --show-contexts --keep-alive 2h
```

Example (rerank with threshold and fallback):
```powershell
python cli.py query "Provide an overview of the exam" `
  --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435 `
  --recall-topk 12 --rerank-topk 12 --min-rerank-score 0.15 --k 6 --show-contexts
```

Other basics:
```powershell
# Basic
python cli.py query "List the prerequisites"

# Target a file
python cli.py query "Provide an overview of the exam" --files "AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf"

# Show final contexts used
python cli.py query "What are study domains" --show-contexts

# Increase answer size
python cli.py query "Provide an overview of the exam" --max-ans-chars 2200

# Debug by page window
python cli.py query "Provide an overview of the exam" --pages 16-20
```

### Save to file
```powershell
# Auto-named under outputs/, HTML
python cli.py query "What are study domains" --save outputs --format html --quiet

# Explicit filename (format inferred by extension)
python cli.py query "List the prerequisites" --out outputs/prereqs.md
```

---

## ⚙️ Configuration (`config.yaml`)
*(unchanged — see existing doc)*

---

## 🧪 Evaluation (Phase 1)
*(unchanged — see existing doc)*

---

## 📤 Outputs (Phase 1.2)
*(unchanged — see existing doc)*

---

## 📈 Performance notes
- First query on a model can take 60–90s due to **cold load**. Warm-up + `--keep-alive` prevents this.
- Default reranker `BAAI/bge-reranker-base` is accurate but **heavy on CPU** (10–20s).  
  Switch to `cross-encoder/ms-marco-MiniLM-L-6-v2` for lighter CPU use.
- Tune `top_k_lexical` / `top_k_dense` (30–40) for latency vs coverage.

---

## 🧯 Troubleshooting
*(unchanged — see existing doc)*

---

## 🧹 Maintenance scripts
*(unchanged — see existing doc)*

---

## 🛠️ Developer Experience (new)
We’ve added tooling for easier development:

- **Makefile**  
  - `make env` — create venv + install runtime + dev deps  
  - `make lint` — run ruff  
  - `make fmt` — run black + isort  
  - `make test` — run pytest  
  - `make run ARGS='query "hello" ...'` — pass args to CLI  
  - `make clean` — remove caches  

- **pyproject.toml** — config for ruff, black, isort, pytest, coverage.  
- **requirements-dev.txt** — pins dev tools.  
- **.env.example** — documents environment variables (`OLLAMA_HOST`, `OLLAMA_MODEL`, etc.).  
- **tests/** — includes a basic smoke test to validate imports.  

### Developer workflow
```powershell
make env
make lint
make fmt
make test
make run ARGS='query "hello" --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435'
```

---

## 🧭 Roadmap
- **Phase 2** – Grounded **LLM synthesis** (citation-faithful, with **abstain**).
- **Phase 2.1 (added)** – Warm-up and keep-alive support for Ollama models (done ✅).
- **Phase 3** – Table-aware extraction.
- **Phase 4** – Corpus management & incremental indexing.
- **Phase 5** – Perf pass: ANN (`hnswlib`), caching, concurrency.
- **Phase 6** – Minimal local UI.
- **Phase 7** – Packaging/DX improvements.

---

## 👤 Author
Erick Perales — Cloud Migration IT Architect, Cloud Migration Specialist  
GitHub: https://github.com/peralese
