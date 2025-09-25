# local-rag-notebook
Local, **LLM‑optional** “NotebookLM‑style” search & Q&A over your own documents. Offline‑first retrieval with optional local LLM synthesis via **Ollama**.

---

## ✨ What’s included (current)
- **Hybrid retrieval**: BM25 (lexical) + dense embeddings → **RRF fusion** + **neighbor expansion**.
- **Optional reranker** (cross‑encoder) to boost precision on the top candidates.
- **Extractive answers with citations**: stitched snippets with **(File, Heading, Page)** references.
- **PDF/MD/TXT/CSV/TSV ingestion** with text normalization (de‑hyphenation, bullets).
- **Outputs** to JSON / Markdown / TXT / HTML (`--out` / `--save` / `--format`).
- **Warm‑up + keep‑alive for Ollama** to avoid model cold starts.
- **DX tooling**: Makefile targets, dev deps, lint/format/test config, and a smoke test.

---

## 📦 Requirements
- Python **3.10+** (tested on Windows w/ **3.13** and WSL).
- No GPU required.
- A few hundred MB of disk for small corpora indexes.

**Core deps** (runtime): `sentence-transformers`, `rank-bm25`, `numpy`, `pypdf`, `pyyaml`, `tqdm`, `pydantic`, `requests`  
**Dev deps**: `ruff`, `black`, `isort`, `pytest`, `python-dotenv`

---

## 🚀 Quickstart
```powershell
# 1) Create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # (macOS/Linux: source .venv/bin/activate)

# 2) Install runtime + dev deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# 3) Put your docs under ./data  (PDF / MD / TXT / CSV / TSV)

# 4) Build the index
python cli.py ingest .\data
```

---

## 🔎 Query (CLI)
```powershell
# Retrieval only (fast, offline by default)
python cli.py query "List the prerequisites"

# Target specific files
python cli.py query "Provide an overview of the exam" --files "AWS-*.pdf"

# Show the stitched contexts used
python cli.py query "What are study domains" --show-contexts

# Save answer to file (HTML)
python cli.py query "What are study domains" --save outputs --format html --quiet
```

### Synthesis with a local LLM (Ollama)
```powershell
# Example using WSL-forwarded Ollama on 11435
python cli.py query "hello world" `
  --synthesize --backend ollama --model llama3.1:8b `
  --endpoint http://localhost:11435 `
  --keep-alive 2h --show-contexts
```

**New/updated flags**
- `--no-warmup` — skip the Ollama warmup even when `--synthesize` is used.
- `--endpoint` — endpoint precedence is now **CLI `--endpoint` > `OLLAMA_HOST` env > `http://localhost:11434`**.
- `-v/--verbose`, `-q/--quiet` — control log verbosity (logs go to **stderr**).
- `--log-json` — print one JSON log object per line to **stderr** (great for debugging).

**Retrieval/rerank controls** (if present in your build):
- `--recall-topk N`, `--rerank-topk N`, `--min-rerank-score F`, `--no-rerank`  
- `--neighbor-window N`, `--max-context-chars N`, `--cite-n N`, `--strict-citations`

**Offline guard**
- By default the tool is offline‑first. Non‑localhost endpoints are rejected unless you pass `--allow-remote`.

---

## 🔥 Warm‑up behavior (Ollama)
- On `query` with `--synthesize --backend ollama`, the CLI sends a tiny **“ping”** to **preload the model**.
- Endpoint resolution: **`--endpoint` > `OLLAMA_HOST` > `http://localhost:11434`**.
- If Ollama isn’t reachable you’ll see a friendly warning, e.g.  
  `warmup skipped: cannot reach Ollama at http://localhost:11435 (ConnectionError)` — the program continues.
- Skip warmup with `--no-warmup`.

**Examples**
```powershell
# Use env endpoint
$env:OLLAMA_HOST = "http://localhost:11435"
python cli.py query "hello world" --synthesize --backend ollama --model llama3.1:8b

# Force a specific endpoint (overrides env)
python cli.py query "hello world" --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435

# Skip model preload
python cli.py query "hello world" --synthesize --backend ollama --model llama3.1:8b --no-warmup
```

---

## 🪟 Windows + WSL notes
- Start Ollama **inside WSL**: `ollama serve` (listens on `http://localhost:11434` inside WSL).  
- If you run the CLI from **Windows**, forward a port to WSL (e.g., `11435 → 11434`) and then either:
  - set `OLLAMA_HOST=http://localhost:11435`, or
  - pass `--endpoint http://localhost:11435`.

---

## 💾 Saving outputs
```powershell
# Auto-named under outputs/, HTML
python cli.py query "What are study domains" --save outputs --format html --quiet

# Explicit filename (format inferred by extension)
python cli.py query "List the prerequisites" --out outputs/prereqs.md
```

---

## ⚙️ Configuration
- `config.yaml` contains retrieval/synthesis defaults used by the app.
- The CLI also reads **environment variables** (via `.env` if present).

### Environment variables (extend `.env.example`)
```env
# Ollama endpoint (used if --endpoint is not provided)
OLLAMA_HOST=http://localhost:11435

# Optional: tune HTTP client timeouts (seconds)
# Defaults: 10s connect, 600s read
OLLAMA_CONNECT_TIMEOUT=10
OLLAMA_READ_TIMEOUT=600

# Optional: cap generation length when max_tokens isn't set by the caller
# (helps keep latency reasonable on CPU models)
OLLAMA_NUM_PREDICT=512

# App logging defaults (can be overridden by CLI -v/-q/--log-json)
LOG_LEVEL=INFO
```
## 🧪 Testing (Phase 2 core)
```bash
make test
# or
pytest -q

---

## 🧪 Evaluation (Phase 1)
- Eval harness accepts a `gold.jsonl` and reports Recall@K, MRR@K, and latency.
- Use it to tune `recall-topk`, reranker cutoff, and neighbor window size.

---

## 📈 Performance notes
- First query on a model can take 30–90s due to **cold load**. Use warm‑up + `--keep-alive` to avoid this.
- Default reranker (e.g., `BAAI/bge-reranker-base`) is accurate but **CPU‑heavy**; try `cross-encoder/ms-marco-MiniLM-L-6-v2` for lighter runs.
- Tune `top_k_lexical` / `top_k_dense` (≈30–40) for latency vs coverage. Consider smaller models (e.g., `llama3.1:3b`) for faster synthesis.

---

## 🧯 Troubleshooting
**“warmup skipped: cannot reach Ollama …”**  
Ollama isn’t running or the endpoint is wrong. Start `ollama serve` in WSL and use `--endpoint` or `OLLAMA_HOST` to match your forwarded port.

**Read timeout calling /api/chat**  
On CPU and big models, long generations can exceed 120s. The client now defaults to a longer read timeout (600s), configurable via:
- `OLLAMA_READ_TIMEOUT` — increase if you still see timeouts.
- Consider a lighter model (`llama3.1:3b`) or cap generation via `OLLAMA_NUM_PREDICT` (e.g., `256`–`512`).

**Pydantic / other imports missing**  
Install deps in the *active* venv:  
`python -m pip install -r requirements.txt -r requirements-dev.txt`

**“make is not recognized” on Windows**  
Either install it (`choco install make`) or run the equivalent raw commands shown in this README.

---

## 🛠️ Developer experience
**Makefile targets**
```bash
make env     # create .venv, install runtime + dev deps
make lint    # ruff
make fmt     # isort + black
make test    # pytest -q
make run ARGS='query "hello" --synthesize --backend ollama --model llama3.1:8b --endpoint http://localhost:11435'
make clean   # remove caches
```

**Project config**
- `pyproject.toml` configures ruff, black, isort, pytest, coverage.
- `requirements-dev.txt` pins dev tools.
- `.env.example` documents `OLLAMA_HOST`, timeouts, and logging defaults.
- `tests/` includes a smoke test to validate imports.

---

## 🧭 Roadmap (high‑level)
- **Phase 2** – Grounded LLM synthesis (citation‑faithful) + **abstain** on low support.
- **Phase 3** – Better table extraction.
- **Phase 4** – Corpus mgmt & incremental indexing.
- **Phase 5** – Perf: ANN (`hnswlib`), caching, concurrency.
- **Phase 6** – Minimal local UI.
- **Phase 7** – Packaging/DX improvements.

---

## 👤 Author
Erick Perales — Cloud Migration IT Architect, Cloud Migration Specialist  
GitHub: https://github.com/peralese
