# local-rag-notebook
Local, **LLM-optional** “NotebookLM‑style” search and Q&A over your own documents.

This repo now includes **Phase 1 / 1.1 / 1.2** upgrades:
- Hybrid retrieval **+ reranker** (config‑toggable).
- **Eval harness** with `gold.jsonl` (Recall@K, MRR@K, latency).
- **Answer length controls** (`--max-ans-chars`, headings on/off, de‑hyphenation).
- **File outputs** (JSON/Markdown/TXT/HTML) via `--out` / `--save` / `--format`.
- Small fixes (heading path rendering, page-range filter for debugging).

---

## ✨ Features (current)
- **Offline‑first.** No external APIs required (LLM synthesis optional & off by default).
- **Hybrid retrieval**: BM25 (lexical) + dense embeddings → **RRF** fusion + **neighbor expansion**.
- **Optional reranker**: cross‑encoder boosts precision on the top candidates.
- **Extractive answers**: concatenated, normalized snippets with **(File, Heading, Page)** citations.
- **PDF normalization**: de‑hyphenation, bullet cleanup.
- **Table‑aware (MVP)**: CSV/TSV rows rendered as key:value lines for strong lexical matches.
- **Target by file** with `--files` (substring match).

---

## 📦 Requirements
- Python **3.11+** recommended (Windows/macOS/Linux). Works on **Windows + Python 3.13** with minor notes (below).
- No GPU required.
- Disk: a few hundred MB for embeddings & indexes at small scale.

**Runtime deps (core)**: `sentence-transformers`, `rank-bm25`, `numpy`, `pypdf`, `pyyaml`, `tqdm`, `pydantic`

> If `sentence-transformers` can’t find a Torch wheel (esp. Python 3.13), install a CPU wheel first or use Python 3.11 (see Troubleshooting).

---

## 🚀 Install & Ingest
```powershell
# 1) Create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # (macOS/Linux: source .venv/bin/activate)

# 2) Install
pip install --upgrade pip
pip install -r requirements.txt

# 3) Put docs in ./data  (PDF / MD / TXT / CSV / TSV)
# 4) Build indexes
python cli.py ingest .\data
```

---

## 🔎 Query (CLI)
```powershell
# Basic
python cli.py query "List the prerequisites"

# Target a file
python cli.py query "Provide an overview of the exam" --files "AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf"

# Show final contexts used
python cli.py query "What are study domains" --show-contexts

# Increase answer size
python cli.py query "Provide an overview of the exam" --max-ans-chars 2200

# Debug by page window (does not constrain eval; only filters contexts at runtime)
python cli.py query "Provide an overview of the exam" --pages 16-20
```

### Save to file
```powershell
# Auto-named under outputs/, HTML
python cli.py query "What are study domains" --save outputs --format html --quiet

# Explicit filename (format inferred by extension)
python cli.py query "List the prerequisites" --out outputs/prereqs.md
```

**Query flags (cheat sheet)**
- `--k N` — override contexts fed to the answerer (default usually 8).
- `--files "foo,bar"` — restrict to files whose path contains those substrings.
- `--pages "start-end"` — filter contexts to an inclusive page range (debug aid).
- `--max-ans-chars N` — answer budget (default 1500; also configurable).
- `--show-contexts` — print the exact chunks used.
- `--out PATH` / `--save DIR` / `--format json|md|txt|html` / `--quiet` — output control.
- `--config PATH` — use a non-default config (defaults to `config.yaml`).

---

## ⚙️ Configuration (`config.yaml`)
```yaml
app:
  data_dir: ./data
  index_dir: ./index_store

models:
  embedding: sentence-transformers/all-MiniLM-L6-v2  # 384-dim, CPU friendly

ingest:
  chunk_tokens: 700
  overlap_tokens: 90

retrieval:
  top_k_lexical: 40
  top_k_dense: 40
  rrf_k: 60
  neighborhood: 1
  reranker:
    enabled: true
    model: BAAI/bge-reranker-base           # precise but heavier on CPU
    top_k_to_rerank: 50
    final_k: 8

# Answer formatting controls
answer:
  max_chars: 1800
  join_with: "\n\n"
  include_headings: true
  dehyphenate: true

# (Reserved) Synthesis is OFF by default; added in Phase 2
synthesis:
  enabled: false
  provider: ollama
  model: llama3.2:3b-instruct
```

**Quick tuning knobs**
- **Precision**: enable reranker; increase `top_k_to_rerank` (30–100).  
- **Latency**: smaller reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) or `enabled: false`.
- **Continuity**: raise `neighborhood` to pull adjacent chunks.
- **Answer length**: increase `answer.max_chars` or pass `--max-ans-chars` per query.

---

## 🧪 Evaluation (Phase 1)
Run:
```powershell
python eval.py --gold .\eval\gold.jsonl --qk 10
# Filter by tags
python eval.py --gold .\eval\gold.jsonl --qk 10 --tags aws
```
- **Recall@K**: Whether any of the **top‑K fused** retrievals match your ground truth.
- **MRR@K**: 1/rank of the first match (within K).  
- **Latency**: p50/p95 in ms for end‑to‑end query.

### Gold file format (`eval/gold.jsonl`)
Each line is a JSON object like:
```json
{
  "qid": "overview",
  "question": "Provide an overview of the exam",
  "must_include": [],
  "any_of": [],
  "must_not_include": [],
  "expected_citations": [
    { "file_contains": "AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf",
      "page_min": 16, "page_max": 20 }
  ],
  "file_filters": ["AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf"],
  "tags": ["aws","fact"]
}
```
**Semantics**
- `expected_citations`: **Eval-only** ground truth; does *not* restrict search.  
  - Omit `page_min/max` to accept **any page** in that file.
- `must_include` / `any_of` / `must_not_include`: simple string checks on final answer.  
- `file_filters`: runtime restriction (same as `--files`) to reduce ambiguity.
- `--qk 10`: K used for retrieval metrics (not answer generation).

---

## 📤 Outputs (Phase 1.2)
- Write results to **JSON / MD / TXT / HTML** via `--out` or `--save`.
- Auto‑naming: `outputs/YYYYMMDD_HHMMSS_question-slug.ext`.

Example Markdown output includes question, answer, citations, and timers.

---

## 📈 Performance notes
- The default reranker `BAAI/bge-reranker-base` is accurate but **heavy on CPU** (10–20s).  
  Switch to:
  ```yaml
  retrieval:
    reranker:
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k_to_rerank: 30
  ```
- Keep `top_k_lexical` / `top_k_dense` reasonable (30–40).  
- If latency still high, disable reranker for bulk eval and re-enable for interactive use.

---

## 🧯 Troubleshooting
**Windows: Hugging Face symlink warning**
- Harmless but uses more disk. Fix by enabling **Developer Mode** or run Python as Admin.  
  To silence: `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`

**PyTorch wheel on Python 3.13**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

**“Native build required” errors**
- This build avoids PyMuPDF/pandas; we use **PyPDF** and the stdlib CSV reader.

**“No matching content found.”**
- Re‑ingest; check `index_store/chunks.jsonl` exists.  
- Use `--files` to scope to the right document.  
- Raise `retrieval.top_k_*` and `--k` a bit.

---

## 🧹 Maintenance scripts
- **Clean indexes/logs/caches**: `clear-parser-data.ps1`  
  ```powershell
  .\clear-parser-data.ps1           # defaults: index_store + logs + caches
  .\clear-parser-data.ps1 -All -Force   # including ./data (careful)
  ```

---

## 🧭 Roadmap
- **Phase 2** – Grounded **LLM synthesis** (off by default), citation‑faithful, with **abstain** when unsure.
- **Phase 3** – Table‑aware extraction (preserve tables, optional Camelot/pdfplumber).
- **Phase 4** – Corpus management & incremental indexing (`ingest --watch`, per‑workspace indexes).
- **Phase 5** – Perf pass: ANN (`hnswlib`) for dense, caching, concurrency.
- **Phase 6** – Minimal local UI (FastAPI + React) with export buttons, clickable citations.
- **Phase 7** – Packaging/DX (pipx install, config profiles, Windows niceties).

---

## 👤 Author
Erick Perales — Cloud Migration IT Architect, Cloud Migration Specialist  
GitHub: https://github.com/peralese
