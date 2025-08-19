# local-rag-notebook
Local, LLM-optional “NotebookLM-style” search and question-answering over your own documents.
**Phase 0** focuses on: ingestion, hybrid retrieval (BM25 + dense embeddings) with RRF fusion, and extractive answers with citations *(File, Heading, Page)* via a simple CLI.

---

## ✨ Features (Phase 0)

- **Offline-first**: no external APIs required.
- **Hybrid retrieval**: lexical (BM25) + dense embeddings → **RRF** merge, with neighbor expansion.
- **Extractive answers**: returns quoted snippets plus **(File, Heading, Page)** citations.
- **PDF text normalization**: de-hyphenation, bullet cleanup (`•`, ``, etc. → `- `).
- **Table-aware (MVP)**: CSV/TSV rows flattened to `key: value` lines for strong lexical matches.
- **Target by file**: `--files` flag to restrict search to certain filenames.

---

## 📦 Requirements

- Python **3.11+** recommended. Works on **Windows** (PowerShell), **macOS**, and **Linux**.
  - On **Windows + Python 3.13**, this build avoids native compilers (no PyMuPDF, no pandas).
- Disk: a few hundred MB for embeddings and indexes (tiny at your current scale).
- No GPU required.

**Runtime deps** (see `requirements.txt`):
- `sentence-transformers` (embeddings; CPU ok)
- `rank-bm25`, `numpy`, `pypdf`, `pyyaml`, `tqdm`, `pydantic`

> If `sentence-transformers` pulls in a PyTorch wheel that isn’t available for your Python version, see **Troubleshooting** below.

---

## 🚀 Quickstart

```bash
# Create and activate a venv
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Put your docs in ./data (PDF / MD / TXT / CSV / TSV)
python cli.py ingest ./data

# Ask questions
python cli.py query "List the prerequisites"
python cli.py query "What ports are required by the database?"
# Limit to specific files (substring match):
python cli.py query "installation steps" --files "AWS-Certified-Machine-Learning-Engineer-Associate_Exam-Guide.pdf"
```

Example output:
```
=== ANSWER ===
- AWS Auto Scaling
- AWS CloudFormation
- ...

=== CITATIONS ===
- ...\AWS-...Exam-Guide.pdf | AWS-...Exam-Guide.pdf > Page 18 | Page 18
```

---

## 🧰 CLI Reference

```
# Ingest a folder of documents (build/refresh indexes)
python cli.py ingest <PATH> [--config config.yaml]

# Query the indexed corpus
python cli.py query "<QUESTION>" [--k 8] [--files "name1,name2"] [--config config.yaml]
```

- `--k`: number of contexts fed to the answerer (default **8**).
- `--files`: comma-separated substrings to filter by file path (case-insensitive).
- `--config`: path to a config file (defaults to `config.yaml`).

---

## ⚙️ Configuration (`config.yaml`)

```yaml
app:
  data_dir: ./data
  index_dir: ./index_store

models:
  embedding: sentence-transformers/all-MiniLM-L6-v2  # 384-dim, CPU-friendly

ingest:
  chunk_tokens: 700
  overlap_tokens: 90
  ocr: auto  # placeholder for future (PyPDF build uses text extraction)

retrieval:
  top_k_lexical: 40
  top_k_dense: 40
  rrf_k: 60
  neighborhood: 1

synthesis:
  enabled: false
  provider: ollama
  model: llama3.2:3b-instruct
```

**Knobs you’ll tweak most often**
- `ingest.chunk_tokens` / `ingest.overlap_tokens`
- `retrieval.top_k_lexical` / `retrieval.top_k_dense` / `retrieval.rrf_k`
- `retrieval.neighborhood` (± adjacent chunks from the same section)

---

## 🧪 How it Works

1. **Ingest**
   - Parse PDFs (via **PyPDF**), Markdown/TXT, CSV/TSV.
   - Normalize text (bullets, de-hyphenation, whitespace).
   - Create multi-granularity entries: **section** and **chunk** (sliding window).
   - Persist chunk metadata to `index_store/chunks.jsonl`.

2. **Index**
   - **BM25** index over tokens.
   - **Dense embeddings** via `sentence-transformers` (FAISS if installed; otherwise NumPy cosine).
   - Store IDs & embeddings in `index_store`.

3. **Retrieve**
   - Query both BM25 and dense → **RRF** fusion.
   - **Neighbor expansion** to pull adjacent chunks from matching sections.

4. **Answer**
   - Lightweight classifier routes *fact/list/compare/compute*.
   - For **list** queries, prefer true bullet lines when available.
   - Return **extractive snippet(s)** + **(File, Heading, Page)** citations.

---

## 🗂️ Supported File Types (Phase 0)

- `.pdf` (digital text) — per-page sections.
- `.md`, `.txt` — Markdown headings become sections (fallback: full text).
- `.csv`, `.tsv` — rows become `Row N: header: value; ...` lines.

---

## 💡 Tips for Better Results

- Keep files inside **`./data`** and re-run `ingest` after changes.
- For PDFs, prefer **digital text** (not scanned images). OCR hooks can be added later.
- If results look like one big blob, increase `overlap_tokens` a bit and re-ingest.
- Use `--files` to target a particular document while validating.

---

## 🧯 Troubleshooting

**Problem:** Install errors about Visual Studio, Meson, or building native code  
**Why:** Some packages ship source-only for new Python versions  
**Fix (already applied here):** This build avoids PyMuPDF and pandas, using **PyPDF** and the **built-in csv** module.

**Problem:** `sentence-transformers` / torch wheel not found (Python 3.13)  
**Fix A (easiest):** Use a **Python 3.11** virtualenv for this project  
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```
**Fix B (stay on 3.13):** Install a CPU PyTorch wheel first, then run the rest  
```powershell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install -r requirements.txt
```

**Problem:** “No matching content found.”  
- Check `index_store/chunks.jsonl` to confirm text was extracted.
- Increase `retrieval.top_k_*` and `--k`.
- Try a more explicit query (include exact terms that appear in the document).

---

## 🧭 Roadmap

- **Phase 1**: Cross-encoder **reranker** (precision boost), eval harness (Recall@K, MRR), richer traces.
- **Phase 2**: Optional **LLM synthesis** (off by default) for nicer prose; citations preserved.
- **Phase 3**: **FastAPI** service + minimal web UI; clickable citations.
- **Phase 4**: Hardening (config profiles, caching, OCR on demand, multilingual option).

---

## 🗂️ Project Structure

```
local-rag-notebook/
  cli.py
  config.yaml
  requirements.txt
  README.md
  data/                # your docs (gitignored)
  index_store/         # built indexes (gitignored)
  logs/                # jsonl logs
  local_rag_notebook/
    app.py
    ingest/
      pdf.py           # PyPDF + normalization
      md_txt.py        # Markdown/TXT + normalization
      csv_tsv.py       # csv module → key:value rows
      normalize.py     # chunking (700/90)
      clean.py         # text normalization (bullets, hyphens)
    index/
      dense.py         # embeddings (FAISS if present; else NumPy cosine)
      lexical.py       # BM25 + robust JSONL reader
      schema.py
    retrieve/
      fuse.py          # RRF + neighbor expansion
    answer/
      classify.py      # fact/list/compare/compute
      extract.py       # extractive snippets + bullet detection
    utils/
      log.py
```

---

## 📝 License

Choose a license (e.g., MIT/Apache-2.0) and add it here.

## 👨‍💻 Author

Erick Perales  
Cloud Migration IT Architect, Cloud Migration Specialist
[https://github.com/peralese](https://github.com/peralese)
