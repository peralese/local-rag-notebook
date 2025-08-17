# Local RAG — Fastembed + Chroma + FastAPI

A tiny, CPU‑friendly local document query service. It uses **fastembed** (`BAAI/bge-small-en-v1.5`) for embeddings, **ChromaDB** for the vector store, and **FastAPI** for an HTTP interface (with Swagger UI). Dockerized for a dead‑simple start.

---

## Features
- ✅ CPU‑only embeddings via **fastembed** (ThinkPad‑friendly)
- ✅ Persistent **Chroma** index under `./data/chroma`
- ✅ Simple API: **/reset**, **/ingest_all**, **/ingest_file**, **/count**, **/query**
- ✅ `docker compose up -d --build web` and you're running
- ♻️ Embedding backend can be flipped to **Ollama** later if needed

---

## Project Layout
```
repo-root/
├─ docker-compose.yml
├─ data/                 # mounted to /data in the container
│  └─ uploads/           # put your files here to ingest
├─ web/
│  ├─ app.py             # FastAPI service
│  ├─ embeddings.py      # fastembed (default) / optional ollama backend
│  ├─ ingest.py          # minimal text-file ingester
│  ├─ vectorstore.py     # Chroma wrapper
│  ├─ requirements.txt
│  └─ Dockerfile
└─ README.md
```
> Tip: add a placeholder `data/uploads/.gitkeep` so the folder exists in fresh clones (the `.gitignore` is set up for this).

---

## Requirements
- Docker & Docker Compose v2
- Ports: `8000` available locally

---

## Quick Start
1. **Put a test file** on the host at `data/uploads/small.txt`:
   ```text
   There is a secret code hidden here: ALPHA-42.
   I also like apples.
   ```

2. **Build & run**:
   ```bash
   docker compose up -d --build web
   ```

3. **Open the API docs**: http://localhost:8000/docs

4. **Reset** → **Ingest** → **Query** (via Swagger UI):
   - `POST /reset` → Execute
   - `POST /ingest_all` → Execute
   - `GET /count` → should be `> 0`
   - `GET /query` with `q=secret code` → you should see the chunk with `ALPHA-42`

### CLI equivalents (PowerShell/CMD-safe)
```bash
# reset
curl -X POST http://localhost:8000/reset

# ingest everything under /data/uploads
curl -X POST http://localhost:8000/ingest_all

# count
curl http://localhost:8000/count

# query
curl "http://localhost:8000/query?q=secret%20code&k=5"
```

---

## Configuration
Environment variables (set in `docker-compose.yml`):
- `EMBED_BACKEND` — `fastembed` (default) or `ollama`
- `EMBED_MODEL` — default `BAAI/bge-small-en-v1.5`
- `VECTOR_DIR` — default `/data/chroma`

> **Note on responses:** `embeddings` may show as `null` unless explicitly requested; Chroma still uses them internally for search.

### Switch to Ollama later (optional)
If you want to try Ollama embeddings again:
1. Run an Ollama container or service reachable from the `web` container.
2. Set env vars on `web`:
   ```yaml
   EMBED_BACKEND=ollama
   EMBED_MODEL=nomic-embed-text
   OLLAMA_HOST=http://ollama:11434
   ```
3. Rebuild `web`:
   ```bash
   docker compose up -d --no-deps --build web
   ```

---

## Supported Files
The reference `ingest.py` reads **plain text** formats (txt/md/py/etc.). Add handlers for PDFs/Docs when needed (see Roadmap).

---

## Troubleshooting
- **Windows PowerShell + heredocs**: not used here; all interactions go through HTTP or simple `curl`.
- **UTF‑8 BOM (`ï»¿`)**: save helper scripts as **UTF‑8 without BOM** or ANSI to avoid BOM issues.
- **`ModuleNotFoundError: vectorstore`**: always interact through the API, or run Python with the working dir that contains `vectorstore.py` (the service already does this).

---

## Roadmap (Recommended)
### Phase 1 — Core RAG polish
- [ ] Robust chunking (e.g., Markdown-aware, paragraph/sentence split, overlap)
- [ ] Metadata capture (filename, path, timestamp, source URL)
- [ ] Duplicate detection & upsert (content hash)
- [ ] Better query response: include `ids`, `distances`, and top‑k docs in output
- [ ] Simple `/answer` endpoint (extract short answer from top hit — no LLM)
- [ ] Basic unit tests for `VectorStore` and `ingest`

### Phase 2 — More formats & quality
- [ ] PDF via `pypdf` or `pdfminer.six` (text‑only to start)
- [ ] DOCX via `python-docx`
- [ ] HTML parsing (readability + boilerplate removal)
- [ ] Optional re‑ranking step (e.g., cross‑encoder) for better precision

### Phase 3 — UX & ops
- [ ] Minimal web UI (drag‑drop ingest, search box, results pane)
- [ ] Auth (API key or local login)
- [ ] Structured logging & basic metrics
- [ ] Backup/restore for Chroma index
- [ ] Background re‑ingest/refresh job

### Phase 4 — LLM answer synthesis (optional)
- [ ] Answer composition from retrieved context
- [ ] Citations (spans back to source chunks)
- [ ] Safety/guardrails (max tokens, cost controls if using external LLMs)

---

## License
MIT (suggested). Update as appropriate.

---

## Acknowledgements
- BAAI/bge-small-en-v1.5 — https://huggingface.co/BAAI/bge-small-en-v1.5
- ChromaDB — https://www.trychroma.com/
- FastAPI — https://fastapi.tiangolo.com/

## 👨‍💻 Author

Erick Perales  
Cloud Migration IT Architect, Cloud Migration Specialist
[https://github.com/peralese](https://github.com/peralese)