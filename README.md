# Local RAG — Fastembed + Chroma + FastAPI

A tiny, CPU-friendly local document query service. It uses **fastembed** (`BAAI/bge-small-en-v1.5`) for embeddings, **ChromaDB** for the vector store, and **FastAPI** for an HTTP interface (with Swagger UI). Dockerized for a dead-simple start.

---

## Features
- ✅ CPU-only embeddings via **fastembed** (ThinkPad-friendly)
- ✅ Persistent **Chroma** index under `./data/chroma`
- ✅ Markdown-aware **chunking** with overlap
- ✅ **Upsert** (de-dupe by content-hash IDs)
- ✅ Rich **metadata** per chunk (path, name, size, mtime, chunk index/count)
- ✅ Simple API: **/reset**, **/ingest_all**, **/ingest_file**, **/count**
- ✅ Search API: **/query** with
  - filters: `path_substr`, `mtime_gt`, `mtime_lt`
  - thresholds: `max_distance` or `min_score` (where `score = 1 - distance`)
  - optional `include_embeddings=true`
  - returns `documents`, `metadatas`, `distances`, `scores`, and `ids`
- ✅ **/delete_by_path** (purges embeddings for a file; does *not* delete the source file)
- ✅ **/answer** (simple heuristic answer from the top hit)
- ✅ Swagger UI at http://localhost:8000/docs
- ♻️ Embedding backend can later be switched to **Ollama** if desired

---

## Project Layout
```
repo-root/
├─ docker-compose.yml
├─ data/                 # mounted to /data in the container
│  └─ uploads/           # put your files here to ingest
├─ web/
│  ├─ app.py             # FastAPI service (endpoints)
│  ├─ embeddings.py      # fastembed (default) / optional ollama backend
│  ├─ ingest.py          # text-file ingester (chunking + metadata + upsert)
│  ├─ vectorstore.py     # Chroma wrapper (query, filters, delete-by-path)
│  ├─ chunker.py         # markdown/paragraph chunker with overlap
│  ├─ requirements.txt
│  └─ Dockerfile
└─ README.md
```
> Tip: add a placeholder `data/uploads/.gitkeep` so the folder exists in fresh clones.

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
curl -X POST "http://localhost:8000/ingest_all?root=%2Fdata%2Fuploads"

# count
curl http://localhost:8000/count

# query (with a min normalized score)
curl "http://localhost:8000/query?q=secret%20code&min_score=0.8"

# delete embeddings for a single file (does NOT delete the file on disk)
curl -X POST "http://localhost:8000/delete_by_path?path=%2Fdata%2Fuploads%2Fsmall.txt&exact=true"
```

---

## API Reference (high-level)

### `POST /reset`
Drops and recreates the `docs` collection.

### `POST /ingest_all?root=/data/uploads`
Walks `root`, chunks supported files, upserts into Chroma.

### `POST /ingest_file?path=/data/uploads/file.txt`
Ingest exactly one file.

### `GET /count`
Returns the current number of chunks in Chroma.

### `GET /query`
Parameters:
- `q` (required) — query text
- `k` — top-k (default 5)
- `include_embeddings` — `true|false`
- `path_substr` — client-side filter on `metadata.path` substring
- `mtime_gt`, `mtime_lt` — UNIX seconds, server-side filter on `metadata.mtime`
- `max_distance` — keep hits with distance ≤ this
- `min_score` — keep hits with score ≥ this (`score = 1 - distance`)

Returns:
- `ids`, `documents`, `metadatas`, `distances`, `scores` (and `embeddings` if requested)

### `GET /answer`
Parameters: `q`, `k` (default 5), optional `path_substr`  
Returns a short `answer` + one `evidence` block (id, distance, score, metadata, snippet).

### `POST /delete_by_path`
Parameters:
- `path` — full container path or a substring
- `exact` — when `true`, deletes by exact path via server-side filter; when `false`, deletes by substring (client-side match + delete by ids)

> **Note:** This only deletes **embeddings** from Chroma. Your source file(s) on disk remain untouched.

---

## Configuration
Environment variables (set in `docker-compose.yml`):
- `EMBED_BACKEND` — `fastembed` (default) or `ollama`
- `EMBED_MODEL` — default `BAAI/bge-small-en-v1.5`
- `VECTOR_DIR` — default `/data/chroma`
- `HF_HOME` — set to `/data/hf` to persist model cache across runs (recommended)
- (If needed) `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` — for corporate networks

> **Note on responses:** `embeddings` may show as `null` unless explicitly requested; Chroma still uses them internally for search.

### Switch to Ollama later (optional)
If you want to try Ollama embeddings again:
1. Run an Ollama service reachable from the `web` container.
2. Set:
   ```yaml
   EMBED_BACKEND=ollama
   EMBED_MODEL=nomic-embed-text
   OLLAMA_HOST=http://ollama:11434
   ```
3. Rebuild:
   ```bash
   docker compose up -d --no-deps --build web
   ```

---

## Supported Files
`ingest.py` handles **plain text** formats by default (`.txt`, `.md`, `.py`, `.json`, `.csv`, `.yaml`, `.yml`, `.toml`).  
Add PDF/DOCX/HTML readers in Phase 2.

---

## Troubleshooting
- **First-run model download**: the fastembed model is fetched on first run; set `HF_HOME=/data/hf` to cache it.
- **Chroma telemetry warning**: harmless; telemetry is disabled, but a message may still appear.
- **`500` on query**: avoid unsupported include keys; we only use `documents`, `metadatas`, `distances` (and `embeddings` when requested).
- **Path filtering**: `path_substr` is applied **client-side** (Chroma 0.5.x doesn’t support `$contains` on metadata). Use `POST /delete_by_path?exact=true` for exact server-side deletes.
- **Windows paths**: all paths inside the container are Linux-style (e.g., `/data/uploads/...`).

---

## Roadmap

### Phase 1 — Core RAG polish (✅ shipped)
- ✅ Markdown-aware chunking with overlap  
- ✅ Metadata capture (path/name/size/mtime/chunk info)  
- ✅ Upsert (content-hash)  
- ✅ Richer `/query` (ids, distances, scores, metadata; optional embeddings)  
- ✅ Filters: `path_substr`, `mtime_gt`, `mtime_lt`  
- ✅ Thresholds: `max_distance`, `min_score` (score = `1 - distance`)  
- ✅ `/delete_by_path` (vector DB only)  
- ✅ `/answer` (heuristic)  
- ✅ Basic tests (chunker/vectorstore; filters/delete)

### Phase 2 — More formats & quality
- [ ] PDF via `pypdf` / `pdfminer.six`
- [ ] DOCX via `python-docx`
- [ ] HTML parsing (readability; boilerplate removal)
- [ ] Optional cross-encoder **re-ranking** over top-k
- [ ] Smarter chunking per format (keep headings with body)

### Phase 3 — UX & ops
- [ ] Minimal web UI (drag-drop ingest, search, results)
- [ ] Auth (API key or local login)
- [ ] `/export` & `/import` for the Chroma collection
- [ ] Structured logging & basic metrics
- [ ] Watcher to auto-ingest changed files (hash-aware)

### Phase 4 — LLM answer synthesis (optional)
- [ ] Compose answers from retrieved context with citations
- [ ] Guardrails (token limits, model controls)
- [ ] Cost/latency tracking if external LLMs are used

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