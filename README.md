# RAG Chatbot

Production-quality Retrieval-Augmented Generation chatbot. Upload PDF / TXT /
Markdown / DOCX documents, ask questions, get answers grounded in the corpus
with citations back to the source chunks.

Built around small, swappable interfaces: changing the embedding model, vector
store, or LLM is a one-file change. Every public function is type-hinted and
docstringed; `mypy --strict` and `ruff` are clean; ~140 tests cover ingestion,
retrieval, generation, and the HTTP API.

## Stack

| Layer | Default | Swap point |
| --- | --- | --- |
| Backend | FastAPI · Python 3.11+ | — |
| LLM | Anthropic Claude (`claude-opus-4-7`) or Google Gemini (`gemini-2.5-flash`) | `app/generation/llm.py` (`make_llm`) |
| Embeddings | `sentence-transformers` · `BAAI/bge-small-en-v1.5` | `app/embeddings/embedder.py` (`make_embedder`) |
| Vector store | ChromaDB (persistent, local, cosine) | `app/retrieval/vector_store.py` (`VectorStore` ABC) |
| Chunking | Recursive character splitter, deterministic SHA-256 chunk ids | `app/ingestion/chunker.py` |
| UI | Streamlit (v1) — Next.js plug point noted in code | `ui/streamlit_app.py` |
| Config | `pydantic-settings` reading `.env` | `app/config.py` |
| Logging | `structlog` (console renderer; swap for JSON in prod) | `app/logging.py` |

## Architecture

```
                              ┌────────────────────────────────────┐
   ┌──────────────────┐       │              FastAPI               │
   │  Streamlit UI    │       │                                    │
   │  ui/streamlit_   │◀─SSE──│  POST  /ingest                     │
   │   app.py         │       │  POST  /chat   (text/event-stream) │
   │                  │       │  GET   /sessions/{id}              │
   │  - upload file   │──────▶│  GET   /documents                  │
   │  - chat input    │       │  DELETE /documents/{id}            │
   │  - sources panel │       │  GET   /health                     │
   └──────────────────┘       └────────────────┬───────────────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────┐
        ▼                                      ▼                              ▼
┌─────────────────┐               ┌─────────────────────┐         ┌─────────────────────┐
│  Ingestion      │               │  Retrieval          │         │  Generation         │
│  pipeline       │               │                     │         │                     │
│                 │               │  Retriever          │         │  ChatService        │
│  load loader →  │               │   - dense (cosine)  │         │   - query rewrite   │
│  chunk          │  ──upsert──▶  │   - hybrid (BM25 +  │  ──▶    │   - retrieve        │
│  embed          │               │     dense via RRF)  │         │   - inject context  │
│  add to store   │               │                     │         │   - stream answer   │
└────────┬────────┘               │  VectorStore (ABC)  │         │                     │
         │                        │   - ChromaVectorStore         │  LLM (ABC)          │
         ▼                        └─────────┬───────────┘         │   - AnthropicLLM    │
   uploads/<hash>_<name>                    │                     │     (streaming +    │
                                            ▼                     │      prompt cache)  │
                                       .chroma/                   └─────────────────────┘
                                       (persistent index)
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
# 1. Clone
git clone https://github.com/TXsShadowFox/AI-ChatBot-with-RAG.git
cd AI-ChatBot-with-RAG

# 2. Install deps (creates .venv, resolves lock)
uv sync --extra dev

# 3. Copy env template and add your Anthropic key
Copy-Item .env.example .env
# then edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# 4. (optional) Override config via .env — see "Configuration" below
```

On first chat call the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) is
downloaded from Hugging Face and cached locally.

## Run

```powershell
# Terminal 1 — API
uv run uvicorn app.main:app --reload

# Terminal 2 — UI
uv run streamlit run ui/streamlit_app.py
```

Then visit:

- API: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Streamlit: <http://localhost:8501>

## How to ingest

Three options.

**Via the UI**: open the Streamlit app, drop a PDF/TXT/MD/DOCX into the
sidebar uploader. The file is persisted under `uploads/` with a content-hash
prefix so re-uploading identical content is idempotent.

**Via curl with a server-side path**:

```powershell
curl -X POST -F "path=$(pwd)/docs/spec.pdf" http://127.0.0.1:8000/ingest
```

**Via curl with a file upload**:

```powershell
curl -X POST -F "file=@docs/spec.pdf;type=application/pdf" `
    http://127.0.0.1:8000/ingest
```

Response shape:

```json
{
  "file_path": "uploads/8f2c1.../spec.pdf",
  "chunks_added": 142,
  "chunks_replaced": 0
}
```

Re-ingesting the same file produces `chunks_replaced == chunks_added` and net
zero change (dedupe by deterministic `chunk_id`).

## How to chat

The `/chat` endpoint streams **Server-Sent Events**. Three event types in
order:

1. `event: citations` — `{session_id, rewritten_query, citations: [...]}`
2. `event: delta` — `{text}` — repeated for each LLM text chunk
3. `event: done` — `{session_id}`, OR `event: error` — `{error_type, message, session_id}` if the stream fails mid-way

```powershell
curl -N -X POST http://127.0.0.1:8000/chat `
    -H "Content-Type: application/json" `
    -d '{\"message\": \"What is in this document?\"}'
```

Pass the `session_id` from the citations event in subsequent requests to keep
conversation history; query rewriting kicks in once history is non-empty.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness; returns provider + model id |
| `POST` | `/ingest` | Ingest one doc (`file=` upload OR `path=` form field) |
| `POST` | `/chat` | Stream a chat response as SSE |
| `GET` | `/sessions/{id}` | Return session turns (literal user/assistant messages, no injected context) |
| `GET` | `/documents` | List distinct source files in the vector store |
| `DELETE` | `/documents/{source_file:path}` | Delete every chunk for that source. Returns `chunks_deleted`. |

Full schemas live in [app/api/schemas.py](app/api/schemas.py); Swagger UI is at
`/docs` when the server is running.

## Configuration

All knobs are env vars (or via `.env`). See [.env.example](.env.example).

| Key | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `gemini`; `openai` is reserved |
| `LLM_MODEL` | `claude-opus-4-7` | Model id for the chosen provider (e.g. `gemini-2.5-flash`) |
| `LLM_MAX_TOKENS` | `1024` | Default per-response cap |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Model id |
| `VECTOR_STORE_PATH` | `./.chroma` | Persistent Chroma dir |
| `VECTOR_STORE_COLLECTION` | `rag_chunks` | Chroma collection name |
| `UPLOADS_PATH` | `./uploads` | Where uploaded files land |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Recursive char splitter |
| `TOP_K` | `5` | Retriever neighbours |
| `HYBRID_SEARCH` | `false` | Toggle dense + BM25 with RRF |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | uvicorn bind |
| `LOG_LEVEL` | `INFO` | structlog filter level |

### Swapping providers

- **LLM**: change `LLM_PROVIDER` / `LLM_MODEL`. Built-in providers are
  `anthropic` (streaming + prompt caching) and `gemini` (streaming via the
  `google-genai` SDK; caching is a no-op — Gemini's cache API has a token
  floor that small system prompts don't meet). To add a new provider,
  subclass `LLM` in [app/generation/llm.py](app/generation/llm.py) and wire
  it into `make_llm()`. No call sites change.
- **Embedder**: change `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`. To add a new
  provider, subclass `Embedder` in [app/embeddings/embedder.py](app/embeddings/embedder.py).
- **Vector store**: subclass `VectorStore` in
  [app/retrieval/vector_store.py](app/retrieval/vector_store.py) (Qdrant /
  pgvector / etc.) and update the `get_vector_store` dependency in
  [app/api/routes.py](app/api/routes.py).

### Hybrid search

Set `HYBRID_SEARCH=true` in `.env` (or env vars). The retriever then runs both
dense (cosine via Chroma) and sparse (BM25 over `vector_store.get_all()`) for
each query, fuses the two rankings via Reciprocal Rank Fusion:

```
score(d) = Σ 1 / (RRF_K + rank_in_list(d))
```

…and returns the top-`TOP_K`. BM25 is built in memory per query — fine for
small/medium corpora, swap for a persistent sparse index at scale.

## Project layout

```
rag-chatbot/
  app/
    main.py                  FastAPI app: router, exception handlers, startup
    config.py                pydantic-settings
    logging.py               structlog setup
    ingestion/
      loaders.py             PDF / TXT / MD / DOCX + LOADER_REGISTRY
      chunker.py             Recursive char splitter + deterministic chunk_id
      pipeline.py            ingest_file + IngestionPipeline (dedupe-on-reingest)
    retrieval/
      vector_store.py        VectorStore ABC + ChromaVectorStore + SearchResult
      retriever.py           Dense + hybrid (BM25 + RRF)
      reranker.py            Stub for a future cross-encoder reranker
    embeddings/
      embedder.py            Embedder ABC + SentenceTransformerEmbedder + factory
    generation/
      llm.py                 LLM ABC + AnthropicLLM (streaming + prompt cache)
      prompts.py             RAG system prompt + query-rewrite template
      chat.py                Session memory + ChatService + ChatStream
    api/
      routes.py              All endpoints + dependency providers
      schemas.py             Pydantic request/response models
  ui/
    streamlit_app.py         Sidebar uploader + chat panel + sources expander
  tests/                     ~145 tests covering every layer
  pyproject.toml
  .env.example
```

## Testing, type-checking, linting

```powershell
uv run pytest             # ~145 tests
uv run mypy               # strict on app/
uv run mypy ui/streamlit_app.py    # strict on UI
uv run ruff check app tests ui
```

Quality bar (all currently green):

- `mypy --strict` clean on `app/` and `ui/`
- `ruff` clean
- Every public function has a docstring
- Pytest happy-path + edge-case coverage on ingestion and retrieval (and on
  every other layer too)

## Build phases (history)

1. **Skeleton.** Scaffolding, config, `.env.example`, README stub.
2. **Ingestion (PDF).** Loader interface + chunker + tests on a tiny fixture.
3. **Retrieval.** `VectorStore` ABC + Chroma impl + `Retriever`; round-trip
   retrieval verified.
4. **LLM.** `LLM` ABC + Anthropic streaming impl with prompt caching.
5. **Chat orchestration.** Query rewriting, citation extraction, session memory.
6. **FastAPI routes** wired end-to-end; curl-testable.
7. **Streamlit UI** with file uploader, chat panel, expandable sources.
8. **More loaders + hybrid search + dedupe-on-reingest.**
9. **Polish.** Error handling, structlog logging, this README.

## Non-goals (v1)

Auth / multi-tenant isolation · fine-tuning · agentic tool use beyond retrieval
· k8s deployment configs. Docker is fine; k8s is out.

## Screenshots

Add screenshots of the running UI to `docs/screenshots/` and reference them
here:

- `docs/screenshots/chat.png` — chat conversation with the Sources expander open
- `docs/screenshots/upload.png` — sidebar after uploading a PDF
- `docs/screenshots/citations.png` — close-up of one citation showing source file, page, and score

(These aren't generated automatically — capture them by running the app and
saving from your browser.)
