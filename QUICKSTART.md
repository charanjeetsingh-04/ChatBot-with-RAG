# Quickstart — Running the RAG Chatbot

Step-by-step to get the chatbot running on a fresh machine.
For the deeper "what / why" see [README.md](README.md).

---

## 1. Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (fast Python package manager)
- An **API key** for one of the supported LLM providers:
  - Anthropic — <https://console.anthropic.com/>
  - Google Gemini — <https://aistudio.google.com/apikey>

## 2. Install dependencies

From the project root:

```powershell
uv sync --extra dev
```

This creates `.venv/`, resolves `uv.lock`, and installs everything (FastAPI,
ChromaDB, sentence-transformers, the LLM SDKs, Streamlit, pytest, etc.).

## 3. Configure `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and pick a provider:

**For Google Gemini (free tier available):**

```ini
GEMINI_API_KEY=AIza...your-key...
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
```

**For Anthropic Claude:**

```ini
ANTHROPIC_API_KEY=sk-ant-...your-key...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-7
```

Everything else (chunking, retrieval, vector store path) has sensible defaults.

## 4. Start the backend

```powershell
uv run uvicorn app.main:app --reload
```

The FastAPI server will boot on <http://127.0.0.1:8000>.
Sanity check:

```powershell
curl http://127.0.0.1:8000/health
```

Expected output:

```json
{"status":"ok","llm_provider":"gemini","llm_model":"gemini-2.5-flash","embedding_provider":"sentence-transformers"}
```

> **First request is slow.** The embedding model (`BAAI/bge-small-en-v1.5`,
> ~130 MB) downloads from Hugging Face on the first chat or ingest call.
> Subsequent calls use the local cache and are instant.

## 5. Start the UI

In a **separate terminal**:

```powershell
uv run streamlit run ui/streamlit_app.py
```

Then open <http://localhost:8501>.

## 6. Use it

1. **Expand the sidebar** (click the `>` arrow at the top-left if it's collapsed).
2. **Upload a document** — PDF, TXT, MD, or DOCX. You'll see "Ingested N chunks".
3. **Ask a clear question** about that document in the chat input.
4. **Click "Sources (N)"** under each answer to see the exact chunks the LLM used.

### Tips for getting good answers

The model is configured to refuse hallucinated answers — it will reply *"I don't
know based on the provided context."* if your question is too vague or the answer
isn't in the documents. Phrase questions specifically:

| Bad | Good |
|---|---|
| `describe` | `Summarize this document.` |
| `tell me about it` | `What does Module 5 cover?` |
| `more` | `List the key outcomes.` |

---

## Optional: verify everything is wired correctly

```powershell
uv run pytest        # ~156 tests
uv run mypy app      # strict on app/
uv run ruff check app tests ui
```

All three should be green.

---

## Useful endpoints (Swagger at `/docs`)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness; reports provider + model |
| `POST` | `/ingest` | Upload a document (`file=` or `path=`) |
| `POST` | `/chat` | Stream a chat reply as SSE |
| `GET` | `/sessions/{id}` | Fetch a conversation's turns |
| `GET` | `/documents` | List ingested source files |
| `DELETE` | `/documents/{id}` | Remove every chunk for one source |

## Troubleshooting

- **"Backend unreachable" in the Streamlit sidebar** — the FastAPI server isn't
  running (or is on a different port). Check Step 4.
- **`401` from the LLM** — wrong API key or wrong provider/key combination.
  Re-check the `.env` step.
- **First chat hangs for 30–60 s** — the embedding model is downloading. Wait it
  out; only happens once.
- **Ingest succeeds but chat says "I don't know"** — the answer probably isn't
  in the uploaded document, or the question is too vague. See the tips above.
