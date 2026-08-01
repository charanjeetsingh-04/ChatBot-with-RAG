"""HTTP routes for ingest / chat / sessions / documents.

Dependency providers cache singletons via `lru_cache` so the embedder, vector
store, LLM, and session store are constructed once per process. Tests override
them via `app.dependency_overrides` on the FastAPI app.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    CitationOut,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentList,
    IngestResponse,
    SessionOut,
    TurnOut,
)
from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder, make_embedder
from app.generation.chat import ChatService, InMemorySessionStore, SessionStore
from app.generation.llm import LLM, make_llm
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore, VectorStore

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dependency providers (cached per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder. Loading sentence-transformers is expensive — cache it."""
    s = get_settings()
    return make_embedder(provider=s.embedding_provider, model=s.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Process-wide persistent vector store."""
    s = get_settings()
    return ChromaVectorStore(
        path=s.vector_store_path,
        collection_name=s.vector_store_collection,
    )


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """Process-wide LLM client."""
    s = get_settings()
    key_map: dict[str, str | None] = {
        "anthropic": s.anthropic_api_key,
        "gemini": s.gemini_api_key,
        "openai": s.openai_api_key,
    }
    return make_llm(
        provider=s.llm_provider,
        model=s.llm_model,
        api_key=key_map.get(s.llm_provider),
        max_tokens_default=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Process-wide session store (in-memory for v1)."""
    return InMemorySessionStore()


def get_retriever(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Retriever:
    """Per-request retriever bound to the cached singletons.

    Honours `HYBRID_SEARCH` and `RRF_K` from settings — flip the env var to
    enable BM25 + RRF fusion without code changes.
    """
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k,
        hybrid=settings.hybrid_search,
        rrf_k=settings.rrf_k,
    )


def get_chat_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm: Annotated[LLM, Depends(get_llm)],
    sessions: Annotated[SessionStore, Depends(get_session_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Per-request chat service."""
    return ChatService(retriever=retriever, llm=llm, sessions=sessions, settings=settings)


def get_ingestion_pipeline(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionPipeline:
    """Per-request ingestion pipeline."""
    return IngestionPipeline(vector_store=vector_store, embedder=embedder, settings=settings)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile | None, File()] = None,
    path: Annotated[str | None, Form()] = None,
) -> IngestResponse:
    """Ingest a document. Provide either an uploaded `file` or a `path` on the server."""
    if file is None and path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either an uploaded file or a path.",
        )
    if file is not None and path is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either file or path, not both.",
        )

    if file is not None:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        safe_name = Path(file.filename or "upload").name
        content_hash = hashlib.sha256(content).hexdigest()[:12]
        settings.uploads_path.mkdir(parents=True, exist_ok=True)
        target = settings.uploads_path / f"{content_hash}_{safe_name}"
        target.write_bytes(content)
    else:
        assert path is not None
        target = Path(path)

    try:
        result = pipeline.ingest(target)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    return IngestResponse(
        file_path=target.as_posix(),
        chunks_added=result.chunks_added,
        chunks_replaced=result.chunks_replaced,
    )


@router.post("/chat")
def chat(
    req: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """Stream a chat response as Server-Sent Events.

    Event sequence:
      `event: citations`  one line, carries the assigned session_id, the rewritten
                          query (if any), and the list of citations.
      `event: delta`      many lines, each carries a text chunk.
      `event: done`       one line, signals completion (with session_id).
    """
    session_id = req.session_id or str(uuid.uuid4())
    try:
        stream_handle = service.stream_reply(session_id, req.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    citations_payload = {
        "session_id": session_id,
        "rewritten_query": stream_handle.rewritten_query,
        "citations": [
            CitationOut(
                source_file=c.source_file,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                score=c.score,
                page=c.page,
            ).model_dump()
            for c in stream_handle.citations
        ],
    }

    def event_stream() -> Iterator[bytes]:
        yield _sse("citations", citations_payload)
        try:
            for delta in stream_handle.text_stream():
                yield _sse("delta", {"text": delta})
        except Exception as exc:  # noqa: BLE001 — surface any LLM/transport failure as SSE error
            logger.exception(
                "chat_stream_failed",
                session_id=session_id,
                error_type=type(exc).__name__,
            )
            yield _sse(
                "error",
                {
                    "session_id": session_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return
        yield _sse("done", {"session_id": session_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str,
    sessions: Annotated[SessionStore, Depends(get_session_store)],
) -> SessionOut:
    """Return the stored turns for a session. 404 if the session does not exist."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found.",
        )
    return SessionOut(
        id=session.id,
        turns=[TurnOut(role=t.role, content=t.content) for t in session.turns],
    )


@router.get("/documents", response_model=DocumentList)
def list_documents(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentList:
    """List every distinct source document currently stored."""
    return DocumentList(
        documents=[DocumentInfo(source_file=s) for s in vector_store.list_sources()]
    )


@router.delete("/documents/{document_id:path}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DeleteDocumentResponse:
    """Remove every chunk whose `source_file` metadata equals `document_id`.

    `document_id` is the POSIX source path returned by GET /documents (URL-encoded
    in the request path). Re-ingesting after deletion produces fresh chunks.
    """
    deleted = vector_store.delete_by_source(document_id)
    return DeleteDocumentResponse(source_file=document_id, chunks_deleted=deleted)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: object) -> bytes:
    """Serialise one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
