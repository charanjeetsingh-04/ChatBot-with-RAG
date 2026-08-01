"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Result of a single ingest call.

    `chunks_replaced` is the number of chunks that existed for the same source
    file before this call. On a fresh ingest it is 0; re-ingesting the same
    file gives `chunks_replaced == chunks_added`.
    """

    file_path: str
    chunks_added: int
    chunks_replaced: int = 0


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    message: str = Field(..., min_length=1, description="The user's question.")
    session_id: str | None = Field(
        default=None,
        description="Existing session id to continue; omit to start a new session.",
    )


class CitationOut(BaseModel):
    """A chunk that was provided to the model as context for the current turn."""

    source_file: str
    chunk_id: str
    snippet: str
    score: float
    page: int | None = None


class TurnOut(BaseModel):
    """One stored turn of a session, returned by GET /sessions/{id}."""

    role: str
    content: str


class SessionOut(BaseModel):
    """Full session history."""

    id: str
    turns: list[TurnOut]


class DocumentInfo(BaseModel):
    """One ingested source document."""

    source_file: str


class DocumentList(BaseModel):
    """Listing of known source documents in the vector store."""

    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """Result of DELETE /documents/{document_id}."""

    source_file: str
    chunks_deleted: int
