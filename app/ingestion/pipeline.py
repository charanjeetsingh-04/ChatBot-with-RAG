"""End-to-end ingestion orchestration.

Two layers:
- `ingest_file(path, settings)` → list[Chunk]. Load + chunk only; no embedding,
  no storage. Useful for tests, debugging, and re-chunking.
- `IngestionPipeline` → wraps the above with embedding + vector-store upsert.

Dedupe on re-ingest (Phase 8) will live in `IngestionPipeline.ingest` by
comparing existing `chunk_id`s in the store before upserting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import Chunk, RecursiveCharacterChunker
from app.ingestion.loaders import get_loader
from app.retrieval.vector_store import VectorStore

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting a single file.

    `chunks_replaced` counts chunks that existed for the same `source_file`
    before this call and were removed as part of the dedupe / replace step.
    On a fresh ingest it is 0; on an idempotent re-ingest it equals
    `chunks_added`.
    """

    file_path: Path
    chunks_added: int
    chunks_replaced: int = 0


def ingest_file(path: Path, settings: Settings | None = None) -> list[Chunk]:
    """Load and chunk a single file. Returns chunks without embedding or storing.

    The loader is dispatched from the file extension; chunk size and overlap
    come from `Settings` (env-configurable).
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot ingest, file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Cannot ingest, not a file: {path}")

    s = settings or get_settings()
    loader = get_loader(path)
    docs = loader.load(path)
    chunker = RecursiveCharacterChunker(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )
    return chunker.chunk_documents(docs)


class IngestionPipeline:
    """Load + chunk + embed + store, holding the embedder and vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        """Bind dependencies. `settings` defaults to the process-wide config."""
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = settings or get_settings()

    def ingest(self, path: Path, replace: bool = True) -> IngestResult:
        """Run the full ingestion pipeline for one file.

        With `replace=True` (default), every chunk already stored under the
        same `source_file` is deleted first, so re-ingesting an updated file
        removes the stale chunks rather than letting them linger alongside the
        new ones. Re-ingesting an unchanged file is a no-op net (same chunk
        ids reappear via upsert).
        """
        chunks = ingest_file(path, settings=self.settings)
        chunks_replaced = 0
        if chunks:
            if replace:
                source = str(chunks[0].metadata.get("source_file", path.as_posix()))
                chunks_replaced = self.vector_store.delete_by_source(source)
            embeddings = self.embedder.embed_documents([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        logger.info(
            "ingested",
            file_path=path.as_posix(),
            chunks_added=len(chunks),
            chunks_replaced=chunks_replaced,
        )
        return IngestResult(
            file_path=path, chunks_added=len(chunks), chunks_replaced=chunks_replaced
        )
