"""Vector store interface and the ChromaDB implementation.

Other backends (Qdrant, pgvector, ...) drop in by subclassing `VectorStore`.
Distances are converted to similarity scores in [0, 1] so callers don't have to
know which space the underlying store uses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb

from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class SearchResult:
    """One match returned by a vector search.

    `score` is in [0, 1]; higher = more relevant. For cosine-space stores this
    is `1 - cosine_distance`.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class VectorStore(ABC):
    """Abstract storage for chunks + their embeddings."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Upsert chunks with their embeddings. Idempotent on `chunk_id`."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top-`k` chunks closest to `query_embedding`."""

    @abstractmethod
    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks whose `source_file` metadata equals `source_file`.

        Returns the number of chunks removed.
        """

    @abstractmethod
    def list_sources(self) -> list[str]:
        """Return the sorted list of distinct `source_file` values in the store."""

    @abstractmethod
    def count(self) -> int:
        """Total number of chunks currently stored."""

    @abstractmethod
    def get_all(self) -> list[SearchResult]:
        """Return every stored chunk (score is 0; this is not a ranked result).

        Used by hybrid retrieval to build an in-memory BM25 index. Heavy on large
        collections — only call when needed.
        """


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store using cosine space."""

    def __init__(
        self,
        path: Path | None,
        collection_name: str = "rag_chunks",
    ) -> None:
        """Open or create a Chroma collection.

        `path=None` uses an in-memory (ephemeral) client; pass a directory for
        persistent storage.
        """
        if path is None:
            self._client = chromadb.EphemeralClient()
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Upsert chunks. Duplicate `chunk_id`s overwrite their prior entry."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=cast(Any, [[float(x) for x in e] for e in embeddings]),
            documents=[c.content for c in chunks],
            metadatas=cast(Any, [_scalar_metadata(c.metadata) for c in chunks]),
        )

    def search(
        self,
        query_embedding: Sequence[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top-`k` matches as `SearchResult`s ordered by score desc."""
        if k <= 0:
            return []
        res = self._collection.query(
            query_embeddings=cast(Any, [[float(x) for x in query_embedding]]),
            n_results=k,
            where=cast(Any, where),
        )
        ids_batch = res.get("ids") or [[]]
        docs_batch = res.get("documents") or [[]]
        metas_batch = res.get("metadatas") or [[]]
        dists_batch = res.get("distances") or [[]]
        ids = ids_batch[0] if ids_batch else []
        docs = docs_batch[0] if docs_batch else []
        metas = metas_batch[0] if metas_batch else []
        dists = dists_batch[0] if dists_batch else []

        results: list[SearchResult] = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=str(doc or ""),
                    metadata=dict(meta or {}),
                    score=max(0.0, 1.0 - float(dist)),
                )
            )
        return results

    def delete_by_source(self, source_file: str) -> int:
        """Remove every chunk whose metadata matches the given source file."""
        before = self._collection.count()
        self._collection.delete(where=cast(Any, {"source_file": source_file}))
        after = self._collection.count()
        return int(before - after)

    def list_sources(self) -> list[str]:
        """List distinct `source_file` metadata values across the collection."""
        result = self._collection.get(include=["metadatas"])
        metas = result.get("metadatas") or []
        sources: set[str] = set()
        for m in metas:
            if m is None:
                continue
            src = m.get("source_file")
            if isinstance(src, str):
                sources.add(src)
        return sorted(sources)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return int(self._collection.count())

    def get_all(self) -> list[SearchResult]:
        """Return every stored chunk with score=0 (unranked)."""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[SearchResult] = []
        for chunk_id, doc, meta in zip(ids, docs, metas, strict=False):
            out.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=str(doc or ""),
                    metadata=dict(meta or {}),
                    score=0.0,
                )
            )
        return out


def _scalar_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Coerce metadata values to Chroma-supported scalar types."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, bool | int | float | str):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out
