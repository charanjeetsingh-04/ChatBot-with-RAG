"""Tests for the ChromaDB-backed vector store."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.vector_store import ChromaVectorStore, SearchResult
from tests.conftest import HashEmbedder


def _make_chunk(
    chunk_id: str,
    content: str,
    source_file: str = "/tmp/doc.pdf",
    page: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata={"source_file": source_file, "page": page, "chunk_index": chunk_index},
    )


@pytest.fixture()
def store() -> ChromaVectorStore:
    # Chroma's ephemeral client shares state across instances with the same
    # collection name, so each test gets a fresh collection.
    return ChromaVectorStore(path=None, collection_name=f"test_{uuid.uuid4().hex[:8]}")


def test_empty_store_counts_zero(store: ChromaVectorStore) -> None:
    assert store.count() == 0
    assert store.list_sources() == []


def test_add_then_count(store: ChromaVectorStore, hash_embedder: HashEmbedder) -> None:
    chunks = [
        _make_chunk("a1", "apollo moon armstrong", chunk_index=0),
        _make_chunk("a2", "voyager interstellar probe", chunk_index=1),
    ]
    embs = hash_embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embs)
    assert store.count() == 2
    assert store.list_sources() == ["/tmp/doc.pdf"]


def test_search_returns_added_chunk_for_matching_query(
    store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    chunks = [
        _make_chunk("apollo", "Apollo 11 Armstrong Aldrin Moon landing 1969"),
        _make_chunk("voyager", "Voyager 1 Voyager 2 interstellar Golden Record"),
        _make_chunk("jwst", "James Webb Space Telescope infrared L2"),
    ]
    embs = hash_embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embs)

    query_emb = hash_embedder.embed_query("who walked on the Moon during Apollo")
    results = store.search(query_emb, k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "apollo"
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Scores should be monotonically non-increasing.
    assert results[0].score >= results[1].score >= results[2].score


def test_search_with_zero_k_returns_empty(
    store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    assert store.search(hash_embedder.embed_query("anything"), k=0) == []


def test_upsert_is_idempotent(store: ChromaVectorStore, hash_embedder: HashEmbedder) -> None:
    chunks = [_make_chunk("dup", "same content")]
    embs = hash_embedder.embed_documents([c.content for c in chunks])
    store.add(chunks, embs)
    store.add(chunks, embs)
    assert store.count() == 1


def test_delete_by_source_removes_matching_chunks(
    store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    a = [
        _make_chunk("a1", "alpha", source_file="/tmp/a.pdf", chunk_index=0),
        _make_chunk("a2", "alpha two", source_file="/tmp/a.pdf", chunk_index=1),
    ]
    b = [_make_chunk("b1", "beta", source_file="/tmp/b.pdf", chunk_index=0)]
    store.add(a + b, hash_embedder.embed_documents([c.content for c in a + b]))
    assert store.count() == 3

    deleted = store.delete_by_source("/tmp/a.pdf")
    assert deleted == 2
    assert store.count() == 1
    assert store.list_sources() == ["/tmp/b.pdf"]


def test_add_length_mismatch_raises(
    store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    chunks = [_make_chunk("c1", "x"), _make_chunk("c2", "y")]
    embs = hash_embedder.embed_documents(["x"])
    with pytest.raises(ValueError, match="length mismatch"):
        store.add(chunks, embs)


def test_add_empty_list_is_noop(store: ChromaVectorStore) -> None:
    store.add([], [])
    assert store.count() == 0


def test_persistent_client_roundtrip(
    tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    """Chunks added by one client instance survive in a new one on the same path."""
    path = tmp_path / "chroma"
    s1 = ChromaVectorStore(path=path, collection_name="persist_test")
    s1.add(
        [_make_chunk("p1", "persistent payload")],
        hash_embedder.embed_documents(["persistent payload"]),
    )
    assert s1.count() == 1

    s2 = ChromaVectorStore(path=path, collection_name="persist_test")
    assert s2.count() == 1
    assert s2.list_sources() == ["/tmp/doc.pdf"]
