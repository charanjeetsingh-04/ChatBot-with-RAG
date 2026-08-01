"""Tests for hybrid retrieval (BM25 + dense via RRF) and `VectorStore.get_all`."""

from __future__ import annotations

import uuid

import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.retriever import Retriever, _tokenize
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
def populated_store(hash_embedder: HashEmbedder) -> ChromaVectorStore:
    store = ChromaVectorStore(path=None, collection_name=f"hyb_{uuid.uuid4().hex[:8]}")
    chunks = [
        _make_chunk("apollo", "Apollo 11 Armstrong Aldrin Moon landing 1969"),
        _make_chunk("voyager", "Voyager 1 Voyager 2 interstellar Golden Record"),
        _make_chunk("jwst", "James Webb Space Telescope infrared L2"),
    ]
    store.add(chunks, hash_embedder.embed_documents([c.content for c in chunks]))
    return store


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


def test_get_all_returns_every_chunk(populated_store: ChromaVectorStore) -> None:
    out = populated_store.get_all()
    assert len(out) == 3
    ids = {r.chunk_id for r in out}
    assert ids == {"apollo", "voyager", "jwst"}
    for r in out:
        assert isinstance(r, SearchResult)
        assert r.score == 0.0
        assert "source_file" in r.metadata


def test_get_all_empty_store() -> None:
    empty = ChromaVectorStore(path=None, collection_name=f"empty_{uuid.uuid4().hex[:8]}")
    assert empty.get_all() == []


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_word() -> None:
    assert _tokenize("Apollo 11, Moon-landing!") == ["apollo", "11", "moon", "landing"]


def test_tokenize_handles_unicode() -> None:
    assert _tokenize("Café résumé") == ["café", "résumé"]


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


def test_hybrid_retrieves_top_chunk_for_matching_query(
    populated_store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    retriever = Retriever(
        embedder=hash_embedder, vector_store=populated_store, top_k=3, hybrid=True
    )
    results = retriever.retrieve("who walked on the Moon during Apollo")
    assert results[0].chunk_id == "apollo"


def test_hybrid_returns_at_most_k(
    populated_store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    retriever = Retriever(
        embedder=hash_embedder, vector_store=populated_store, top_k=2, hybrid=True
    )
    results = retriever.retrieve("Apollo")
    assert len(results) == 2


def test_hybrid_empty_query_returns_empty(
    populated_store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    retriever = Retriever(
        embedder=hash_embedder, vector_store=populated_store, top_k=3, hybrid=True
    )
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_hybrid_against_empty_store(hash_embedder: HashEmbedder) -> None:
    """No chunks at all → no results, no crash."""
    store = ChromaVectorStore(path=None, collection_name=f"none_{uuid.uuid4().hex[:8]}")
    retriever = Retriever(embedder=hash_embedder, vector_store=store, top_k=3, hybrid=True)
    assert retriever.retrieve("Apollo") == []


def test_hybrid_rescues_exact_match_dense_misses(hash_embedder: HashEmbedder) -> None:
    """Where dense embeddings disagree with the keyword overlap, BM25 still surfaces
    the exact-keyword chunk via RRF."""
    store = ChromaVectorStore(path=None, collection_name=f"rescue_{uuid.uuid4().hex[:8]}")
    # Distractors share many bag-of-words bins with the query but no actual term overlap;
    # the rare-term chunk wins on BM25 because the query token appears only there.
    chunks = [
        _make_chunk("rare", "tardigrade survives extreme conditions in space"),
        _make_chunk("d1", "the quick brown fox jumps over the lazy dog repeatedly here"),
        _make_chunk("d2", "lorem ipsum dolor sit amet consectetur adipiscing elit doloremque"),
        _make_chunk("d3", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"),
    ]
    store.add(chunks, hash_embedder.embed_documents([c.content for c in chunks]))

    retriever = Retriever(
        embedder=hash_embedder, vector_store=store, top_k=4, hybrid=True, rrf_k=60
    )
    results = retriever.retrieve("tardigrade")
    assert results[0].chunk_id == "rare"


def test_hybrid_scores_are_rrf_fused(
    populated_store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    """RRF scores are ~1/(rrf_k+rank+1) summed across the two lists.

    With rrf_k=60 and a chunk that's #1 in both lists, the fused score is at most
    2/61 ≈ 0.0328. That's the upper bound regardless of how strongly the chunk matched.
    """
    retriever = Retriever(
        embedder=hash_embedder, vector_store=populated_store, top_k=3, hybrid=True, rrf_k=60
    )
    results = retriever.retrieve("Apollo Armstrong Moon")
    assert results[0].score <= 2.0 / 61.0 + 1e-9
    # Scores should be non-increasing.
    for a, b in zip(results, results[1:], strict=False):
        assert a.score >= b.score


def test_dense_only_still_works_when_hybrid_disabled(
    populated_store: ChromaVectorStore, hash_embedder: HashEmbedder
) -> None:
    """Sanity check the default code path is unchanged."""
    retriever = Retriever(
        embedder=hash_embedder, vector_store=populated_store, top_k=3, hybrid=False
    )
    results = retriever.retrieve("Apollo Armstrong Moon")
    assert results[0].chunk_id == "apollo"
    # Dense scores are normalised cosine similarities in [0, 1], not RRF.
    assert results[0].score > 0.0
    assert results[0].score <= 1.0


def test_hybrid_where_filter_scopes_both_branches(hash_embedder: HashEmbedder) -> None:
    """`where` should filter both the dense and the BM25 candidate pools."""
    store = ChromaVectorStore(path=None, collection_name=f"where_{uuid.uuid4().hex[:8]}")
    chunks = [
        _make_chunk("a-apollo", "Apollo 11 Armstrong Moon", source_file="a.pdf", chunk_index=0),
        _make_chunk("b-apollo", "Apollo 11 Armstrong Moon", source_file="b.pdf", chunk_index=0),
    ]
    store.add(chunks, hash_embedder.embed_documents([c.content for c in chunks]))

    retriever = Retriever(
        embedder=hash_embedder, vector_store=store, top_k=2, hybrid=True
    )
    results = retriever.retrieve("Apollo Moon", where={"source_file": "a.pdf"})
    assert results, "expected matches under the filter"
    assert all(r.metadata["source_file"] == "a.pdf" for r in results)
