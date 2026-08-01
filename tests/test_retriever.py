"""End-to-end retrieval tests: ingest → embed → store → query.

The spec's required test: "a known chunk is retrieved for a known query."
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from tests.conftest import HashEmbedder


def _build(hash_embedder: HashEmbedder, sample_pdf: Path) -> Retriever:
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40, top_k=3)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"retriever_{uuid.uuid4().hex[:8]}")
    pipeline = IngestionPipeline(vector_store=store, embedder=hash_embedder, settings=settings)
    pipeline.ingest(sample_pdf)
    return Retriever(embedder=hash_embedder, vector_store=store, top_k=settings.top_k)


def test_known_query_retrieves_known_chunk(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    """The Apollo query must surface the Apollo page (page 1) as the top hit."""
    retriever = _build(hash_embedder, sample_pdf)
    results = retriever.retrieve("Apollo 11 Armstrong Aldrin Moon landing")
    assert results, "expected at least one result"
    assert results[0].metadata["page"] == 1
    assert "Apollo" in results[0].content


def test_voyager_query_retrieves_voyager_chunk(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    retriever = _build(hash_embedder, sample_pdf)
    results = retriever.retrieve("Voyager interstellar Golden Record")
    assert results[0].metadata["page"] == 2
    assert "Voyager" in results[0].content


def test_jwst_query_retrieves_jwst_chunk(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    retriever = _build(hash_embedder, sample_pdf)
    results = retriever.retrieve("James Webb infrared Lagrange L2")
    assert results[0].metadata["page"] == 3
    assert "Webb" in results[0].content


def test_empty_query_returns_no_results(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    retriever = _build(hash_embedder, sample_pdf)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_respects_explicit_k(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    retriever = _build(hash_embedder, sample_pdf)
    results = retriever.retrieve("Apollo", k=1)
    assert len(results) == 1


def test_retriever_where_filter(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    """Metadata `where` filter restricts results to a single page."""
    retriever = _build(hash_embedder, sample_pdf)
    results = retriever.retrieve("space", where={"page": 2})
    assert results, "expected results when filtering to page 2"
    assert all(r.metadata["page"] == 2 for r in results)


def test_ingestion_pipeline_reports_chunk_count(
    hash_embedder: HashEmbedder, sample_pdf: Path
) -> None:
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"count_{uuid.uuid4().hex[:8]}")
    pipeline = IngestionPipeline(vector_store=store, embedder=hash_embedder, settings=settings)
    result = pipeline.ingest(sample_pdf)
    assert result.file_path == sample_pdf
    assert result.chunks_added > 0
    assert result.chunks_added == store.count()
