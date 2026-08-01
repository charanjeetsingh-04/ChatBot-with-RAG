"""Tests for dedupe-on-reingest behavior in `IngestionPipeline.ingest`."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_store import ChromaVectorStore
from tests.conftest import HashEmbedder


def _build_pipeline(hash_embedder: HashEmbedder) -> tuple[IngestionPipeline, ChromaVectorStore]:
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"dedupe_{uuid.uuid4().hex[:8]}")
    return IngestionPipeline(vector_store=store, embedder=hash_embedder, settings=settings), store


def test_first_ingest_reports_zero_replaced(
    sample_pdf: Path, hash_embedder: HashEmbedder
) -> None:
    pipeline, store = _build_pipeline(hash_embedder)
    result = pipeline.ingest(sample_pdf)
    assert result.chunks_added > 0
    assert result.chunks_replaced == 0
    assert store.count() == result.chunks_added


def test_reingesting_same_file_is_idempotent(
    sample_pdf: Path, hash_embedder: HashEmbedder
) -> None:
    pipeline, store = _build_pipeline(hash_embedder)
    first = pipeline.ingest(sample_pdf)
    assert first.chunks_replaced == 0

    second = pipeline.ingest(sample_pdf)
    assert second.chunks_added == first.chunks_added
    assert second.chunks_replaced == first.chunks_added
    assert store.count() == first.chunks_added


def test_reingesting_modified_file_replaces_old_chunks(
    tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    """Mutating a file's content at the same path should leave no stale chunks behind."""
    pipeline, store = _build_pipeline(hash_embedder)
    notes = tmp_path / "notes.txt"

    notes.write_text("Apollo 11 landed on the Moon in 1969.", encoding="utf-8")
    first = pipeline.ingest(notes)
    apollo_count = first.chunks_added
    assert apollo_count > 0
    assert store.count() == apollo_count

    notes.write_text(
        "Voyager 1 entered interstellar space in 2012 carrying the Golden Record.",
        encoding="utf-8",
    )
    second = pipeline.ingest(notes)
    assert second.chunks_replaced == apollo_count

    # Verify the new content survived and the old content is gone.
    all_chunks = store.get_all()
    joined = " ".join(c.content for c in all_chunks)
    assert "Voyager" in joined
    assert "Apollo" not in joined


def test_replace_false_disables_dedupe(
    tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    """With `replace=False`, identical content still upserts (no growth), but stale
    chunks from a prior version would remain. Here we just confirm upsert is
    idempotent on identical content."""
    pipeline, store = _build_pipeline(hash_embedder)
    notes = tmp_path / "notes.txt"
    notes.write_text("Apollo 11 landed on the Moon in 1969.", encoding="utf-8")

    pipeline.ingest(notes, replace=False)
    initial_count = store.count()
    pipeline.ingest(notes, replace=False)
    # Upsert keys are deterministic chunk_ids, so the count must not double.
    assert store.count() == initial_count
