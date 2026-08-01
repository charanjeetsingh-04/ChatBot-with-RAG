"""End-to-end ingestion pipeline tests (load + chunk)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.pipeline import ingest_file


def test_ingest_pdf_produces_chunks_with_metadata(sample_pdf: Path) -> None:
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40)  # type: ignore[call-arg]
    chunks = ingest_file(sample_pdf, settings=settings)

    assert chunks, "expected at least one chunk from the sample PDF"
    pages_seen = {ch.metadata["page"] for ch in chunks}
    assert pages_seen == {1, 2, 3}

    for ch in chunks:
        assert ch.metadata["source_file"] == sample_pdf.as_posix()
        assert ch.metadata["chunk_index"] == int(ch.metadata["chunk_index"])
        assert len(ch.chunk_id) == 16
        assert len(ch.content) <= settings.chunk_size


def test_ingest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_file(tmp_path / "does_not_exist.pdf")


def test_ingest_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "nope.xyz"
    p.write_text("hi")
    with pytest.raises(ValueError, match="No loader registered"):
        ingest_file(p)


def test_ingest_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        ingest_file(tmp_path)
