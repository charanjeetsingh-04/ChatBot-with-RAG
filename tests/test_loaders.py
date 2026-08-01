"""Tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders import (
    LOADER_REGISTRY,
    BaseLoader,
    Document,
    PDFLoader,
    get_loader,
)


def test_get_loader_returns_pdf_loader_for_pdf_extension(tmp_path: Path) -> None:
    loader = get_loader(tmp_path / "anything.pdf")
    assert isinstance(loader, PDFLoader)


def test_get_loader_case_insensitive(tmp_path: Path) -> None:
    loader = get_loader(tmp_path / "anything.PDF")
    assert isinstance(loader, PDFLoader)


def test_get_loader_raises_for_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No loader registered"):
        get_loader(tmp_path / "anything.xyz")


def test_pdf_loader_returns_one_doc_per_page(sample_pdf: Path) -> None:
    docs = PDFLoader().load(sample_pdf)
    assert len(docs) == 3
    for i, doc in enumerate(docs, start=1):
        assert isinstance(doc, Document)
        # source_file is normalised to POSIX so URL paths work cross-platform.
        assert doc.metadata["source_file"] == sample_pdf.as_posix()
        assert "\\" not in doc.metadata["source_file"]
        assert doc.metadata["page"] == i


def test_pdf_loader_extracts_text_per_page(
    sample_pdf: Path, sample_pdf_page_texts: tuple[str, ...]
) -> None:
    docs = PDFLoader().load(sample_pdf)
    # PDF text extraction can re-flow whitespace and break long lines, so we
    # check that distinctive tokens from each page survive the round-trip.
    distinctive = ("Apollo", "Voyager", "Webb")
    for doc, token in zip(docs, distinctive, strict=True):
        assert token in doc.content


def test_base_loader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLoader()  # type: ignore[abstract]


def test_registry_contains_pdf() -> None:
    assert ".pdf" in LOADER_REGISTRY
