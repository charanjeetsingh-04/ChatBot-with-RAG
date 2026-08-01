"""Tests for the Phase 8 loaders: TextLoader, MarkdownLoader, DOCXLoader."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as DocxDocument

from app.ingestion.loaders import (
    LOADER_REGISTRY,
    DOCXLoader,
    MarkdownLoader,
    TextLoader,
    get_loader,
)

# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ext", "expected"),
    [
        (".txt", TextLoader),
        (".md", MarkdownLoader),
        (".markdown", MarkdownLoader),
        (".docx", DOCXLoader),
        (".TXT", TextLoader),
        (".MD", MarkdownLoader),
        (".DOCX", DOCXLoader),
    ],
)
def test_registry_dispatches_new_extensions(
    tmp_path: Path, ext: str, expected: type[object]
) -> None:
    loader = get_loader(tmp_path / f"file{ext}")
    assert isinstance(loader, expected)  # type: ignore[arg-type]


def test_registry_contains_all_new_types() -> None:
    for ext in (".txt", ".md", ".markdown", ".docx"):
        assert ext in LOADER_REGISTRY


# ---------------------------------------------------------------------------
# TextLoader
# ---------------------------------------------------------------------------


def test_text_loader_returns_single_document(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    docs = TextLoader().load(p)
    assert len(docs) == 1
    assert docs[0].content == "hello world"
    assert docs[0].metadata["source_file"] == p.as_posix()
    assert "page" not in docs[0].metadata


def test_text_loader_strips_outer_whitespace(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("\n\n  body  \n\n", encoding="utf-8")
    docs = TextLoader().load(p)
    assert docs[0].content == "body"


def test_text_loader_handles_unicode(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("héllo — wörld 🌍", encoding="utf-8")
    docs = TextLoader().load(p)
    assert "héllo" in docs[0].content
    assert "🌍" in docs[0].content


# ---------------------------------------------------------------------------
# MarkdownLoader
# ---------------------------------------------------------------------------


def test_markdown_loader_preserves_raw_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nParagraph with **bold**.\n", encoding="utf-8")
    docs = MarkdownLoader().load(p)
    assert len(docs) == 1
    assert "# Title" in docs[0].content
    assert "**bold**" in docs[0].content
    assert docs[0].metadata["source_file"] == p.as_posix()


def test_markdown_loader_dispatches_both_extensions(tmp_path: Path) -> None:
    md = tmp_path / "a.md"
    md.write_text("md content", encoding="utf-8")
    markdown = tmp_path / "b.markdown"
    markdown.write_text("markdown content", encoding="utf-8")

    assert MarkdownLoader().load(md)[0].content == "md content"
    assert MarkdownLoader().load(markdown)[0].content == "markdown content"


# ---------------------------------------------------------------------------
# DOCXLoader
# ---------------------------------------------------------------------------


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    doc = DocxDocument()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(str(path))


def test_docx_loader_extracts_paragraphs(tmp_path: Path) -> None:
    p = tmp_path / "doc.docx"
    _make_docx(p, ["First paragraph.", "Second paragraph."])

    docs = DOCXLoader().load(p)
    assert len(docs) == 1
    assert "First paragraph." in docs[0].content
    assert "Second paragraph." in docs[0].content
    assert docs[0].metadata["source_file"] == p.as_posix()


def test_docx_loader_skips_empty_paragraphs(tmp_path: Path) -> None:
    p = tmp_path / "doc.docx"
    _make_docx(p, ["Real text.", "", "   ", "More text."])

    docs = DOCXLoader().load(p)
    # Paragraphs are joined with \n\n; the empty ones should be skipped.
    assert docs[0].content == "Real text.\n\nMore text."


# ---------------------------------------------------------------------------
# Pipeline integration smoke (using HashEmbedder + ChromaVectorStore)
# ---------------------------------------------------------------------------


def test_ingest_pipeline_handles_text_md_and_docx(
    tmp_path: Path, hash_embedder: object
) -> None:
    import uuid

    from app.config import Settings
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.vector_store import ChromaVectorStore
    from tests.conftest import HashEmbedder

    assert isinstance(hash_embedder, HashEmbedder)

    txt = tmp_path / "notes.txt"
    txt.write_text("Apollo 11 landed on the Moon in 1969.", encoding="utf-8")
    md = tmp_path / "voyager.md"
    md.write_text("# Voyager\n\nVoyager 1 reached interstellar space in 2012.", encoding="utf-8")
    docx_path = tmp_path / "jwst.docx"
    _make_docx(
        docx_path,
        ["James Webb Space Telescope launched on December 25 2021."],
    )

    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"mixed_{uuid.uuid4().hex[:8]}")
    pipeline = IngestionPipeline(
        vector_store=store, embedder=hash_embedder, settings=settings
    )
    for path in (txt, md, docx_path):
        result = pipeline.ingest(path)
        assert result.chunks_added > 0

    sources = store.list_sources()
    assert {txt.as_posix(), md.as_posix(), docx_path.as_posix()} <= set(sources)
