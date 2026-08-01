"""Tests for the recursive character chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    Chunk,
    RecursiveCharacterChunker,
    _chunk_id,
)
from app.ingestion.loaders import Document


def test_empty_text_returns_no_chunks() -> None:
    c = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert c.split_text("") == []
    assert c.split_text("   \n   ") == []


def test_short_text_returns_single_chunk() -> None:
    c = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    out = c.split_text("Hello world.")
    assert out == ["Hello world."]


def test_long_text_splits_into_multiple_chunks() -> None:
    c = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=20)
    # Five paragraphs that together exceed chunk_size.
    text = "\n\n".join(
        f"Paragraph {i} contains some words to make the chunker do real work here."
        for i in range(5)
    )
    chunks = c.split_text(text)
    assert len(chunks) > 1
    for ch in chunks:
        assert len(ch) <= c.chunk_size


def test_overlap_is_respected_between_adjacent_chunks() -> None:
    c = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=20)
    text = " ".join(f"word{i:03d}" for i in range(50))
    chunks = c.split_text(text)
    assert len(chunks) >= 2
    # The tail of chunk i should appear at the head of chunk i+1 (greedy overlap).
    for a, b in zip(chunks, chunks[1:], strict=False):
        a_tail = a[-c.chunk_overlap :]
        # At least one whole word from the overlap window should re-appear at the start.
        overlap_words = [w for w in a_tail.split() if w.startswith("word")]
        assert overlap_words, f"no overlap candidate words in tail: {a_tail!r}"
        assert any(w in b[: c.chunk_overlap + 10] for w in overlap_words)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=100)


def test_invalid_sizes_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveCharacterChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-1)


def test_chunk_document_preserves_and_extends_metadata() -> None:
    c = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    doc = Document(
        content="The quick brown fox jumps over the lazy dog. " * 5,
        metadata={"source_file": "/tmp/foo.pdf", "page": 3},
    )
    chunks = c.chunk_document(doc)
    assert chunks, "expected at least one chunk"
    for i, ch in enumerate(chunks):
        assert isinstance(ch, Chunk)
        assert ch.metadata["source_file"] == "/tmp/foo.pdf"
        assert ch.metadata["page"] == 3
        assert ch.metadata["chunk_index"] == i
        assert len(ch.chunk_id) == 16


def test_chunk_documents_uses_continuous_index() -> None:
    c = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    docs = [
        Document(content="alpha " * 30, metadata={"source_file": "f.pdf", "page": 1}),
        Document(content="beta " * 30, metadata={"source_file": "f.pdf", "page": 2}),
    ]
    chunks = c.chunk_documents(docs)
    indices = [ch.metadata["chunk_index"] for ch in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_ids_are_deterministic() -> None:
    a = _chunk_id("/tmp/x.pdf", 1, "hello world")
    b = _chunk_id("/tmp/x.pdf", 1, "hello world")
    c = _chunk_id("/tmp/x.pdf", 2, "hello world")
    assert a == b
    assert a != c


def test_separator_hierarchy_prefers_paragraph_breaks() -> None:
    c = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=0)
    text = "Para one is short.\n\nPara two is also fairly short."
    out = c.split_text(text)
    # Both paragraphs fit individually under chunk_size; the splitter should
    # produce one chunk containing the full text since it's still under 80 chars.
    if len(text) <= 80:
        assert out == [text]
    else:
        assert all(len(ch) <= 80 for ch in out)


def test_hard_split_for_very_long_token() -> None:
    c = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=0)
    # A single word longer than chunk_size must be hard-split at chunk_size.
    long_token = "x" * 75
    out = c.split_text(long_token)
    assert all(len(ch) <= 20 for ch in out)
    assert "".join(out) == long_token
