"""Tests for the prompt templates and formatting helpers."""

from __future__ import annotations

from app.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    format_context_block,
    format_query_rewrite_user_turn,
)


def test_rag_system_prompt_mentions_grounding_rule() -> None:
    """The RAG system prompt must instruct the model to answer only from context."""
    assert "ONLY" in RAG_SYSTEM_PROMPT
    assert "I don't know based on the provided context." in RAG_SYSTEM_PROMPT
    assert "<context>" in RAG_SYSTEM_PROMPT


def test_query_rewrite_prompt_demands_single_query() -> None:
    assert "Output ONLY the rewritten query" in QUERY_REWRITE_SYSTEM_PROMPT


def test_format_query_rewrite_user_turn_with_history() -> None:
    history = [("user", "What is RAG?"), ("assistant", "Retrieval-Augmented Generation.")]
    out = format_query_rewrite_user_turn(history, "And what does the R stand for?")
    assert "What is RAG?" in out
    assert "Retrieval-Augmented Generation." in out
    assert out.rstrip().endswith("Latest question: And what does the R stand for?")


def test_format_query_rewrite_user_turn_without_history() -> None:
    out = format_query_rewrite_user_turn([], "First question?")
    assert out == "Latest question: First question?"


def test_format_context_block_empty() -> None:
    out = format_context_block([])
    assert "<context>" in out
    assert "no chunks retrieved" in out


def test_format_context_block_renders_chunks_with_sources() -> None:
    chunks = [
        ("apollo.pdf", "abc123", "Apollo 11 landed on the Moon in 1969."),
        ("voyager.pdf", "def456", "Voyager 1 entered interstellar space in 2012."),
    ]
    out = format_context_block(chunks)
    assert out.startswith("<context>")
    assert out.endswith("</context>")
    assert 'source="apollo.pdf"' in out
    assert 'id="abc123"' in out
    assert "Apollo 11 landed on the Moon in 1969." in out
    assert "Voyager 1 entered interstellar space in 2012." in out


def test_format_context_block_preserves_chunk_order() -> None:
    chunks = [("a.pdf", "1", "first"), ("b.pdf", "2", "second")]
    out = format_context_block(chunks)
    assert out.index("first") < out.index("second")
