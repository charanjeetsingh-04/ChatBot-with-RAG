"""Smoke tests for the configuration module."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


def test_defaults_load_without_env() -> None:
    """Settings should instantiate with built-in defaults when no env vars are set."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider == "anthropic"
    assert s.llm_model.startswith("claude-")
    assert s.embedding_provider == "sentence-transformers"
    assert s.top_k == 5
    assert s.chunk_overlap < s.chunk_size


def test_overlap_must_be_smaller_than_size() -> None:
    """chunk_overlap >= chunk_size must raise a validation error."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    """get_settings() returns the same singleton instance on repeat calls."""
    assert get_settings() is get_settings()
