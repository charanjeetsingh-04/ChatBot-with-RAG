"""Tests for the embedder interface and the test HashEmbedder."""

from __future__ import annotations

import math

import pytest

from app.embeddings.embedder import Embedder, make_embedder
from tests.conftest import HashEmbedder


def test_hash_embedder_dimension_matches_vectors(hash_embedder: HashEmbedder) -> None:
    vec = hash_embedder.embed_query("hello world")
    assert len(vec) == hash_embedder.dimension


def test_hash_embedder_normalizes_vectors(hash_embedder: HashEmbedder) -> None:
    vec = hash_embedder.embed_query("apollo armstrong moon eagle")
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, abs_tol=1e-6)


def test_hash_embedder_empty_text_returns_zero_vector(hash_embedder: HashEmbedder) -> None:
    vec = hash_embedder.embed_query("")
    assert len(vec) == hash_embedder.dimension
    assert all(x == 0.0 for x in vec)


def test_hash_embedder_batch_matches_single(hash_embedder: HashEmbedder) -> None:
    texts = ["alpha beta", "gamma delta", "alpha gamma"]
    batch = hash_embedder.embed_documents(texts)
    singles = [hash_embedder.embed_query(t) for t in texts]
    assert batch == singles


def test_hash_embedder_similar_text_closer_than_unrelated(hash_embedder: HashEmbedder) -> None:
    apollo = hash_embedder.embed_query("Apollo 11 Armstrong Aldrin Moon")
    apollo_q = hash_embedder.embed_query("who landed on the Moon during Apollo")
    voyager = hash_embedder.embed_query("Voyager interstellar Golden Record")

    sim_match = sum(a * b for a, b in zip(apollo, apollo_q, strict=False))
    sim_off = sum(a * b for a, b in zip(apollo, voyager, strict=False))
    assert sim_match > sim_off


def test_embedder_is_abstract() -> None:
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


def test_make_embedder_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        make_embedder("not-a-real-provider", "whatever")


def test_make_embedder_openai_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        make_embedder("openai", "text-embedding-3-small")


def test_hash_embedder_invalid_dim_rejected() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashEmbedder(dim=0)
