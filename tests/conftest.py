"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fpdf import FPDF

from app.embeddings.embedder import Embedder
from app.generation.llm import LLM, ChatMessage

PAGE_TEXTS: tuple[str, ...] = (
    "Apollo 11 was the spaceflight that first landed humans on the Moon in July 1969. "
    "Commander Neil Armstrong and pilot Buzz Aldrin landed the lunar module Eagle on "
    "the Sea of Tranquility, while Michael Collins orbited above in Columbia.",
    "The Voyager program launched two robotic probes in 1977 to study the outer Solar "
    "System. Voyager 1 became the first spacecraft to enter interstellar space in 2012, "
    "carrying the Golden Record of sounds and images from Earth.",
    "James Webb Space Telescope launched on December 25 2021 and operates near the "
    "Sun Earth L2 Lagrange point. It observes primarily in infrared, complementing "
    "Hubbles ultraviolet and visible-light coverage.",
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a 3-page PDF with known content for ingestion tests."""
    out_dir = tmp_path_factory.mktemp("pdf_fixtures")
    pdf_path = out_dir / "sample.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for text in PAGE_TEXTS:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(w=0, h=8, text=text)
    pdf.output(str(pdf_path))
    return pdf_path


@pytest.fixture(scope="session")
def sample_pdf_page_texts() -> tuple[str, ...]:
    """The plain-text content used to build the sample PDF, one entry per page."""
    return PAGE_TEXTS


class HashEmbedder(Embedder):
    """Deterministic test-only embedder.

    Maps each whitespace-separated token to a fixed bucket via SHA-256, builds a
    bag-of-words vector, and L2-normalizes. Texts sharing distinctive words land
    close in cosine space — enough to verify retrieval logic without loading BGE.
    """

    def __init__(self, dim: int = 128) -> None:
        """Choose the output dimension. Higher dims reduce hash collisions."""
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self._dim = dim

    @property
    def dimension(self) -> int:
        """Return the configured output dimensionality."""
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "big") % self._dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode each input text into a normalized hashed bag-of-words vector."""
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query the same way as documents (symmetric)."""
        return self._embed_one(text)


@pytest.fixture()
def hash_embedder() -> HashEmbedder:
    """Fresh deterministic embedder per test."""
    return HashEmbedder()


class StubLLM(LLM):
    """Deterministic LLM stub for tests.

    Records every `stream` and `complete` call so tests can assert on the
    messages, system prompt, and cache flag the orchestrator sent.
    """

    def __init__(
        self,
        stream_response: str = "stub stream response",
        complete_response: str = "stub complete response",
    ) -> None:
        """Bind canned responses."""
        self._stream_response = stream_response
        self._complete_response = complete_response
        self.stream_calls: list[dict[str, object]] = []
        self.complete_calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        """Return the stub model id."""
        return "stub-llm"

    def stream(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Record the call and yield the canned response, one word per delta."""
        self.stream_calls.append(
            {
                "messages": list(messages),
                "system": system,
                "max_tokens": max_tokens,
                "cache_system": cache_system,
            }
        )
        tokens = self._stream_response.split()
        for i, tok in enumerate(tokens):
            yield tok + (" " if i < len(tokens) - 1 else "")

    def complete(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Record the call and return the canned response."""
        self.complete_calls.append(
            {
                "messages": list(messages),
                "system": system,
                "max_tokens": max_tokens,
            }
        )
        return self._complete_response


@pytest.fixture()
def stub_llm() -> StubLLM:
    """Fresh stub LLM per test."""
    return StubLLM()
