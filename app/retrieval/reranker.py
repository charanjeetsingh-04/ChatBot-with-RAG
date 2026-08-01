"""Reranker interface (stub).

A cross-encoder reranker — re-scoring retrieved chunks against the query using
a more expensive model — is deferred to a later phase. The interface is here so
the chat pipeline can plumb it through without churn when it lands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.retrieval.vector_store import SearchResult


class Reranker(ABC):
    """Reorders a list of `SearchResult`s by relevance to a query."""

    @abstractmethod
    def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Return `results` reordered (and optionally trimmed) by re-scored relevance."""
