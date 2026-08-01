"""High-level retriever: query → embed → vector search → ranked `SearchResult`s.

Two modes:
- Dense (default): embed query, run vector store similarity search.
- Hybrid (`hybrid=True`): combine dense results with BM25 sparse results via
  Reciprocal Rank Fusion (`score(d) = Σ 1 / (rrf_k + rank_in_list(d))`).

BM25 is built in-memory on every hybrid call from `vector_store.get_all()`.
Fine for small to medium corpora; a production setup would maintain a persistent
sparse index alongside the dense one.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import SearchResult, VectorStore

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25. Simple and good enough for v1."""
    return _TOKEN_RE.findall(text.lower())


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Local equality-only `where` filter, used when filtering BM25's full-corpus results."""
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


class Retriever:
    """Embeds queries and returns the most relevant chunks from a `VectorStore`.

    With `hybrid=True`, fuses dense and BM25 rankings via RRF.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        hybrid: bool = False,
        rrf_k: int = 60,
    ) -> None:
        """Bind dependencies and ranking knobs."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.hybrid = hybrid
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return ranked results for `query`. Empty / whitespace queries yield []."""
        if not query or not query.strip():
            return []
        k_eff = k or self.top_k
        if self.hybrid:
            return self._hybrid_retrieve(query, k_eff, where)
        embedding = self.embedder.embed_query(query)
        return self.vector_store.search(embedding, k=k_eff, where=where)

    def _hybrid_retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[SearchResult]:
        embedding = self.embedder.embed_query(query)
        # Pull a wider candidate pool from each side so RRF has room to disagree.
        candidate_pool = max(k * 2, k + 5)
        dense_results = self.vector_store.search(embedding, k=candidate_pool, where=where)

        all_chunks = [
            c for c in self.vector_store.get_all() if _matches_where(c.metadata, where)
        ]
        sparse_results: list[SearchResult] = []
        if all_chunks:
            tokenized_corpus = [_tokenize(c.content) for c in all_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(
                zip(all_chunks, scores, strict=True),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )[:candidate_pool]
            sparse_results = [
                SearchResult(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    metadata=c.metadata,
                    score=float(score),
                )
                for c, score in ranked
                if score > 0
            ]

        return self._rrf(dense_results, sparse_results, k)

    def _rrf(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
        k: int,
    ) -> list[SearchResult]:
        fused: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}
        for rank, result in enumerate(dense):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            by_id[result.chunk_id] = result
        for rank, result in enumerate(sparse):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            by_id.setdefault(result.chunk_id, result)

        ranked_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:k]
        return [
            SearchResult(
                chunk_id=cid,
                content=by_id[cid].content,
                metadata=by_id[cid].metadata,
                score=fused[cid],
            )
            for cid in ranked_ids
        ]
