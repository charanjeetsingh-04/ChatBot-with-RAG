"""Embedder interface and the default sentence-transformers implementation.

Add a new provider by subclassing `Embedder` and wiring it into `make_embedder`.
Query vs document encoding is separated so asymmetric models can apply different
preprocessing (e.g. BGE instruction prefixes) without changing call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    """Encodes text into dense float vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions in each output vector."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of documents/chunks. Returns one vector per input."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string."""


class SentenceTransformerEmbedder(Embedder):
    """Local embedder backed by the `sentence-transformers` library.

    Vectors are L2-normalized so cosine similarity == dot product, which is what
    ChromaDB (configured with `hnsw:space=cosine`) expects for relevance scoring.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Load the model. First-time call downloads weights from Hugging Face."""
        from sentence_transformers import SentenceTransformer  # heavy import — lazy

        self._model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the model's output dimensionality."""
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Model {self._model_name!r} reported no embedding dimension")
        return int(dim)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts into normalized vectors."""
        embs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in embs]

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query into a normalized vector."""
        emb = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(x) for x in emb]


def make_embedder(provider: str, model: str) -> Embedder:
    """Construct an Embedder for the given provider and model id.

    Recognised providers: `sentence-transformers`. `openai` is reserved for a
    later phase and currently raises NotImplementedError.
    """
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model)
    if provider == "openai":
        raise NotImplementedError("OpenAI embeddings impl is not wired yet.")
    raise ValueError(f"Unknown embedding provider: {provider!r}")
