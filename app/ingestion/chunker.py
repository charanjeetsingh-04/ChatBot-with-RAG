"""Recursive character text splitter.

Algorithm: split the input on the first separator (from a hierarchy:
paragraph break, line break, sentence, word, char) that appears in the text.
Recurse into any resulting piece that still exceeds `chunk_size`. Finally,
greedily merge adjacent leaf pieces back together up to `chunk_size`, carrying
`chunk_overlap` characters of trailing context into each subsequent chunk.

This mirrors the well-known LangChain `RecursiveCharacterTextSplitter` behavior
without taking the dependency.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.ingestion.loaders import Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Chunk:
    """A chunk of text with provenance metadata.

    `chunk_id` is a deterministic content-derived hash, stable across re-ingest
    of identical content from the same source — used by the Phase 8 dedupe path.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any]


def _chunk_id(source_file: str, page: int | None, content: str) -> str:
    """Return a deterministic 16-char hex id derived from source + page + content."""
    h = hashlib.sha256()
    h.update(source_file.encode("utf-8"))
    h.update(b"|")
    h.update(str(page if page is not None else "").encode("utf-8"))
    h.update(b"|")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:16]


class RecursiveCharacterChunker:
    """Recursive character splitter with overlapping merge."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        """Configure the chunker.

        Raises `ValueError` if `chunk_overlap >= chunk_size`.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def split_text(self, text: str) -> list[str]:
        """Split `text` into chunks of at most `chunk_size` characters."""
        if not text:
            return []
        text = text.strip()
        if not text:
            return []
        leaves = self._recursive_split(text, self.separators)
        return self._merge(leaves)

    def chunk_document(self, doc: Document, start_index: int = 0) -> list[Chunk]:
        """Split one `Document` into chunks, preserving and extending its metadata."""
        pieces = self.split_text(doc.content)
        source_file = str(doc.metadata.get("source_file", ""))
        page = doc.metadata.get("page")
        page_int = int(page) if isinstance(page, int) else None
        chunks: list[Chunk] = []
        for offset, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            idx = start_index + offset
            meta["chunk_index"] = idx
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_file, page_int, piece),
                    content=piece,
                    metadata=meta,
                )
            )
        return chunks

    def chunk_documents(self, docs: Sequence[Document]) -> list[Chunk]:
        """Split a sequence of Documents into a flat list of Chunks.

        `chunk_index` is global within the call (continues across Documents) so
        consumers can reason about ordering of chunks from the same ingestion.
        """
        out: list[Chunk] = []
        running = 0
        for doc in docs:
            new = self.chunk_document(doc, start_index=running)
            out.extend(new)
            running += len(new)
        return out

    def _recursive_split(self, text: str, separators: Sequence[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator = separators[-1] if separators else ""
        remaining: Sequence[str] = ()
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                remaining = ()
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break

        if separator == "":
            return [text[j : j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]

        splits = text.split(separator)
        result: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                result.append(s)
            else:
                result.extend(self._recursive_split(s, remaining or (separator,)))
        return result

    def _merge(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        sep = " "
        sep_len = len(sep)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            add_len = piece_len + (sep_len if current else 0)
            if current and current_len + add_len > self.chunk_size:
                chunks.append(sep.join(current))
                overlap = self._build_overlap(current, sep_len)
                current = list(overlap)
                current_len = sum(len(p) for p in current) + max(0, len(current) - 1) * sep_len
                add_len = piece_len + (sep_len if current else 0)
            current.append(piece)
            current_len += add_len
        if current:
            chunks.append(sep.join(current))
        return chunks

    def _build_overlap(self, pieces: list[str], sep_len: int) -> list[str]:
        if self.chunk_overlap <= 0:
            return []
        overlap: list[str] = []
        total = 0
        for piece in reversed(pieces):
            extra = len(piece) + (sep_len if overlap else 0)
            if total + extra > self.chunk_overlap:
                break
            overlap.insert(0, piece)
            total += extra
        return overlap
