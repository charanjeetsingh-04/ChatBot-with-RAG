"""End-to-end chat flow tests.

These wire the real ingestion + retrieval stack against the small PDF fixture,
and use the deterministic `StubLLM` so we can assert on the messages the chat
service sends without making real API calls.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.generation.chat import (
    ChatService,
    Citation,
    InMemorySessionStore,
)
from app.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from tests.conftest import HashEmbedder, StubLLM


@pytest.fixture()
def chat_service(
    hash_embedder: HashEmbedder,
    sample_pdf: Path,
    stub_llm: StubLLM,
) -> ChatService:
    """Build a ChatService backed by the sample PDF + stub LLM."""
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40, top_k=3)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"chat_{uuid.uuid4().hex[:8]}")
    pipeline = IngestionPipeline(vector_store=store, embedder=hash_embedder, settings=settings)
    pipeline.ingest(sample_pdf)
    retriever = Retriever(embedder=hash_embedder, vector_store=store, top_k=settings.top_k)
    sessions = InMemorySessionStore()
    return ChatService(retriever=retriever, llm=stub_llm, sessions=sessions, settings=settings)


def _drain(service: ChatService, session_id: str, message: str) -> str:
    stream = service.stream_reply(session_id, message)
    text = "".join(stream.text_stream())
    return text


def test_empty_user_message_raises(chat_service: ChatService) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        chat_service.stream_reply("s1", "   ")


def test_first_turn_skips_query_rewrite(chat_service: ChatService, stub_llm: StubLLM) -> None:
    """On the first turn there is no history to use, so `complete` must not be called."""
    stream = chat_service.stream_reply("s1", "Who walked on the Moon?")
    list(stream.text_stream())
    assert stream.rewritten_query is None
    assert stub_llm.complete_calls == []


def test_followup_turn_triggers_query_rewrite(
    chat_service: ChatService, stub_llm: StubLLM
) -> None:
    _drain(chat_service, "s1", "Who walked on the Moon during Apollo?")
    stub_llm.complete_calls.clear()
    _drain(chat_service, "s1", "What year?")
    assert len(stub_llm.complete_calls) == 1
    rewrite_call = stub_llm.complete_calls[0]
    assert rewrite_call["system"] == QUERY_REWRITE_SYSTEM_PROMPT


def test_streamed_text_matches_stub_response(chat_service: ChatService) -> None:
    stream = chat_service.stream_reply("s1", "Who walked on the Moon?")
    out = "".join(stream.text_stream())
    assert out == "stub stream response"
    assert stream.final_text() == out


def test_citations_are_extracted_from_retrieved_chunks(chat_service: ChatService) -> None:
    stream = chat_service.stream_reply("s1", "Who walked on the Moon during Apollo?")
    list(stream.text_stream())
    assert stream.citations, "expected non-empty citations"
    assert all(isinstance(c, Citation) for c in stream.citations)
    assert len(stream.citations) == len(stream.retrieved)
    top = stream.citations[0]
    assert top.source_file.endswith(".pdf")
    assert len(top.chunk_id) == 16
    assert 0.0 <= top.score <= 1.0
    assert top.snippet
    # The Apollo page has page=1 in metadata
    assert top.page == 1


def test_llm_call_uses_rag_system_prompt_with_caching(
    chat_service: ChatService, stub_llm: StubLLM
) -> None:
    list(chat_service.stream_reply("s1", "Who walked on the Moon?").text_stream())
    call = stub_llm.stream_calls[0]
    assert call["system"] == RAG_SYSTEM_PROMPT
    assert call["cache_system"] is True


def test_user_turn_contains_context_block_and_question(
    chat_service: ChatService, stub_llm: StubLLM
) -> None:
    list(chat_service.stream_reply("s1", "Who walked on the Moon?").text_stream())
    msgs = stub_llm.stream_calls[0]["messages"]
    assert isinstance(msgs, list) and msgs
    last = msgs[-1]
    assert last.role == "user"
    assert "<context>" in last.content
    assert "</context>" in last.content
    assert "Question: Who walked on the Moon?" in last.content


def test_session_history_stores_literal_message_not_context_block(
    chat_service: ChatService,
) -> None:
    list(chat_service.stream_reply("s1", "Who walked on the Moon?").text_stream())
    session = chat_service.sessions.get("s1")
    assert session is not None
    assert len(session.turns) == 2
    assert session.turns[0].role == "user"
    assert session.turns[0].content == "Who walked on the Moon?"
    assert "<context>" not in session.turns[0].content
    assert session.turns[1].role == "assistant"
    assert session.turns[1].content == "stub stream response"


def test_prior_turns_sent_to_llm_without_context_blocks(
    chat_service: ChatService, stub_llm: StubLLM
) -> None:
    """Only the latest user turn gets a context block; prior turns stay clean."""
    _drain(chat_service, "s1", "Who walked on the Moon during Apollo?")
    stub_llm.stream_calls.clear()
    _drain(chat_service, "s1", "What year did it happen?")

    msgs = stub_llm.stream_calls[0]["messages"]
    assert isinstance(msgs, list)
    # Expect: prior user turn, prior assistant turn, new context-laden user turn
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert "<context>" not in msgs[0].content
    assert msgs[1].role == "assistant"
    assert "<context>" not in msgs[1].content
    assert msgs[2].role == "user"
    assert "<context>" in msgs[2].content


def test_rewritten_query_drives_retrieval(
    hash_embedder: HashEmbedder,
    sample_pdf: Path,
    sample_pdf_page_texts: tuple[str, ...],
) -> None:
    """The rewritten query should be the search input, not the original short question.

    We craft a stub LLM whose `complete` returns a Voyager-focused query so that the
    retriever lands on page 2 even though the user's literal question is "And what?".
    """
    settings = Settings(_env_file=None, chunk_size=300, chunk_overlap=40, top_k=3)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"rwq_{uuid.uuid4().hex[:8]}")
    pipeline = IngestionPipeline(vector_store=store, embedder=hash_embedder, settings=settings)
    pipeline.ingest(sample_pdf)
    retriever = Retriever(embedder=hash_embedder, vector_store=store, top_k=settings.top_k)
    sessions = InMemorySessionStore()
    llm = StubLLM(complete_response="Voyager 1 interstellar Golden Record")
    chat = ChatService(retriever=retriever, llm=llm, sessions=sessions, settings=settings)

    # Seed a prior turn so the rewriter runs on the next call.
    list(chat.stream_reply("s1", "Tell me about a NASA mission.").text_stream())
    llm.complete_calls.clear()
    llm.stream_calls.clear()

    stream = chat.stream_reply("s1", "And what about that one?")
    list(stream.text_stream())

    assert stream.rewritten_query == "Voyager 1 interstellar Golden Record"
    assert stream.citations[0].page == 2
    assert "Voyager" in stream.retrieved[0].content


def test_zero_retrieval_results_still_streams_response(
    hash_embedder: HashEmbedder,
    stub_llm: StubLLM,
) -> None:
    """An empty corpus should still produce a stream — the model is told to say it doesn't know."""
    settings = Settings(_env_file=None, top_k=3)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"empty_{uuid.uuid4().hex[:8]}")
    retriever = Retriever(embedder=hash_embedder, vector_store=store, top_k=settings.top_k)
    chat = ChatService(
        retriever=retriever, llm=stub_llm, sessions=InMemorySessionStore(), settings=settings
    )
    stream = chat.stream_reply("s1", "Anything?")
    out = "".join(stream.text_stream())
    assert out == "stub stream response"
    assert stream.citations == []
    assert "no chunks retrieved" in stub_llm.stream_calls[0]["messages"][-1].content


def test_citation_snippet_is_truncated_for_long_chunks(
    hash_embedder: HashEmbedder,
    stub_llm: StubLLM,
) -> None:
    """Long chunk content should be truncated in the citation snippet."""
    from app.ingestion.chunker import Chunk

    settings = Settings(_env_file=None, top_k=1)  # type: ignore[call-arg]
    store = ChromaVectorStore(path=None, collection_name=f"trunc_{uuid.uuid4().hex[:8]}")
    long_text = "word " * 200  # ~1000 chars
    chunk = Chunk(
        chunk_id="abc1234567890def",
        content=long_text.strip(),
        metadata={"source_file": "long.pdf", "page": 1, "chunk_index": 0},
    )
    embs = hash_embedder.embed_documents([chunk.content])
    store.add([chunk], embs)
    retriever = Retriever(embedder=hash_embedder, vector_store=store, top_k=settings.top_k)
    chat = ChatService(
        retriever=retriever, llm=stub_llm, sessions=InMemorySessionStore(), settings=settings
    )

    stream = chat.stream_reply("s1", "anything word")
    list(stream.text_stream())
    assert stream.citations, "expected at least one citation"
    snippet = stream.citations[0].snippet
    assert len(snippet) < len(long_text)
    assert snippet.endswith("...")
