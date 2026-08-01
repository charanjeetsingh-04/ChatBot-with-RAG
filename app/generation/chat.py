"""Chat orchestration: session memory + query rewriting + retrieval + generation.

Flow for one user message:

1. Look up (or create) the session.
2. If prior turns exist, rewrite the user's question into a standalone search query
   using the conversation history (non-streaming LLM call).
3. Retrieve top-k chunks for the standalone query.
4. Build the LLM message list: prior turns verbatim, plus a final user turn that
   prepends the `<context>` block to the literal question. Prior turns deliberately
   do NOT carry their old context — only the latest turn does.
5. Open a streaming LLM call. Yield text deltas to the caller and accumulate the
   full assistant response.
6. On stream completion, append the assistant turn (literal text, no context) to
   the session and persist it.

Citations are derived from the retrieved chunks, not parsed out of the model's
output — every retrieved chunk that was provided as context is surfaced so the
caller can render "Sources" exactly as the model saw them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from app.config import Settings, get_settings
from app.generation.llm import LLM, ChatMessage
from app.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    format_context_block,
    format_query_rewrite_user_turn,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)

Role = Literal["user", "assistant"]


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatTurn:
    """One literal turn of the conversation. Stored without injected context."""

    role: Role
    content: str


@dataclass
class Session:
    """In-memory chat session: ordered turns plus an id."""

    id: str
    turns: list[ChatTurn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        """Append a user turn."""
        self.turns.append(ChatTurn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant turn."""
        self.turns.append(ChatTurn(role="assistant", content=content))


class SessionStore(ABC):
    """Pluggable session storage. Swap for Redis / SQL by subclassing."""

    @abstractmethod
    def get(self, session_id: str) -> Session | None:
        """Return the session, or None if it doesn't exist."""

    @abstractmethod
    def get_or_create(self, session_id: str) -> Session:
        """Return the session, creating an empty one if it didn't exist."""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Persist the session (overwrites any prior state for the same id)."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete the session. Returns True if it existed."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return a sorted list of all session ids."""


class InMemorySessionStore(SessionStore):
    """Process-local session store. Lost on restart — fine for v1."""

    def __init__(self) -> None:
        """Initialise an empty store."""
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        """Return the stored session for `session_id`, or None."""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        """Return the session, creating an empty one if absent."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(id=session_id)
        return self._sessions[session_id]

    def save(self, session: Session) -> None:
        """Persist `session` under its id."""
        self._sessions[session.id] = session

    def delete(self, session_id: str) -> bool:
        """Drop the session. Returns True iff one existed."""
        return self._sessions.pop(session_id, None) is not None

    def list_ids(self) -> list[str]:
        """Return sorted session ids currently stored."""
        return sorted(self._sessions.keys())


# ---------------------------------------------------------------------------
# Citations and chat stream handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """A chunk that was supplied to the model as context for this turn."""

    source_file: str
    chunk_id: str
    snippet: str
    score: float
    page: int | None = None


@dataclass
class ChatStream:
    """Handle for streaming an assistant reply.

    `citations` and `rewritten_query` are populated synchronously before the
    LLM stream opens. `text_stream()` yields text deltas; on full iteration it
    appends the completed assistant turn to the session.
    """

    rewritten_query: str | None
    retrieved: list[SearchResult]
    citations: list[Citation]
    _deltas: Iterator[str]
    _on_complete: Any  # Callable[[str], None] — widened so mypy accepts the dataclass field
    _accumulated: list[str] = field(default_factory=list)
    _finalized: bool = False

    def text_stream(self) -> Iterator[str]:
        """Yield text deltas; finalize the session when the underlying iterator is exhausted."""
        try:
            for delta in self._deltas:
                self._accumulated.append(delta)
                yield delta
        finally:
            if not self._finalized:
                self._finalized = True
                self._on_complete("".join(self._accumulated))

    def final_text(self) -> str:
        """Return the full accumulated response text. Call after `text_stream()` is consumed."""
        return "".join(self._accumulated)


# ---------------------------------------------------------------------------
# Chat service
# ---------------------------------------------------------------------------


class ChatService:
    """Orchestrates retrieval + generation + memory for one chatbot."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        sessions: SessionStore,
        settings: Settings | None = None,
    ) -> None:
        """Bind dependencies. `settings` defaults to the process-wide config."""
        self.retriever = retriever
        self.llm = llm
        self.sessions = sessions
        self.settings = settings or get_settings()

    def stream_reply(self, session_id: str, user_message: str) -> ChatStream:
        """Run the full RAG pipeline for one user message and return a stream handle."""
        if not user_message or not user_message.strip():
            raise ValueError("user_message must be non-empty")

        session = self.sessions.get_or_create(session_id)

        rewritten = self._rewrite_query(session.turns, user_message) if session.turns else None
        search_query = rewritten if rewritten else user_message

        retrieved = self.retriever.retrieve(search_query, k=self.settings.top_k)

        context_chunks = [
            (
                str(r.metadata.get("source_file", "?")),
                r.chunk_id,
                r.content,
            )
            for r in retrieved
        ]
        context_block = format_context_block(context_chunks)
        user_turn_with_context = f"{context_block}\n\nQuestion: {user_message}"

        llm_messages: list[ChatMessage] = [
            ChatMessage(role=t.role, content=t.content) for t in session.turns
        ]
        llm_messages.append(ChatMessage(role="user", content=user_turn_with_context))

        session.add_user(user_message)

        deltas = self.llm.stream(
            messages=llm_messages,
            system=RAG_SYSTEM_PROMPT,
            cache_system=True,
        )

        citations = [self._to_citation(r) for r in retrieved]

        logger.info(
            "chat_turn_started",
            session_id=session_id,
            message_chars=len(user_message),
            prior_turns=len(session.turns) - 1,
            rewritten=bool(rewritten),
            retrieved=len(retrieved),
        )

        def on_complete(text: str) -> None:
            session.add_assistant(text)
            self.sessions.save(session)
            logger.info(
                "chat_turn_completed",
                session_id=session_id,
                response_chars=len(text),
            )

        return ChatStream(
            rewritten_query=rewritten,
            retrieved=list(retrieved),
            citations=citations,
            _deltas=deltas,
            _on_complete=on_complete,
        )

    def _rewrite_query(self, prior_turns: Sequence[ChatTurn], question: str) -> str | None:
        history = [(t.role, t.content) for t in prior_turns]
        user_turn = format_query_rewrite_user_turn(history, question)
        rewritten = self.llm.complete(
            messages=[ChatMessage(role="user", content=user_turn)],
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            max_tokens=200,
        ).strip()
        return rewritten or None

    @staticmethod
    def _to_citation(result: SearchResult) -> Citation:
        page_raw = result.metadata.get("page")
        page = int(page_raw) if isinstance(page_raw, int) else None
        return Citation(
            source_file=str(result.metadata.get("source_file", "?")),
            chunk_id=result.chunk_id,
            snippet=_snippet(result.content),
            score=result.score,
            page=page,
        )


def _snippet(text: str, max_len: int = 220) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return f"{cut}..."
