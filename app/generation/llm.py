"""LLM interface and provider implementations.

Design notes:
- The interface exposes both `stream` (token-by-token deltas for chat UI) and
  `complete` (non-streaming, for short utility calls like query rewriting).
- `cache_system=True` attaches `cache_control: {"type": "ephemeral"}` to the
  Anthropic system block. Other providers may no-op the flag if explicit-only
  caching APIs make per-call caching impractical (see `GeminiLLM`).
- Opus 4.7 removes `temperature`, `top_p`, `top_k`, and `budget_tokens` — sending
  any of them returns a 400. None are exposed on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import anthropic
from google import genai
from google.genai import types as genai_types

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single chat turn passed to the LLM."""

    role: Role
    content: str


class LLM(ABC):
    """Provider-agnostic LLM interface."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The model id this instance will call."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Yield response text deltas in arrival order. Caller consumes until exhausted."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the full response text in one call (non-streaming)."""


class AnthropicLLM(LLM):
    """Anthropic implementation. Default model is Claude Opus 4.7."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-7",
        max_tokens_default: int = 1024,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Bind credentials and defaults. `client` is an injection point for tests."""
        self._client: anthropic.Anthropic = client or anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens_default = max_tokens_default

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self._model

    def stream(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Yield text deltas from a streamed Anthropic response."""
        kwargs = self._build_kwargs(messages, system, max_tokens, cache_system)
        with self._client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream

    def complete(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single-shot completion; returns concatenated text blocks from the response."""
        kwargs = self._build_kwargs(messages, system, max_tokens, cache_system=False)
        response = self._client.messages.create(**kwargs)
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(str(block.text))
        return "".join(parts).strip()

    def _build_kwargs(
        self,
        messages: Sequence[ChatMessage],
        system: str | None,
        max_tokens: int | None,
        cache_system: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens_default,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        system_blocks = self._build_system(system, cache_system)
        if system_blocks is not None:
            kwargs["system"] = system_blocks
        return kwargs

    @staticmethod
    def _build_system(
        system: str | None, cache: bool
    ) -> list[dict[str, Any]] | None:
        if not system:
            return None
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]


class GeminiLLM(LLM):
    """Google Gemini implementation (google-genai SDK).

    Roles are mapped on the way out: our `ChatMessage.role == "assistant"`
    becomes `"model"` in Gemini's `contents`. The `cache_system` flag is a
    no-op here — Gemini's prompt caching is an explicit `caches.create(...)`
    lifecycle with a minimum token floor, not a per-call header, so wiring
    it through would silently fail for small system prompts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_tokens_default: int = 1024,
        client: genai.Client | None = None,
    ) -> None:
        """Bind credentials and defaults. `client` is an injection point for tests."""
        self._client: genai.Client = client or genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens_default = max_tokens_default

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self._model

    def stream(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Yield text deltas from a streamed Gemini response."""
        contents = self._to_contents(messages)
        config = self._build_config(system, max_tokens)
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=cast(Any, contents),
            config=config,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def complete(
        self,
        messages: Sequence[ChatMessage],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single-shot completion; returns the concatenated response text."""
        contents = self._to_contents(messages)
        config = self._build_config(system, max_tokens)
        response = self._client.models.generate_content(
            model=self._model,
            contents=cast(Any, contents),
            config=config,
        )
        return (response.text or "").strip()

    @staticmethod
    def _to_contents(messages: Sequence[ChatMessage]) -> list[genai_types.Content]:
        out: list[genai_types.Content] = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            out.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=m.content)],
                )
            )
        return out

    def _build_config(
        self, system: str | None, max_tokens: int | None
    ) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens if max_tokens is not None else self._max_tokens_default,
        )


def make_llm(
    provider: str,
    model: str,
    api_key: str | None = None,
    max_tokens_default: int = 1024,
) -> LLM:
    """Construct an `LLM` for the configured provider.

    Recognised providers: `anthropic`, `gemini`. `openai` is reserved for a later phase.
    """
    if provider == "anthropic":
        return AnthropicLLM(
            api_key=api_key,
            model=model,
            max_tokens_default=max_tokens_default,
        )
    if provider == "gemini":
        return GeminiLLM(
            api_key=api_key,
            model=model,
            max_tokens_default=max_tokens_default,
        )
    if provider == "openai":
        raise NotImplementedError("OpenAI LLM impl is not wired yet.")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
