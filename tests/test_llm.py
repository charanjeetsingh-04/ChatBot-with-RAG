"""Tests for the LLM interface and the Anthropic implementation.

A `FakeAnthropic` client captures the kwargs passed to `messages.stream()` /
`messages.create()` so we can assert on request shape (model, system block,
cache_control) without making real API calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import anthropic
import pytest
from google import genai

from app.generation.llm import (
    LLM,
    AnthropicLLM,
    ChatMessage,
    GeminiLLM,
    make_llm,
)


@dataclass
class _FakeStream:
    chunks: list[str]
    final: Any = None

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    @property
    def text_stream(self) -> Iterator[str]:
        return iter(self.chunks)

    def get_final_message(self) -> Any:
        return self.final


@dataclass
class _FakeMessages:
    stream_chunks: list[str] = field(default_factory=lambda: ["Hello", ", ", "world."])
    complete_text: str = "complete-response"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append({"method": "stream", "kwargs": kwargs})
        return _FakeStream(chunks=list(self.stream_chunks))

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({"method": "create", "kwargs": kwargs})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.complete_text)]
        )


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def _make_llm(
    model: str = "claude-opus-4-7",
    max_tokens_default: int = 1024,
) -> tuple[AnthropicLLM, _FakeAnthropic]:
    fake = _FakeAnthropic()
    llm = AnthropicLLM(
        model=model,
        max_tokens_default=max_tokens_default,
        client=cast(anthropic.Anthropic, fake),
    )
    return llm, fake


def test_llm_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLM()  # type: ignore[abstract]


def test_anthropic_llm_exposes_model() -> None:
    llm, _ = _make_llm(model="claude-opus-4-7")
    assert llm.model == "claude-opus-4-7"


def test_stream_yields_text_chunks_in_order() -> None:
    llm, _ = _make_llm()
    out = list(llm.stream([ChatMessage("user", "hi")]))
    assert out == ["Hello", ", ", "world."]


def test_stream_sends_model_and_messages() -> None:
    llm, fake = _make_llm(model="claude-opus-4-7", max_tokens_default=512)
    list(llm.stream([ChatMessage("user", "ping")]))
    call = fake.messages.calls[0]
    assert call["method"] == "stream"
    assert call["kwargs"]["model"] == "claude-opus-4-7"
    assert call["kwargs"]["max_tokens"] == 512
    assert call["kwargs"]["messages"] == [{"role": "user", "content": "ping"}]


def test_stream_omits_system_when_none() -> None:
    llm, fake = _make_llm()
    list(llm.stream([ChatMessage("user", "x")], system=None))
    assert "system" not in fake.messages.calls[0]["kwargs"]


def test_stream_attaches_cache_control_when_requested() -> None:
    llm, fake = _make_llm()
    list(llm.stream([ChatMessage("user", "x")], system="SYS", cache_system=True))
    system = fake.messages.calls[0]["kwargs"]["system"]
    assert system == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]


def test_stream_omits_cache_control_when_not_requested() -> None:
    llm, fake = _make_llm()
    list(llm.stream([ChatMessage("user", "x")], system="SYS", cache_system=False))
    system = fake.messages.calls[0]["kwargs"]["system"]
    assert system == [{"type": "text", "text": "SYS"}]
    assert "cache_control" not in system[0]


def test_stream_max_tokens_override() -> None:
    llm, fake = _make_llm(max_tokens_default=100)
    list(llm.stream([ChatMessage("user", "x")], max_tokens=999))
    assert fake.messages.calls[0]["kwargs"]["max_tokens"] == 999


def test_stream_does_not_send_temperature_top_p_top_k() -> None:
    """Opus 4.7 returns 400 if these are present — guard against accidental regressions."""
    llm, fake = _make_llm()
    list(llm.stream([ChatMessage("user", "x")], system="SYS"))
    kwargs = fake.messages.calls[0]["kwargs"]
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert "thinking" not in kwargs


def test_stream_preserves_message_roles_and_order() -> None:
    llm, fake = _make_llm()
    msgs = [
        ChatMessage("user", "hi"),
        ChatMessage("assistant", "hello"),
        ChatMessage("user", "and now?"),
    ]
    list(llm.stream(msgs))
    assert fake.messages.calls[0]["kwargs"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"},
    ]


def test_complete_returns_concatenated_text_blocks() -> None:
    llm, _ = _make_llm()
    out = llm.complete([ChatMessage("user", "ping")])
    assert out == "complete-response"


def test_complete_uses_create_not_stream() -> None:
    llm, fake = _make_llm()
    llm.complete([ChatMessage("user", "ping")])
    assert fake.messages.calls[0]["method"] == "create"


def test_complete_includes_system_block_without_cache_control() -> None:
    """`complete` is for short utility calls (e.g. query rewriting); never caches."""
    llm, fake = _make_llm()
    llm.complete([ChatMessage("user", "x")], system="rewrite this")
    system = fake.messages.calls[0]["kwargs"]["system"]
    assert system == [{"type": "text", "text": "rewrite this"}]


def test_make_llm_anthropic_returns_correct_model() -> None:
    llm = make_llm(provider="anthropic", model="claude-opus-4-7", api_key="sk-fake")
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == "claude-opus-4-7"


def test_make_llm_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_llm(provider="bogus", model="x")


def test_make_llm_openai_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        make_llm(provider="openai", model="gpt-x")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@dataclass
class _FakeGenAIModels:
    stream_chunks: list[str] = field(default_factory=lambda: ["Hel", "lo", " world."])
    complete_text: str = "gemini-complete-response"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content_stream(self, **kwargs: Any) -> Iterator[Any]:
        self.calls.append({"method": "stream", "kwargs": kwargs})
        return iter(SimpleNamespace(text=t) for t in self.stream_chunks)

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append({"method": "complete", "kwargs": kwargs})
        return SimpleNamespace(text=self.complete_text)


class _FakeGenAIClient:
    def __init__(self) -> None:
        self.models = _FakeGenAIModels()


def _make_gemini(
    model: str = "gemini-2.5-flash",
    max_tokens_default: int = 1024,
) -> tuple[GeminiLLM, _FakeGenAIClient]:
    fake = _FakeGenAIClient()
    llm = GeminiLLM(
        model=model,
        max_tokens_default=max_tokens_default,
        client=cast(genai.Client, fake),
    )
    return llm, fake


def test_gemini_exposes_model() -> None:
    llm, _ = _make_gemini(model="gemini-2.5-flash")
    assert llm.model == "gemini-2.5-flash"


def test_gemini_stream_yields_text_chunks_in_order() -> None:
    llm, _ = _make_gemini()
    out = list(llm.stream([ChatMessage("user", "hi")]))
    assert out == ["Hel", "lo", " world."]


def test_gemini_stream_skips_empty_text_chunks() -> None:
    llm, fake = _make_gemini()
    fake.models.stream_chunks = ["a", "", "b"]
    out = list(llm.stream([ChatMessage("user", "hi")]))
    assert out == ["a", "b"]


def test_gemini_stream_sends_model_and_max_tokens() -> None:
    llm, fake = _make_gemini(model="gemini-2.5-flash", max_tokens_default=512)
    list(llm.stream([ChatMessage("user", "ping")]))
    call = fake.models.calls[0]
    assert call["method"] == "stream"
    assert call["kwargs"]["model"] == "gemini-2.5-flash"
    assert call["kwargs"]["config"].max_output_tokens == 512


def test_gemini_stream_passes_system_instruction() -> None:
    llm, fake = _make_gemini()
    list(llm.stream([ChatMessage("user", "x")], system="SYS"))
    config = fake.models.calls[0]["kwargs"]["config"]
    assert config.system_instruction == "SYS"


def test_gemini_stream_omits_system_when_none() -> None:
    llm, fake = _make_gemini()
    list(llm.stream([ChatMessage("user", "x")], system=None))
    config = fake.models.calls[0]["kwargs"]["config"]
    assert config.system_instruction is None


def test_gemini_maps_assistant_role_to_model() -> None:
    llm, fake = _make_gemini()
    msgs = [
        ChatMessage("user", "hi"),
        ChatMessage("assistant", "hello"),
        ChatMessage("user", "and now?"),
    ]
    list(llm.stream(msgs))
    contents = fake.models.calls[0]["kwargs"]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [c.parts[0].text for c in contents] == ["hi", "hello", "and now?"]


def test_gemini_cache_system_is_noop() -> None:
    """`cache_system` flag must not crash and must not alter the config — Gemini
    caching is an explicit lifecycle, not a per-call header."""
    llm, fake = _make_gemini()
    list(llm.stream([ChatMessage("user", "x")], system="SYS", cache_system=True))
    config = fake.models.calls[0]["kwargs"]["config"]
    assert config.system_instruction == "SYS"


def test_gemini_complete_returns_text() -> None:
    llm, _ = _make_gemini()
    out = llm.complete([ChatMessage("user", "ping")])
    assert out == "gemini-complete-response"


def test_gemini_complete_uses_generate_content_not_stream() -> None:
    llm, fake = _make_gemini()
    llm.complete([ChatMessage("user", "ping")])
    assert fake.models.calls[0]["method"] == "complete"


def test_gemini_complete_max_tokens_override() -> None:
    llm, fake = _make_gemini(max_tokens_default=100)
    llm.complete([ChatMessage("user", "x")], max_tokens=42)
    config = fake.models.calls[0]["kwargs"]["config"]
    assert config.max_output_tokens == 42


def test_make_llm_gemini_returns_correct_model() -> None:
    llm = make_llm(provider="gemini", model="gemini-2.5-flash", api_key="AI-fake")
    assert isinstance(llm, GeminiLLM)
    assert llm.model == "gemini-2.5-flash"
