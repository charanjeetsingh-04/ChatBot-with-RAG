"""Tests for the in-memory session store."""

from __future__ import annotations

from app.generation.chat import ChatTurn, InMemorySessionStore, Session


def test_get_returns_none_for_missing() -> None:
    store = InMemorySessionStore()
    assert store.get("nope") is None


def test_get_or_create_creates_empty_session() -> None:
    store = InMemorySessionStore()
    s = store.get_or_create("abc")
    assert isinstance(s, Session)
    assert s.id == "abc"
    assert s.turns == []


def test_get_or_create_returns_existing_session() -> None:
    store = InMemorySessionStore()
    s1 = store.get_or_create("abc")
    s1.add_user("hi")
    s2 = store.get_or_create("abc")
    assert s2 is s1
    assert len(s2.turns) == 1


def test_save_persists_session() -> None:
    store = InMemorySessionStore()
    s = Session(id="x")
    s.turns.append(ChatTurn(role="user", content="hello"))
    store.save(s)
    fetched = store.get("x")
    assert fetched is not None
    assert fetched.turns[0].content == "hello"


def test_delete_returns_true_when_present() -> None:
    store = InMemorySessionStore()
    store.get_or_create("a")
    assert store.delete("a") is True
    assert store.get("a") is None


def test_delete_returns_false_when_missing() -> None:
    store = InMemorySessionStore()
    assert store.delete("never-there") is False


def test_list_ids_is_sorted() -> None:
    store = InMemorySessionStore()
    store.get_or_create("z")
    store.get_or_create("a")
    store.get_or_create("m")
    assert store.list_ids() == ["a", "m", "z"]


def test_session_add_helpers_append_turns_in_order() -> None:
    s = Session(id="x")
    s.add_user("question 1")
    s.add_assistant("answer 1")
    s.add_user("question 2")
    assert [(t.role, t.content) for t in s.turns] == [
        ("user", "question 1"),
        ("assistant", "answer 1"),
        ("user", "question 2"),
    ]
