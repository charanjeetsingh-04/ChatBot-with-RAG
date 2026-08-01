"""HTTP API tests using FastAPI's TestClient.

Dependency overrides swap in deterministic test components (HashEmbedder,
StubLLM, fresh ChromaVectorStore, fresh InMemorySessionStore) so no API key
or external network is required.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.api.routes import (
    get_embedder,
    get_llm,
    get_session_store,
    get_vector_store,
)
from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder
from app.generation.chat import InMemorySessionStore
from app.generation.llm import LLM
from app.main import app
from app.retrieval.vector_store import ChromaVectorStore, VectorStore
from tests.conftest import HashEmbedder, StubLLM


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Per-test settings: small chunks, fixed top_k, uploads dir under tmp_path."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        chunk_size=300,
        chunk_overlap=40,
        top_k=3,
        uploads_path=tmp_path / "uploads",
    )


@pytest.fixture()
def test_store() -> ChromaVectorStore:
    """Fresh ephemeral vector store per test (unique collection name)."""
    return ChromaVectorStore(path=None, collection_name=f"api_{uuid.uuid4().hex[:8]}")


@pytest.fixture()
def test_sessions() -> InMemorySessionStore:
    """Fresh in-memory session store per test."""
    return InMemorySessionStore()


@pytest.fixture()
def client(
    test_settings: Settings,
    hash_embedder: HashEmbedder,
    test_store: ChromaVectorStore,
    stub_llm: StubLLM,
    test_sessions: InMemorySessionStore,
) -> Iterator[TestClient]:
    """TestClient with all heavy deps swapped for deterministic test components."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_embedder] = lambda: cast(Embedder, hash_embedder)
    app.dependency_overrides[get_vector_store] = lambda: cast(VectorStore, test_store)
    app.dependency_overrides[get_llm] = lambda: cast(LLM, stub_llm)
    app.dependency_overrides[get_session_store] = lambda: test_sessions
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_provider_and_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "anthropic"
    assert body["llm_model"].startswith("claude-")


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


def test_ingest_via_path(client: TestClient, sample_pdf: Path) -> None:
    r = client.post("/ingest", data={"path": str(sample_pdf)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks_added"] > 0
    assert "/" in body["file_path"]  # POSIX path
    assert "\\" not in body["file_path"]


def test_ingest_via_upload(
    client: TestClient,
    sample_pdf: Path,
    test_settings: Settings,
) -> None:
    with sample_pdf.open("rb") as fp:
        r = client.post(
            "/ingest",
            files={"file": ("uploaded.pdf", fp, "application/pdf")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks_added"] > 0
    # File is persisted under uploads/ with a content-hash prefix.
    assert "/uploads/" in body["file_path"]
    assert "uploaded.pdf" in body["file_path"]
    assert (test_settings.uploads_path).exists()


def test_ingest_both_file_and_path_rejected(client: TestClient, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as fp:
        r = client.post(
            "/ingest",
            data={"path": str(sample_pdf)},
            files={"file": ("doc.pdf", fp, "application/pdf")},
        )
    assert r.status_code == 400
    assert "not both" in r.json()["detail"]


def test_ingest_neither_rejected(client: TestClient) -> None:
    r = client.post("/ingest")
    assert r.status_code == 400


def test_ingest_missing_path_returns_404(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/ingest", data={"path": str(tmp_path / "nope.pdf")})
    assert r.status_code == 404


def test_ingest_unsupported_extension_returns_400(client: TestClient, tmp_path: Path) -> None:
    bad = tmp_path / "x.xyz"
    bad.write_text("hi")
    r = client.post("/ingest", data={"path": str(bad)})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


def _parse_sse(body: bytes) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE response body into a list of (event_name, data_dict)."""
    events: list[tuple[str, dict[str, object]]] = []
    for frame in body.decode("utf-8").split("\n\n"):
        if not frame.strip():
            continue
        event_name = ""
        data_str = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_str = line[len("data: "):]
        if event_name and data_str:
            events.append((event_name, json.loads(data_str)))
    return events


def test_chat_empty_message_returns_422(client: TestClient) -> None:
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_streams_citations_deltas_and_done(
    client: TestClient, sample_pdf: Path
) -> None:
    # Ingest first so there's a corpus.
    assert client.post("/ingest", data={"path": str(sample_pdf)}).status_code == 200

    r = client.post("/chat", json={"message": "Who walked on the Moon?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.content)
    event_names = [e[0] for e in events]
    assert event_names[0] == "citations"
    assert event_names[-1] == "done"
    assert "delta" in event_names

    citations_evt = events[0][1]
    assert isinstance(citations_evt["session_id"], str)
    assert isinstance(citations_evt["citations"], list)
    assert len(cast(list[object], citations_evt["citations"])) > 0
    first_citation = cast(list[dict[str, object]], citations_evt["citations"])[0]
    assert set(first_citation.keys()) >= {"source_file", "chunk_id", "snippet", "score"}

    deltas = [cast(dict[str, str], e[1])["text"] for e in events if e[0] == "delta"]
    assert "".join(deltas) == "stub stream response"


def test_chat_auto_generates_session_id(client: TestClient, sample_pdf: Path) -> None:
    client.post("/ingest", data={"path": str(sample_pdf)})

    r1 = client.post("/chat", json={"message": "first"})
    r2 = client.post("/chat", json={"message": "second"})
    evt1 = _parse_sse(r1.content)[0][1]
    evt2 = _parse_sse(r2.content)[0][1]
    assert evt1["session_id"] != evt2["session_id"]


def test_chat_stream_failure_emits_error_event(
    client: TestClient, sample_pdf: Path, stub_llm: StubLLM
) -> None:
    """If the LLM stream raises mid-iteration, the SSE response must end with an
    `error` event rather than dropping the connection silently."""
    client.post("/ingest", data={"path": str(sample_pdf)})

    original_stream = stub_llm.stream

    def boom(*args: object, **kwargs: object) -> Iterator[str]:
        del args, kwargs
        yield "partial "
        raise RuntimeError("boom")

    stub_llm.stream = boom  # type: ignore[method-assign]
    try:
        r = client.post("/chat", json={"message": "Anything"})
    finally:
        stub_llm.stream = original_stream  # type: ignore[method-assign]

    assert r.status_code == 200
    events = _parse_sse(r.content)
    event_names = [e[0] for e in events]
    assert event_names[0] == "citations"
    assert event_names[-1] == "error"
    assert "done" not in event_names

    err = events[-1][1]
    assert err["error_type"] == "RuntimeError"
    assert err["message"] == "boom"
    assert "session_id" in err


def test_chat_session_id_persists(client: TestClient, sample_pdf: Path) -> None:
    client.post("/ingest", data={"path": str(sample_pdf)})

    r1 = client.post("/chat", json={"message": "first"})
    sid = cast(str, _parse_sse(r1.content)[0][1]["session_id"])

    r2 = client.post("/chat", json={"message": "second", "session_id": sid})
    evt2 = _parse_sse(r2.content)[0][1]
    assert evt2["session_id"] == sid


# ---------------------------------------------------------------------------
# /sessions/{id}
# ---------------------------------------------------------------------------


def test_get_session_returns_404_for_missing(client: TestClient) -> None:
    r = client.get("/sessions/does-not-exist")
    assert r.status_code == 404


def test_get_session_returns_history(client: TestClient, sample_pdf: Path) -> None:
    client.post("/ingest", data={"path": str(sample_pdf)})
    chat = client.post("/chat", json={"message": "Who landed on the Moon?"})
    sid = cast(str, _parse_sse(chat.content)[0][1]["session_id"])

    r = client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == sid
    assert len(body["turns"]) == 2
    assert body["turns"][0]["role"] == "user"
    assert body["turns"][0]["content"] == "Who landed on the Moon?"
    assert body["turns"][1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# /documents
# ---------------------------------------------------------------------------


def test_list_documents_after_ingest(client: TestClient, sample_pdf: Path) -> None:
    r0 = client.get("/documents")
    assert r0.status_code == 200
    assert r0.json() == {"documents": []}

    client.post("/ingest", data={"path": str(sample_pdf)})

    r1 = client.get("/documents")
    docs = r1.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["source_file"] == sample_pdf.as_posix()


def test_delete_document_removes_chunks(
    client: TestClient,
    sample_pdf: Path,
    test_store: ChromaVectorStore,
) -> None:
    client.post("/ingest", data={"path": str(sample_pdf)})
    before = test_store.count()
    assert before > 0

    source = sample_pdf.as_posix()
    r = client.delete(f"/documents/{source}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks_deleted"] == before
    assert test_store.count() == 0


def test_delete_document_zero_when_unknown(client: TestClient) -> None:
    r = client.delete("/documents/never-ingested.pdf")
    assert r.status_code == 200
    assert r.json()["chunks_deleted"] == 0


def test_reingest_after_delete_repopulates(
    client: TestClient,
    sample_pdf: Path,
    test_store: ChromaVectorStore,
) -> None:
    """The 'remove and re-index' workflow from the spec."""
    client.post("/ingest", data={"path": str(sample_pdf)})
    n = test_store.count()

    client.delete(f"/documents/{sample_pdf.as_posix()}")
    assert test_store.count() == 0

    client.post("/ingest", data={"path": str(sample_pdf)})
    assert test_store.count() == n


# ---------------------------------------------------------------------------
# End-to-end: ingest → chat → session → delete
# ---------------------------------------------------------------------------


def test_full_flow(
    client: TestClient,
    sample_pdf: Path,
    test_store: ChromaVectorStore,
) -> None:
    # 1. Ingest
    r_ing = client.post("/ingest", data={"path": str(sample_pdf)})
    assert r_ing.status_code == 200
    assert test_store.count() > 0

    # 2. Chat
    r_chat = client.post("/chat", json={"message": "What is Apollo 11?"})
    assert r_chat.status_code == 200
    events = _parse_sse(r_chat.content)
    sid = cast(str, events[0][1]["session_id"])

    # 3. Session history
    r_sess = client.get(f"/sessions/{sid}")
    assert r_sess.status_code == 200
    assert len(r_sess.json()["turns"]) == 2

    # 4. Document listed
    r_docs = client.get("/documents")
    assert any(
        d["source_file"] == sample_pdf.as_posix() for d in r_docs.json()["documents"]
    )

    # 5. Delete
    r_del = client.delete(f"/documents/{sample_pdf.as_posix()}")
    assert r_del.status_code == 200
    assert r_del.json()["chunks_deleted"] > 0
