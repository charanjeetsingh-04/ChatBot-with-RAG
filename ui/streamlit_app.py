"""Streamlit UI for the RAG chatbot.

Talks to the FastAPI backend via HTTP:
- GET  /health             — backend status indicator in the sidebar
- POST /ingest             — file uploader (multipart upload)
- GET  /documents          — list of ingested sources
- DELETE /documents/{id}   — per-source delete button
- POST /chat               — streamed reply (Server-Sent Events) rendered live

Run with:
    uv run streamlit run ui/streamlit_app.py

Set RAG_API_BASE in the environment (or in the sidebar) to point at a non-default
backend URL.

A v2 of the frontend would plug in here: replace this file with a Next.js client
that hits the same FastAPI endpoints; the backend contract does not change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any, cast

import httpx
import streamlit as st

DEFAULT_API_BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Backend client helpers
# ---------------------------------------------------------------------------


def fetch_health(api_base: str) -> dict[str, Any] | None:
    """Return the /health payload, or None if the backend is unreachable."""
    try:
        r = httpx.get(f"{api_base}/health", timeout=5.0)
        r.raise_for_status()
        return cast(dict[str, Any], r.json())
    except httpx.HTTPError:
        return None


def list_documents(api_base: str) -> list[dict[str, Any]]:
    """Return the list of ingested documents from /documents."""
    r = httpx.get(f"{api_base}/documents", timeout=10.0)
    r.raise_for_status()
    return cast(list[dict[str, Any]], r.json()["documents"])


def ingest_upload(
    api_base: str, name: str, data: bytes, content_type: str
) -> dict[str, Any]:
    """POST a file to /ingest as multipart form data. Returns the JSON response."""
    r = httpx.post(
        f"{api_base}/ingest",
        files={"file": (name, data, content_type)},
        timeout=120.0,
    )
    r.raise_for_status()
    return cast(dict[str, Any], r.json())


def delete_document(api_base: str, source_file: str) -> dict[str, Any]:
    """DELETE /documents/{source_file}. Returns the JSON response."""
    r = httpx.delete(f"{api_base}/documents/{source_file}", timeout=10.0)
    r.raise_for_status()
    return cast(dict[str, Any], r.json())


def stream_chat(
    api_base: str, message: str, session_id: str | None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event_name, data) tuples from the SSE response of POST /chat."""
    payload: dict[str, Any] = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    with httpx.stream(
        "POST",
        f"{api_base}/chat",
        json=payload,
        timeout=httpx.Timeout(60.0, read=None),
    ) as r:
        r.raise_for_status()
        event_name = ""
        data_buf: list[str] = []
        for raw in r.iter_lines():
            line = raw if isinstance(raw, str) else raw.decode("utf-8")
            if line == "":
                if event_name and data_buf:
                    yield event_name, json.loads("\n".join(data_buf))
                event_name = ""
                data_buf = []
                continue
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_buf.append(line[len("data: ") :])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_sources(citations: list[dict[str, Any]]) -> None:
    """Render the expandable Sources panel for an assistant message."""
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for i, c in enumerate(citations, start=1):
            page_str = f" · page {c['page']}" if c.get("page") is not None else ""
            st.markdown(
                f"**{i}. {c['source_file']}**{page_str} · "
                f"score `{c['score']:.3f}` · "
                f"chunk `{c['chunk_id']}`"
            )
            st.markdown(f"> {c['snippet']}")
            if i < len(citations):
                st.divider()


def _render_message(msg: dict[str, Any]) -> None:
    """Render a single chat message (re-renders happen on every Streamlit rerun)."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("citations") or [])
            rewritten = msg.get("rewritten_query")
            if rewritten:
                st.caption(f"Searched for: _{rewritten}_")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Initialise Streamlit session state on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API_BASE


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render the sidebar: config, uploader, document list, new-conversation."""
    with st.sidebar:
        st.header("Settings")
        st.session_state.api_base = st.text_input(
            "Backend URL",
            value=st.session_state.api_base,
            help="FastAPI backend base URL.",
        )
        api_base = st.session_state.api_base

        health = fetch_health(api_base)
        if health:
            st.success(f"Backend ok · `{health['llm_model']}`")
        else:
            st.error(
                "Backend unreachable. Start it with "
                "`uv run uvicorn app.main:app --reload`."
            )

        st.divider()
        st.header("Documents")

        uploaded = st.file_uploader(
            "Upload document",
            type=["pdf", "txt", "md", "markdown", "docx"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            # Cache key per (name, size) so we don't re-ingest on every rerun while the
            # uploader still holds the same file.
            ingest_key = f"ingested_{uploaded.name}_{uploaded.size}"
            if not st.session_state.get(ingest_key):
                with st.spinner(f"Ingesting {uploaded.name}…"):
                    try:
                        result = ingest_upload(
                            api_base,
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/pdf",
                        )
                        replaced = result.get("chunks_replaced", 0)
                        suffix = f" (replaced {replaced})" if replaced else ""
                        st.success(
                            f"Ingested **{result['chunks_added']}** chunks "
                            f"from `{uploaded.name}`{suffix}."
                        )
                        st.session_state[ingest_key] = True
                    except httpx.HTTPError as e:
                        st.error(f"Ingest failed: {e}")

        try:
            docs = list_documents(api_base)
        except httpx.HTTPError:
            docs = []

        if docs:
            st.subheader(f"Ingested ({len(docs)})")
            for d in docs:
                src = d["source_file"]
                display = src.rsplit("/", 1)[-1]
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 `{display}`")
                if col2.button("✕", key=f"del_{src}", help=f"Delete {src}"):
                    try:
                        result = delete_document(api_base, src)
                        st.toast(f"Removed {result['chunks_deleted']} chunks.")
                        st.rerun()
                    except httpx.HTTPError as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.caption("No documents yet — upload a PDF to begin.")

        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Top-level Streamlit page."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon=":books:",
        layout="wide",
    )
    _init_state()

    st.title("RAG Chatbot")
    st.caption("Ask questions grounded in your uploaded documents.")

    _render_sidebar()

    for msg in st.session_state.messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a question…")
    if not prompt:
        return

    user_msg: dict[str, Any] = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            events = stream_chat(
                st.session_state.api_base,
                prompt,
                st.session_state.session_id,
            )

            citations: list[dict[str, Any]] = []
            rewritten: str | None = None

            for event_name, event_data in events:
                if event_name == "citations":
                    citations = event_data.get("citations") or []
                    rewritten = event_data.get("rewritten_query")
                    st.session_state.session_id = event_data.get("session_id")
                    break

            stream_error: dict[str, Any] | None = None

            def text_deltas() -> Iterator[str]:
                nonlocal stream_error
                for event_name, event_data in events:
                    if event_name == "delta":
                        yield event_data.get("text", "")
                    elif event_name == "error":
                        stream_error = event_data
                        return
                    elif event_name == "done":
                        return

            full_text = st.write_stream(text_deltas())
            full_text_str = full_text if isinstance(full_text, str) else "".join(full_text)

            if stream_error is not None:
                msg = stream_error.get("message", "unknown error")
                err_type = stream_error.get("error_type", "Error")
                st.error(f"{err_type}: {msg}")

            _render_sources(citations)
            if rewritten:
                st.caption(f"Searched for: _{rewritten}_")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_text_str,
                    "citations": citations,
                    "rewritten_query": rewritten,
                }
            )
        except httpx.HTTPError as e:
            err = f"Chat request failed: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
