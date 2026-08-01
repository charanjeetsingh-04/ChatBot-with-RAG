"""Prompt templates used by the chat orchestrator.

`RAG_SYSTEM_PROMPT` is intentionally written first and frozen — it forms the
stable prefix that benefits from prompt caching. Volatile content (retrieved
chunks for the current query, the latest user question) belongs in the user
turn so the cached prefix stays valid across requests.
"""

from __future__ import annotations

from collections.abc import Sequence

RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant operating in a retrieval-augmented setting.

You will receive:
1. A user question.
2. A set of <context> blocks containing chunks retrieved from a custom document corpus, each tagged with its source file and chunk id.

Rules — follow them literally:
- Answer ONLY using information present in the <context> blocks. Do not draw on general world knowledge or invent facts, names, dates, numbers, or quotations not present in the context.
- If the context does not contain enough information to answer the question, reply exactly: "I don't know based on the provided context." Do not speculate.
- Keep answers concise and directly responsive to the question. Prefer plain prose over filler.
- When a specific claim relies on a chunk, you may briefly note its source inline like (source: filename.pdf, chunk_id). The application also surfaces citations separately, so do not list every chunk at the end.
- If multiple chunks conflict, say so plainly and quote the relevant excerpts."""


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search queries.

Given a short conversation and the user's latest question, output ONE rewritten query that captures the intent assuming no prior conversation context is available.

Output ONLY the rewritten query. No preamble, no quotes, no explanation, no trailing punctuation beyond what the query needs."""


def format_query_rewrite_user_turn(history: Sequence[tuple[str, str]], question: str) -> str:
    """Render the user-turn payload for the query rewriter.

    `history` is an ordered list of `(role, content)` tuples for prior turns.
    The latest user question is passed separately so it ends up last.
    """
    if not history:
        return f"Latest question: {question}"
    lines = ["Conversation so far:"]
    for role, content in history:
        lines.append(f"{role}: {content}")
    lines.append("")
    lines.append(f"Latest question: {question}")
    return "\n".join(lines)


def format_context_block(chunks: Sequence[tuple[str, str, str]]) -> str:
    """Render retrieved chunks as a single user-turn fragment.

    Each chunk is `(source_file, chunk_id, content)`. The resulting string is
    intended to be concatenated with the user's question in the user turn.
    """
    if not chunks:
        return "<context>(no chunks retrieved)</context>"
    blocks: list[str] = ["<context>"]
    for source_file, chunk_id, content in chunks:
        blocks.append(f'  <chunk source="{source_file}" id="{chunk_id}">')
        blocks.append(content)
        blocks.append("  </chunk>")
    blocks.append("</context>")
    return "\n".join(blocks)
