"""Generate docs/about-this-project.pdf -- a one-shot project overview.

Run once with:
    uv run python scripts/generate_overview_pdf.py

Re-run to regenerate after editing the text below.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "about-this-project.pdf"


SECTIONS: list[tuple[str, str]] = [
    (
        "What it is",
        "A production-quality Retrieval-Augmented Generation (RAG) chatbot. "
        "You upload PDF / TXT / Markdown / DOCX documents into a local corpus, "
        "ask questions, and receive answers that are grounded in -- and cite "
        "back to -- the exact chunks of those documents. No hallucinated facts; "
        "if the answer is not in the corpus, the bot says so plainly.",
    ),
    (
        "What it does (end-to-end flow)",
        "1. Ingest:   load a document, split into ~800-char chunks with overlap, "
        "embed each chunk with a sentence-transformer model, store in ChromaDB "
        "with deterministic chunk ids (re-uploads dedupe automatically).\n"
        "2. Retrieve: embed the user's question, run cosine search against the "
        "vector store (optionally fuse with BM25 keyword scores via Reciprocal "
        "Rank Fusion). Return the top-k relevant chunks.\n"
        "3. Generate: build a prompt that places the retrieved chunks in a "
        "<context> block alongside the question, stream the LLM's reply token "
        "by token over Server-Sent Events.\n"
        "4. Cite:     every retrieved chunk that was shown to the model is "
        "surfaced as a citation (source file, page, score, snippet) so the "
        "user can verify the answer.",
    ),
    (
        "Stack",
        "API:         FastAPI + uvicorn (Python 3.11+)\n"
        "LLM:         Anthropic Claude (claude-opus-4-7) or Google Gemini "
        "(gemini-2.5-flash) -- swappable in one file.\n"
        "Embeddings:  sentence-transformers, default BAAI/bge-small-en-v1.5.\n"
        "Vector DB:   ChromaDB (persistent, local, cosine similarity).\n"
        "Chunking:    Recursive character splitter with deterministic SHA-256 ids.\n"
        "UI:          Streamlit (sidebar uploader + chat panel + sources panel).\n"
        "Config:      pydantic-settings reading .env.\n"
        "Logging:     structlog.",
    ),
    (
        "Key features",
        "- Streaming responses (SSE) with live token deltas in the UI.\n"
        "- Inline citations + a separate Sources panel showing scores and "
        "chunk snippets.\n"
        "- Session memory: follow-up questions are query-rewritten into "
        "standalone search queries using prior turns.\n"
        "- Dedupe-on-reingest: re-uploading the same file produces a net-zero "
        "change to the vector store.\n"
        "- Pluggable interfaces (LLM, Embedder, VectorStore are ABCs) -- "
        "switching providers is a one-file change.\n"
        "- Strict quality bar: mypy --strict clean, ruff clean, ~156 tests "
        "covering ingestion, retrieval, generation, and the HTTP API.",
    ),
    (
        "Why use RAG instead of a plain LLM?",
        "Plain LLMs hallucinate when asked about content they were not trained "
        "on. RAG grounds answers in a private document corpus you control, so "
        "the model can answer questions about *your* docs (policies, manuals, "
        "research papers, codebases) with verifiable citations. The system "
        "prompt is deliberately strict: the model is told to refuse rather "
        "than guess.",
    ),
    (
        "How to run (one-screen summary)",
        "1. uv sync --extra dev\n"
        "2. Copy .env.example to .env and set GEMINI_API_KEY (or "
        "ANTHROPIC_API_KEY) + LLM_PROVIDER.\n"
        "3. Terminal 1: uv run uvicorn app.main:app --reload\n"
        "4. Terminal 2: uv run streamlit run ui/streamlit_app.py\n"
        "5. Open http://localhost:8501, upload a document via the sidebar, "
        "ask questions.\n\n"
        "See QUICKSTART.md and README.md for the full guide.",
    ),
    (
        "Non-goals (v1)",
        "Authentication, multi-tenant isolation, fine-tuning, agentic tool use "
        "beyond retrieval, and Kubernetes deployment configs are deliberately "
        "out of scope. Docker packaging is fine to add; orchestration is not.",
    ),
]


class OverviewPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "RAG Chatbot -- Project Overview", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(110, 110, 110)
        self.cell(
            0,
            6,
            "Retrieval-Augmented Generation chatbot with citations, "
            "swappable providers, and a Streamlit UI.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def main() -> None:
    pdf = OverviewPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for heading, body in SECTIONS:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5.5, body)
        pdf.ln(3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
