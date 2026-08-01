"""Application configuration loaded from environment / .env via pydantic-settings.

All knobs the rest of the app needs live here. Swap providers, models, chunking,
or retrieval parameters by editing `.env` — no code changes required.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "gemini", "openai"]
EmbeddingProvider = Literal["sentence-transformers", "openai"]


class Settings(BaseSettings):
    """Typed application settings.

    Values are loaded from environment variables, falling back to `.env` in the
    project root. See `.env.example` for the full list of supported keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Provider API keys ----
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key.")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key.")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key (optional).")

    # ---- LLM ----
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = Field(default=1024, ge=1, le=64_000)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # ---- Embeddings ----
    embedding_provider: EmbeddingProvider = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ---- Vector store ----
    vector_store_path: Path = Path("./.chroma")
    vector_store_collection: str = "rag_chunks"

    # ---- Uploads ----
    uploads_path: Path = Path("./uploads")

    # ---- Chunking ----
    chunk_size: int = Field(default=800, ge=100, le=8_000)
    chunk_overlap: int = Field(default=120, ge=0, le=2_000)

    # ---- Retrieval ----
    top_k: int = Field(default=5, ge=1, le=50)
    hybrid_search: bool = False
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion constant.")

    # ---- API ----
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65_535)

    # ---- Logging ----
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info: object) -> int:
        """Ensure chunk overlap is strictly smaller than chunk size."""
        size = getattr(info, "data", {}).get("chunk_size", 800)
        if v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({size})")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached `Settings` instance."""
    return Settings()
