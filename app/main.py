"""FastAPI application entrypoint."""

from __future__ import annotations

import anthropic
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.logging import configure_logging

_settings_at_boot = get_settings()
configure_logging(_settings_at_boot.log_level)
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot with citations.",
    version="0.1.0",
)


@app.exception_handler(anthropic.APIError)
async def _anthropic_error_handler(
    request: Request, exc: anthropic.APIError
) -> JSONResponse:
    """Translate Anthropic SDK errors into clean HTTP responses.

    Auth/permission/not-found errors come back as 4xx; everything else is a 502
    (bad gateway to the upstream LLM). The Anthropic request_id is preserved
    when present, so support tickets can correlate.
    """
    status = 502
    if isinstance(exc, anthropic.AuthenticationError):
        status = 401
    elif isinstance(exc, anthropic.PermissionDeniedError):
        status = 403
    elif isinstance(exc, anthropic.NotFoundError):
        status = 404
    elif isinstance(exc, anthropic.BadRequestError):
        status = 400
    elif isinstance(exc, anthropic.RateLimitError):
        status = 429

    request_id = getattr(exc, "request_id", None) or getattr(
        getattr(exc, "response", None), "headers", {}
    ).get("request-id")

    logger.warning(
        "anthropic_error",
        path=str(request.url.path),
        status=status,
        error_type=type(exc).__name__,
        request_id=request_id,
        message=str(exc),
    )
    return JSONResponse(
        status_code=status,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "anthropic_request_id": request_id,
        },
    )


app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns the configured LLM provider and model for sanity checks."""
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
    }
