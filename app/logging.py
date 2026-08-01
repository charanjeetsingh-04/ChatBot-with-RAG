"""Structured logging via structlog.

Call `configure_logging(level)` once at process startup. Module code obtains a
logger with `logger = structlog.get_logger(__name__)` and emits events as
keyword arguments:

    logger.info("ingested", file_path=str(path), chunks_added=n)

Output is human-readable on a TTY (colors, key=value) and plain key=value
otherwise. Swap `ConsoleRenderer` for `JSONRenderer` in production if you want
structured logs going to a log aggregator.
"""

from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog and the stdlib logger. Idempotent within a process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True
