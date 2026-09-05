"""Langfuse open-source observability and tracing integration.

Captures LLM latencies, prompt inputs, token usage, and retrieval contexts
without impacting application performance.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_langfuse_callback(settings: Settings | None = None) -> Any | None:
    """Return a Langfuse CallbackHandler if configured, otherwise None."""
    cfg = settings or get_settings()

    public_key = cfg.langfuse_public_key
    secret_key = (
        cfg.langfuse_secret_key.get_secret_value()
        if cfg.langfuse_secret_key
        else None
    )

    if not public_key or not secret_key:
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=cfg.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (host=%s)", cfg.langfuse_host)
        return handler
    except Exception as exc:
        logger.warning("Could not initialize Langfuse callback handler: %s", exc)
        return None
