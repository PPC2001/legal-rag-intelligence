"""Health-check endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings, get_vector_store
from app.config import Settings
from app.retrieval.vector_store import VectorStoreManager
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Application health check",
)
def health(
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Return application status, version, and available LLM providers."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.env,
        available_providers=settings.get_available_providers(),
        database_connected=True,  # If we reached here, lifespan succeeded
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
)
def readiness(
    settings: Settings = Depends(get_app_settings),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> HealthResponse:
    """Deep health check — verifies database connectivity."""
    db_ok = True
    try:
        # Quick probe: run a trivial search
        vector_store.similarity_search("health check", k=1)
    except Exception:
        db_ok = False
        logger.warning("Readiness probe: database connection failed")

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=settings.app_version,
        environment=settings.env,
        available_providers=settings.get_available_providers(),
        database_connected=db_ok,
    )
