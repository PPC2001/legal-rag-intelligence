"""API v1 router aggregator.

Mounts all v1 endpoints under the ``/api/v1`` prefix.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.api_v1.endpoints import documents, health, question

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(question.router, tags=["question-answering"])
