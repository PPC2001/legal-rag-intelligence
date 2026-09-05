"""Celery application configuration for asynchronous document ingestion."""

from __future__ import annotations

import os

from celery import Celery

from app.config import get_settings

settings = get_settings()

broker_url = settings.celery_broker_url or settings.redis_url or os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend_url = settings.redis_url or os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "legal_rag_tasks",
    broker=broker_url,
    backend=backend_url,
    include=["app.tasks.ingestion_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max for huge documents
    worker_prefetch_multiplier=1,
)
