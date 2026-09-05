"""Background async task processing package powered by Celery & Redis."""

from app.tasks.celery_app import celery_app

__all__ = ["celery_app"]
