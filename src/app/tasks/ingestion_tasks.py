"""Celery tasks for background document ingestion and chunking."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.document_processing.ingestion import IngestionService
from app.retrieval.vector_store import VectorStoreManager
from app.storage.s3_storage import get_storage_service
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.ingest_document_task")
def ingest_document_task(self, file_key_or_path: str, filename: str) -> dict:
    """Asynchronously chunk, embed, and store a document in PGVector."""
    logger.info("Starting async ingestion task %s for file %s", self.request.id, filename)

    storage = get_storage_service()
    vector_store = VectorStoreManager()
    ingestion_service = IngestionService(vector_store)

    # 1. Fetch file bytes from MinIO/S3 or local storage
    file_bytes = storage.get_file_bytes(file_key_or_path)

    # 2. Write to secure temp file for loader processing
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        # 3. Ingest document with original filename
        response = ingestion_service.ingest_file(tmp_path, original_filename=filename)
        logger.info(
            "Completed task %s: %s (%d chunks, status=%s)",
            self.request.id,
            filename,
            response.chunks_created,
            response.status,
        )
        if response.status == "error":
            raise RuntimeError(f"Ingestion failed: {response.message}")

        return {
            "task_id": self.request.id,
            "document_id": response.document_id,
            "filename": filename,
            "chunks_created": response.chunks_created,
            "status": response.status,
            "message": response.message,
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
