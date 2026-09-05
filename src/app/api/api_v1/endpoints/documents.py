"""Document ingestion endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.deps import (
    get_app_settings,
    get_bm25_retriever,
    get_ingestion_service,
    get_vector_store,
)
from app.config import Settings
from app.document_processing.ingestion import IngestionService
from app.document_processing.loaders import SUPPORTED_EXTENSIONS
from app.retrieval.bm25_retriever import BM25RetrieverService
from app.retrieval.vector_store import VectorStoreManager
from app.schemas import (
    AsyncIngestionResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestionResponse,
    TaskStatusResponse,
)
from app.storage.s3_storage import get_storage_service
from app.tasks.celery_app import celery_app
from app.tasks.ingestion_tasks import ingest_document_task

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_file(file: UploadFile, settings: Settings) -> str:
    """Validate file extension and size. Return the sanitised filename."""
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Filename is required.")

    # Sanitise filename — keep only the basename
    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    return filename


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def ingest_document(
    file: UploadFile,
    settings: Settings = Depends(get_app_settings),
    ingestion: IngestionService = Depends(get_ingestion_service),
    bm25: BM25RetrieverService = Depends(get_bm25_retriever),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> IngestionResponse:
    """Upload a PDF, DOCX, or TXT file and ingest it into the RAG pipeline.

    The document is chunked, embedded, and stored in PGVector.
    The BM25 index is rebuilt after ingestion.
    """
    filename = _validate_file(file, settings)

    # Ensure upload directory exists
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file to disk
    save_path = upload_dir / filename
    content = await file.read()

    # Check file size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    save_path.write_bytes(content)
    logger.info("Saved upload to %s (%d bytes)", save_path, len(content))

    try:
        # Ingest
        result = ingestion.ingest_file(save_path)

        # Rebuild BM25 index with all documents
        if result.status == "success":
            all_docs = vector_store.get_all_documents()
            bm25.update_documents(all_docs)

        return result

    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Ingestion failed: {exc}",
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents",
)
def list_documents(
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> DocumentListResponse:
    """Return metadata about all ingested documents."""
    all_docs = vector_store.get_all_documents()

    # Group by document_id
    doc_groups: dict[str, dict] = {}
    for doc in all_docs:
        doc_id = doc.metadata.get("document_id", "unknown")
        if doc_id not in doc_groups:
            doc_groups[doc_id] = {
                "document_id": doc_id,
                "filename": doc.metadata.get("source", "unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "chunk_count": 0,
                "ingested_at": "",
            }
        doc_groups[doc_id]["chunk_count"] += 1

    documents = [DocumentInfo(**info) for info in doc_groups.values()]
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
def delete_document(
    document_id: str,
    vector_store: VectorStoreManager = Depends(get_vector_store),
    bm25: BM25RetrieverService = Depends(get_bm25_retriever),
) -> None:
    """Remove a document and all its chunks from the store."""
    vector_store.delete_by_document_id(document_id)

    # Rebuild BM25 index
    all_docs = vector_store.get_all_documents()
    bm25.update_documents(all_docs)

    logger.info("Deleted document %s", document_id)


@router.post(
    "/upload-async",
    response_model=AsyncIngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for asynchronous background processing (1M scale)",
)
async def upload_document_async(
    file: UploadFile,
    settings: Settings = Depends(get_app_settings),
) -> AsyncIngestionResponse:
    """Upload a document to MinIO / local storage and queue background ingestion via Celery.

    Enables non-blocking, distributed processing of large documents and batch uploads.
    """
    filename = _validate_file(file, settings)
    content = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    # Upload to MinIO/S3 or local storage
    storage = get_storage_service(settings)
    storage_key = storage.upload_file(content, filename, file.content_type)

    # Dispatch Celery background task
    try:
        task = ingest_document_task.delay(storage_key, filename)
        task_id = task.id
    except Exception as exc:
        logger.warning("Celery dispatch failed: %s. Falling back to local ID.", exc)
        task_id = "local-sync-queue"

    logger.info(
        "Queued async ingestion for %s (task_id=%s, storage_key=%s)",
        filename,
        task_id,
        storage_key,
    )

    return AsyncIngestionResponse(
        task_id=task_id,
        filename=filename,
        storage_key=storage_key,
        status="PENDING",
        message="Document uploaded and queued for background ingestion.",
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Check status of an asynchronous ingestion task",
)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Query Celery result backend for current status of an ingestion task."""
    task = celery_app.AsyncResult(task_id)

    if task.failed():
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILURE",
            error=str(task.result),
        )

    if task.ready():
        return TaskStatusResponse(
            task_id=task_id,
            status="SUCCESS",
            result=task.result if isinstance(task.result, dict) else {"data": str(task.result)},
        )

    return TaskStatusResponse(
        task_id=task_id,
        status=task.status or "PENDING",
    )
