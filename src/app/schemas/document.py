"""Document and Ingestion schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    """Result of a document ingestion operation."""

    document_id: str = Field(description="Unique identifier for the ingested document")
    filename: str = Field(description="Original filename")
    chunks_created: int = Field(description="Number of chunks produced")
    status: str = Field(description="Ingestion status: success | error")
    message: str = Field(default="", description="Additional detail")


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""

    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    file_type: str = Field(description="File extension")
    chunk_count: int = Field(description="Number of chunks")
    ingested_at: str = Field(description="ISO-8601 ingestion timestamp")


class DocumentListResponse(BaseModel):
    """Response for listing ingested documents."""

    documents: list[DocumentInfo] = Field(description="All ingested documents")
    total: int = Field(description="Total count")


class AsyncIngestionResponse(BaseModel):
    """Response when a document upload is queued for background processing."""

    task_id: str = Field(description="Celery background task ID")
    filename: str = Field(description="Uploaded filename")
    storage_key: str = Field(description="MinIO/S3 storage key or local path")
    status: str = Field(description="Task status: PENDING")
    message: str = Field(description="User message")


class TaskStatusResponse(BaseModel):
    """Status of an asynchronous ingestion task."""

    task_id: str = Field(description="Celery background task ID")
    status: str = Field(description="Status: PENDING | STARTED | SUCCESS | FAILURE")
    result: dict | None = Field(default=None, description="Result payload if finished")
    error: str | None = Field(default=None, description="Error message if failed")

