"""Unit tests for the 1M-scale open-source components:

MinIO/S3 Storage, FlashRank Reranker, Semantic Cache, and Celery Tasks.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.api_v1.api import api_router
from app.cache.semantic_cache import SemanticCache, _cosine_similarity
from app.config import Settings, get_settings
from app.retrieval.reranker import FlashRankReranker
from app.storage.s3_storage import S3StorageService


@pytest.fixture()
def app():
    """Create a minimal test FastAPI app."""
    @asynccontextmanager
    async def _test_lifespan(app: FastAPI):
        yield

    settings = get_settings()
    application = FastAPI(
        title="Legal RAG QA Test",
        version=settings.app_version,
        lifespan=_test_lifespan,
    )
    application.include_router(api_router)
    return application


def test_cosine_similarity():
    """Verify vector cosine similarity calculation."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [0.0, 1.0, 0.0]

    assert round(_cosine_similarity(vec1, vec2), 4) == 1.0
    assert round(_cosine_similarity(vec1, vec3), 4) == 0.0
    assert _cosine_similarity([], []) == 0.0


def test_storage_local_fallback(tmp_path: Path):
    """S3StorageService should fallback to local filesystem when S3 is unconfigured."""
    settings = Settings(
        database_url="postgresql://user:pass@localhost:5432/test",
        upload_dir=str(tmp_path),
        s3_endpoint_url=None,
        s3_access_key=None,
    )
    storage = S3StorageService(settings)
    assert not storage.is_s3_enabled

    file_bytes = b"Hello, this is a test contract."
    storage_key = storage.upload_file(file_bytes, "test_contract.txt")

    assert Path(storage_key).exists()
    assert storage.get_file_bytes(storage_key) == file_bytes
    assert storage.delete_file(storage_key)
    assert not Path(storage_key).exists()


def test_reranker_fallback():
    """FlashRankReranker should gracefully return top-k candidates even if model is not loaded."""
    settings = Settings(
        database_url="postgresql://user:pass@localhost:5432/test",
        enable_reranker=False,
    )
    reranker = FlashRankReranker(settings)
    docs = [
        Document(page_content="Section 1: General provisions"),
        Document(page_content="Section 2: Payment terms"),
        Document(page_content="Section 3: Termination clause"),
    ]

    reranked = reranker.rerank("What are the payment terms?", docs, top_k=2)
    assert len(reranked) == 2


def test_semantic_cache_offline():
    """SemanticCache should gracefully act as no-op when Redis is unconfigured or offline."""
    settings = Settings(
        database_url="postgresql://user:pass@localhost:5432/test",
        enable_semantic_cache=True,
        redis_url=None,
    )
    cache = SemanticCache(settings)
    assert not cache.is_available
    assert cache.lookup([0.1, 0.2, 0.3]) is None


def test_async_upload_endpoint(app, tmp_path: Path):
    """POST /api/v1/documents/upload-async should queue upload and return 202."""
    client = TestClient(app)

    file_content = b"Sample legal clause for async testing."
    test_file = io.BytesIO(file_content)

    with patch("app.tasks.ingestion_tasks.ingest_document_task.delay") as mock_task:
        mock_task.return_value = MagicMock(id="test-task-12345")

        response = client.post(
            "/api/v1/documents/upload-async",
            files={"file": ("sample.txt", test_file, "text/plain")},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == "test-task-12345"
        assert data["filename"] == "sample.txt"
        assert data["status"] == "PENDING"
