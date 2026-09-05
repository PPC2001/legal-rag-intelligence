"""Tests for API endpoints (using FastAPI TestClient with mocked services)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


@pytest.fixture()
def app():
    """Create a test FastAPI app with mock state set directly."""

    @asynccontextmanager
    async def _test_lifespan(app: FastAPI):
        yield

    from app.api.api_v1.api import api_router
    from app.config import get_settings

    settings = get_settings()

    application = FastAPI(
        title="Legal RAG QA Test",
        version=settings.app_version,
        lifespan=_test_lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    # Set mock state BEFORE creating TestClient (state persists across lifespan)
    mock_vs = MagicMock()
    mock_vs.get_all_documents.return_value = []
    mock_vs.similarity_search.return_value = []
    application.state.vector_store = mock_vs

    application.state.bm25_retriever = MagicMock()
    application.state.rag_chain = MagicMock()
    application.state.ingestion_service = MagicMock()

    return application


@pytest.fixture()
def client(app):
    """FastAPI TestClient."""
    return TestClient(app)


class TestHealthEndpoints:
    """Verify health and readiness probes."""

    def test_health_returns_ok(self, client: TestClient):
        """GET /api/v1/health should return 200 with status ok."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data

    def test_readiness_returns_status(self, client: TestClient):
        """GET /api/v1/health/ready should return database status."""
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "database_connected" in data


class TestQuestionEndpoints:
    """Verify the /ask endpoint."""

    def test_ask_validation_short_question(self, client: TestClient):
        """Questions shorter than 3 chars should fail validation."""
        resp = client.post("/api/v1/ask", json={"question": "hi"})
        assert resp.status_code == 422

    def test_ask_validation_missing_question(self, client: TestClient):
        """Missing question field should fail validation."""
        resp = client.post("/api/v1/ask", json={})
        assert resp.status_code == 422


class TestDocumentEndpoints:
    """Verify document management endpoints."""

    def test_list_documents_empty(self, client: TestClient):
        """Empty store should return an empty document list."""
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_upload_unsupported_type(self, client: TestClient, tmp_path):
        """Uploading an unsupported file type should return 415."""
        bad_file = tmp_path / "test.md"
        bad_file.write_text("markdown content")
        with open(bad_file, "rb") as f:
            resp = client.post(
                "/api/v1/documents/ingest",
                files={"file": ("test.md", f, "text/markdown")},
            )
        assert resp.status_code == 415

    def test_create_app_factory(self):
        """create_app should instantiate a FastAPI app with proper metadata."""
        from app.main import create_app

        application = create_app()
        assert application.title == "Legal RAG QA"
        openapi_paths = application.openapi()["paths"]
        assert "/api/v1/health" in openapi_paths
        assert "/api/v1/documents" in openapi_paths
        assert "/api/v1/ask" in openapi_paths
