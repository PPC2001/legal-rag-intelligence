"""Legal RAG QA — Application entry point and factory.

Inspired by standard enterprise FastAPI structure:
  - Application factory: create_app()
  - Middleware: CORS, GZip, X-Request-ID
  - Lifespan: DB connection, BM25 indexing, hybrid retriever setup
  - Routers: health, documents, question answering
  - Direct execution: python -m app.main
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.document_processing.ingestion import IngestionService
from app.rag.chain import RAGChain
from app.retrieval.bm25_retriever import BM25RetrieverService
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# ── Lifespan (Startup / Shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database, retrievers, and RAG chain on startup."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.info(
        "Starting %s v%s [env=%s, debug=%s]",
        settings.app_name,
        settings.app_version,
        settings.env,
        settings.debug,
    )

    # 1. Vector Store (Neon PGVector)
    vector_store = VectorStoreManager(settings)
    app.state.vector_store = vector_store

    # 2. BM25 Keyword Search
    all_docs = vector_store.get_all_documents()
    bm25 = BM25RetrieverService(documents=all_docs, k=settings.retrieval_top_k)
    app.state.bm25_retriever = bm25

    # 3. Hybrid Retriever (RRF)
    hybrid = HybridRetriever(vector_store, bm25)

    # 4. RAG Chain
    app.state.rag_chain = RAGChain(hybrid, settings)

    # 5. Ingestion Service
    app.state.ingestion_service = IngestionService(vector_store)

    logger.info("Available LLM providers: %s", settings.get_available_providers() or "(none)")
    logger.info("Startup complete — ready to serve requests")

    yield

    logger.info("Shutting down %s", settings.app_name)


# ── Application Factory ──────────────────────────────────────────
def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade RAG system for legal, HR, and insurance "
            "document intelligence. Ask natural-language questions and "
            "receive accurate, cited answers grounded in source documents."
        ),
        lifespan=lifespan,
        root_path=settings.root_path,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. GZip Compression (compresses large legal text / JSON responses)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # 3. Request-ID Tracking
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # 4. Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal error occurred. Please try again.",
                "type": type(exc).__name__,
            },
        )

    # 5. Include API Routers
    from app.api.api_v1.api import api_router

    app.include_router(api_router)

    return app


def main() -> None:
    """CLI / programmatic runner."""
    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.uvicorn_reload,
        log_level=settings.log_level.lower(),
    )


# ── CLI Runner ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
