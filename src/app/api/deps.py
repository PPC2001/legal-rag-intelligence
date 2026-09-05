"""FastAPI dependency injection.

All heavy objects (vector store, BM25 index, RAG chain, ingestion
service) are created once during the lifespan and stored in
``app.state``. Dependencies retrieve them from there.
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings, get_settings
from app.document_processing.ingestion import IngestionService
from app.rag.chain import RAGChain
from app.retrieval.bm25_retriever import BM25RetrieverService
from app.retrieval.vector_store import VectorStoreManager


def get_app_settings() -> Settings:
    """Return the cached application settings."""
    return get_settings()


def get_vector_store(request: Request) -> VectorStoreManager:
    """Retrieve the vector store manager from app state."""
    return request.app.state.vector_store


def get_bm25_retriever(request: Request) -> BM25RetrieverService:
    """Retrieve the BM25 retriever from app state."""
    return request.app.state.bm25_retriever


def get_rag_chain(request: Request) -> RAGChain:
    """Retrieve the RAG chain from app state."""
    return request.app.state.rag_chain


def get_ingestion_service(request: Request) -> IngestionService:
    """Retrieve the ingestion service from app state."""
    return request.app.state.ingestion_service
