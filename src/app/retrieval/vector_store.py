"""PGVector vector store manager.

Wraps ``langchain-postgres`` to provide a single point of access for
creating, connecting to, and querying the Neon PostgreSQL vector store.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_embeddings(settings: Settings) -> Embeddings:
    """Instantiate the embedding model from settings."""
    provider = settings.embedding_provider.lower()

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = settings.embedding_model
        if "text-embedding-004" in model:
            model = "models/gemini-embedding-001"

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=settings.get_api_key("gemini"),
            output_dimensionality=settings.embedding_dimensions,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.get_api_key("openai"),
        )

    raise ValueError(f"Unsupported embedding provider: '{provider}'")


class VectorStoreManager:
    """Manages the PGVector store lifecycle and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = _build_embeddings(self._settings)

        logger.info(
            "Connecting to PGVector (collection=%s)",
            self._settings.collection_name,
        )
        self._store = PGVector(
            embeddings=self._embeddings,
            collection_name=self._settings.collection_name,
            connection=self._settings.database_url,
            use_jsonb=True,
        )

    # ── Public API ───────────────────────────────────────────────

    @property
    def store(self) -> PGVector:
        """Raw PGVector store (for advanced usage)."""
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store *documents*, returning their IDs."""
        ids = self._store.add_documents(documents)
        logger.info("Stored %d chunks in PGVector", len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """Return the *k* most relevant chunks for *query*."""
        top_k = k or self._settings.retrieval_top_k
        return self._store.similarity_search(query, k=top_k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the *k* most relevant chunks with relevance scores."""
        top_k = k or self._settings.retrieval_top_k
        return self._store.similarity_search_with_score(query, k=top_k)

    def get_retriever(self, k: int | None = None):
        """Return a LangChain ``VectorStoreRetriever``."""
        top_k = k or self._settings.retrieval_top_k
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to *document_id*."""
        # PGVector supports filtering by metadata via the underlying store
        self._store.delete(filter={"document_id": document_id})
        logger.info("Deleted chunks for document_id=%s", document_id)

    def get_all_documents(self) -> list[Document]:
        """Retrieve every stored document chunk (for BM25 index rebuild).

        Queries the underlying PGVector collection table directly via SQLAlchemy,
        avoiding unnecessary and error-prone embedding requests.
        """
        try:
            from sqlalchemy import select

            documents: list[Document] = []
            with self._store._make_sync_session() as session:
                collection = self._store.get_collection(session)
                if not collection:
                    return []
                stmt = select(self._store.EmbeddingStore).filter(
                    self._store.EmbeddingStore.collection_id == collection.uuid
                )
                for record in session.execute(stmt).scalars().all():
                    documents.append(
                        Document(
                            id=record.id,
                            page_content=record.document,
                            metadata=record.cmetadata or {},
                        )
                    )
            return documents
        except Exception:
            logger.warning("Failed to load all documents for BM25 — returning empty")
            return []
