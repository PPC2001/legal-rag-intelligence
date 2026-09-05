"""PostgreSQL native full-text retriever using GIN indexed tsvector.

Replaces in-memory BM25 to allow keyword search over 1,000,000+ document chunks
without storing inverted indexes in application RAM.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import text

from app.config import Settings, get_settings
from app.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class PostgresFullTextRetriever:
    """Performs PostgreSQL native full-text keyword retrieval using tsvector."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings or get_settings()

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Execute a full-text search query using PostgreSQL's tsvector and plainto_tsquery.

        Args:
            query: Plain-text search query.
            k: Maximum number of keyword chunks to return.

        Returns:
            List of matching Document objects ranked by ts_rank_cd.
        """
        top_k = k or self._settings.retrieval_top_k
        if not query.strip():
            return []

        documents: list[Document] = []

        try:
            with self._vector_store.store._make_sync_session() as session:
                collection = self._vector_store.store.get_collection(session)
                collection_id = collection.uuid if collection else None

                # Query using the GIN-indexed tsvector expression or generated column
                sql = text("""
                    SELECT document, cmetadata,
                           ts_rank_cd(
                               to_tsvector('english', document),
                               plainto_tsquery('english', :query)
                           ) AS rank
                    FROM langchain_pg_embedding
                    WHERE (CAST(:collection_id AS uuid) IS NULL OR collection_id = :collection_id)
                      AND (to_tsvector('english', document) @@ plainto_tsquery('english', :query))
                    ORDER BY rank DESC
                    LIMIT :top_k
                """)

                result = session.execute(
                    sql,
                    {
                        "query": query,
                        "collection_id": collection_id,
                        "top_k": top_k,
                    },
                )

                for row in result:
                    metadata: dict[str, Any] = row.cmetadata or {}
                    metadata["fulltext_rank"] = float(row.rank)
                    documents.append(
                        Document(
                            page_content=row.document,
                            metadata=metadata,
                        )
                    )

            logger.info("Postgres FullText retrieved %d chunks for query", len(documents))
            return documents
        except Exception as exc:
            logger.warning(
                "Postgres FullText search error: %s. Falling back to empty list.", exc
            )
            return []
