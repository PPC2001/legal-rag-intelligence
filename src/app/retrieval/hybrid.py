"""Hybrid retriever combining dense vector and sparse BM25 search.

Uses **Reciprocal Rank Fusion (RRF)** to merge ranked result lists
without needing to normalise heterogeneous score distributions.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.retrieval.bm25_retriever import BM25RetrieverService
from app.retrieval.postgres_fulltext import PostgresFullTextRetriever
from app.retrieval.reranker import FlashRankReranker
from app.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Standard RRF constant (matches Elasticsearch / Qdrant defaults).
_RRF_K: int = 60


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k: int = _RRF_K,
) -> list[Document]:
    """Merge multiple ranked document lists via weighted RRF.

    For each document, the fused score is:
        score = Σ  weight_i / (k + rank_i)
    where rank_i is the 1-based position in list i.
    """
    # Track scores keyed by page_content (content-based dedup).
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list, weight in zip(ranked_lists, weights, strict=False):
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            rrf_score = weight / (k + rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score
            # Keep the version with the richest metadata
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)  # type: ignore[arg-type]

    results: list[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = fused_scores[key]
        results.append(doc)

    return results


class HybridRetriever:
    """Combines PGVector (semantic) and Sparse (BM25 or PostgreSQL tsvector) retrieval via RRF,

    with optional FlashRank neural cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        bm25_retriever: BM25RetrieverService,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_retriever
        self._settings = settings or get_settings()
        self._vector_weight = self._settings.vector_weight
        self._bm25_weight = self._settings.bm25_weight
        self._top_k = self._settings.retrieval_top_k
        self._postgres_fulltext = PostgresFullTextRetriever(
            self._vector_store, self._settings
        )
        self._reranker = (
            FlashRankReranker(self._settings)
            if self._settings.enable_reranker
            else None
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """Run hybrid retrieval and return fused, reranked results.

        Steps:
          1. Dense vector search via PGVector
          2. Sparse search via PostgreSQL tsvector or in-memory BM25
          3. Merge with Reciprocal Rank Fusion (RRF)
          4. Optional neural reranking via FlashRank
        """
        k = top_k or self._top_k
        # Fetch more candidates if reranking is enabled
        fetch_k = max(k * 3, 15) if (self._reranker and self._reranker.is_available) else k

        # 1. Vector search
        vector_results = self._vector_store.similarity_search(query, k=fetch_k)
        logger.debug("Vector search returned %d results", len(vector_results))

        # 2. Sparse search (PostgreSQL tsvector or in-memory BM25)
        if self._settings.sparse_search_backend.lower() == "postgres":
            sparse_results = self._postgres_fulltext.search(query, k=fetch_k)
            logger.debug("PostgreSQL tsvector search returned %d results", len(sparse_results))
        else:
            self._bm25.k = fetch_k
            sparse_results = self._bm25.invoke(query)
            logger.debug("BM25 search returned %d results", len(sparse_results))

        # 3. RRF fusion
        fused = _reciprocal_rank_fusion(
            ranked_lists=[vector_results, sparse_results],
            weights=[self._vector_weight, self._bm25_weight],
        )

        # 4. Optional FlashRank neural reranking
        if self._reranker and self._reranker.is_available:
            final = self._reranker.rerank(query, fused, top_k=k)
        else:
            final = fused[:k]

        logger.info(
            "Hybrid retrieval: %d vector + %d sparse → %d fused → %d final results",
            len(vector_results),
            len(sparse_results),
            len(fused),
            len(final),
        )
        return final
