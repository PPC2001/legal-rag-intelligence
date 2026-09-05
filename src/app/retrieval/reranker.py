"""FlashRank Neural Cross-Encoder Reranker.

Provides lightweight, ultra-fast CPU-based neural reranking of retrieved document chunks
using ONNX runtime (<100MB RAM, zero GPU requirement).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Reranks candidate document chunks using FlashRank cross-encoder models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.reranker_model
        self._top_k = self._settings.reranker_top_k
        self._ranker = self._init_ranker()

    def _init_ranker(self) -> Any:
        """Initialize FlashRank Ranker instance."""
        try:
            from flashrank import Ranker

            logger.info("Initializing FlashRank neural reranker model: %s", self._model_name)
            return Ranker(model_name=self._model_name, cache_dir="/tmp/flashrank")
        except Exception as exc:
            logger.warning("Could not initialize FlashRank (%s). Reranking will be bypassed.", exc)
            return None

    @property
    def is_available(self) -> bool:
        """True if the FlashRank model is initialized and ready."""
        return self._ranker is not None

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Rerank *documents* against *query* using cross-attention scoring.

        Args:
            query: User question string.
            documents: Candidate documents from dense/sparse retrieval.
            top_k: Number of highest-scoring documents to return.

        Returns:
            Sorted top-k Document objects with 'rerank_score' in metadata.
        """
        if not documents:
            return []

        k = top_k or self._top_k

        if not self.is_available:
            return documents[:k]

        try:
            from flashrank import RerankRequest

            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(documents)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = self._ranker.rerank(rerank_request)

            reranked_docs: list[Document] = []
            for item in ranked_results[:k]:
                doc_idx = item["id"]
                orig_doc = documents[doc_idx]
                # Preserve original document and attach rerank score
                new_metadata = dict(orig_doc.metadata)
                new_metadata["rerank_score"] = float(item.get("score", 0.0))
                reranked_docs.append(
                    Document(
                        page_content=orig_doc.page_content,
                        metadata=new_metadata,
                    )
                )

            logger.info(
                "Reranked %d candidates down to %d using FlashRank (%s)",
                len(documents),
                len(reranked_docs),
                self._model_name,
            )
            return reranked_docs
        except Exception as exc:
            logger.error(
                "Error during FlashRank reranking: %s. Returning unranked candidates.", exc
            )
            return documents[:k]
