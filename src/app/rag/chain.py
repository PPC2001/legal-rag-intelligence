"""Core RAG chain: retrieve → format → generate → cite.

Orchestrates hybrid retrieval and LLM generation with a hallucination
guard that refuses to answer when retrieved context is insufficient.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.cache import get_semantic_cache
from app.config import Settings, get_settings
from app.llm.factory import LLMFactory
from app.llm.observability import get_langfuse_callback
from app.llm.prompts import QA_PROMPT
from app.retrieval.hybrid import HybridRetriever
from app.schemas import QuestionRequest, RAGResponse, SourceReference

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────

_REFUSAL = "I cannot find this information in the provided documents."


def _format_context(documents: list[Document]) -> str:
    """Build a numbered context string with source annotations."""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        page_label = f", Page {page}" if page else ""
        parts.append(f"[{i}] (Source: {source}{page_label})\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_sources(documents: list[Document]) -> list[SourceReference]:
    """Extract deduplicated source references from retrieved documents."""
    seen: set[str] = set()
    sources: list[SourceReference] = []

    for doc in documents:
        doc_source = doc.metadata.get("source", "unknown")
        doc_page = doc.metadata.get("page", 0)
        key = f"{doc_source}:{doc_page}"

        if key not in seen:
            seen.add(key)
            snippet = doc.page_content[:500].strip()
            score = float(doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0.0)))
            sources.append(
                SourceReference(
                    document=doc_source,
                    page=doc_page,
                    chunk_text=snippet,
                    relevance_score=round(score, 4),
                )
            )

    return sources


# ── RAG Chain ────────────────────────────────────────────────────


class RAGChain:
    """End-to-end RAG pipeline with hybrid retrieval, semantic caching, and strict citation."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = hybrid_retriever
        self._settings = settings or get_settings()
        self._parser = StrOutputParser()
        self._cache = get_semantic_cache(self._settings)

    def ask(self, request: QuestionRequest) -> RAGResponse:
        """Answer a question with cited sources.

        Pipeline:
          0. Semantic cache check (returns in <10ms if hit)
          1. Hybrid retrieval (vector + sparse via RRF + FlashRank)
          2. Hallucination guard (refuse if no relevant context)
          3. LLM generation with grounding prompt and Langfuse tracing
          4. Package response and cache result
        """
        start = time.perf_counter()

        # ── 0. Semantic cache check ──────────────────────────────
        cache = getattr(self, "_cache", None)
        query_vec: list[float] | None = None
        if cache and cache.is_available:
            try:
                query_vec = self._retriever._vector_store._embeddings.embed_query(
                    request.question
                )
                cached = cache.lookup(query_vec)
                if cached:
                    logger.info("Serving answer from Redis Semantic Cache")
                    return cached
            except Exception as exc:
                logger.debug("Cache lookup failed: %s", exc)

        # ── 1. Retrieve ──────────────────────────────────────────
        top_k = self._settings.retrieval_top_k
        retrieved_docs = self._retriever.retrieve(request.question, top_k=top_k)

        # ── 2. Hallucination guard ───────────────────────────────
        if not retrieved_docs:
            return RAGResponse(
                answer=_REFUSAL,
                sources=[],
                sufficient_context=False,
            )

        # ── 3. Generate ─────────────────────────────────────────
        context_str = _format_context(retrieved_docs)

        llm = LLMFactory.create(
            provider=self._settings.default_llm_provider,
            model=self._settings.default_llm_model,
            settings=self._settings,
        )

        langfuse_cb = get_langfuse_callback(self._settings)
        invoke_config = {"callbacks": [langfuse_cb]} if langfuse_cb else {}

        chain = QA_PROMPT | llm | self._parser
        answer: str = chain.invoke(
            {"context": context_str, "question": request.question},
            config=invoke_config,
        )

        # ── 4. Package response & Cache ─────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        sources = _build_sources(retrieved_docs)

        response = RAGResponse(
            answer=answer,
            sources=sources,
            sufficient_context=True,
        )

        if cache and cache.is_available and query_vec:
            cache.store(request.question, query_vec, response)

        logger.info(
            "Answered in %.0f ms (provider=%s, sources=%d)",
            elapsed_ms,
            self._settings.default_llm_provider,
            len(sources),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            sufficient_context=True,
        )

    async def ask_stream(
        self,
        request: QuestionRequest,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token via SSE.

        Yields individual string tokens as they arrive from the LLM.
        """
        top_k = self._settings.retrieval_top_k
        retrieved_docs = self._retriever.retrieve(request.question, top_k=top_k)

        if not retrieved_docs:
            yield _REFUSAL
            return

        context_str = _format_context(retrieved_docs)

        llm = LLMFactory.create(
            provider=self._settings.default_llm_provider,
            model=self._settings.default_llm_model,
            settings=self._settings,
        )

        chain = QA_PROMPT | llm | self._parser
        async for token in chain.astream({"context": context_str, "question": request.question}):
            yield token
