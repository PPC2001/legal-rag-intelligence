"""Redis-powered Semantic Cache for RAG responses.

Enables <10ms sub-millisecond responses for semantically similar repeated queries,
reducing LLM inference costs and vector DB load.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from app.config import Settings, get_settings
from app.schemas.question import RAGResponse, SourceReference

logger = logging.getLogger(__name__)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Stores and retrieves RAG query responses from Redis using vector similarity."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._enabled = self._settings.enable_semantic_cache
        self._threshold = self._settings.cache_similarity_threshold
        self._ttl = self._settings.cache_ttl_seconds
        self._redis = self._init_redis()

    def _init_redis(self) -> Any:
        """Connect to Redis if enabled and configured."""
        if not self._enabled:
            return None

        url = self._settings.redis_url
        if not url:
            logger.info("Semantic cache enabled but REDIS_URL not set; caching bypassed.")
            return None

        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            logger.info("Connected to Redis semantic cache at %s", url)
            return client
        except Exception as exc:
            logger.warning("Could not connect to Redis (%s). Semantic cache bypassed.", exc)
            return None

    @property
    def is_available(self) -> bool:
        """True if Redis client is connected and active."""
        return self._redis is not None

    def lookup(self, query_embedding: list[float]) -> RAGResponse | None:
        """Find cached response if a semantically equivalent query exists."""
        if not self.is_available or not query_embedding:
            return None

        try:
            # Scan cached keys (prefix: rag:cache:*)
            keys = self._redis.keys("rag:cache:*")
            if not keys:
                return None

            best_score = -1.0
            best_entry = None

            for key in keys:
                raw_data = self._redis.get(key)
                if not raw_data:
                    continue
                entry = json.loads(raw_data)
                cached_vec = entry.get("embedding")
                if cached_vec:
                    score = _cosine_similarity(query_embedding, cached_vec)
                    if score > best_score:
                        best_score = score
                        best_entry = entry

            if best_score >= self._threshold and best_entry:
                logger.info(
                    "Semantic Cache HIT (similarity=%.4f >= %.2f) for query: '%s'",
                    best_score,
                    self._threshold,
                    best_entry.get("question", ""),
                )
                sources = [
                    SourceReference(**src) for src in best_entry.get("sources", [])
                ]
                return RAGResponse(
                    answer=best_entry["answer"],
                    sources=sources,
                    sufficient_context=best_entry.get("sufficient_context", True),
                )
        except Exception as exc:
            logger.warning("Error querying semantic cache: %s", exc)

        return None

    def store(
        self,
        question: str,
        query_embedding: list[float],
        response: RAGResponse,
    ) -> None:
        """Cache a generated response in Redis."""
        if not self.is_available or not query_embedding:
            return

        try:
            import hashlib

            # Deterministic key based on question hash
            key_id = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
            redis_key = f"rag:cache:{key_id}"

            payload = {
                "question": question,
                "embedding": query_embedding,
                "answer": response.answer,
                "sources": [src.model_dump() for src in response.sources],
                "sufficient_context": response.sufficient_context,
            }

            self._redis.setex(
                redis_key,
                self._ttl,
                json.dumps(payload),
            )
            logger.info("Cached RAG response in Redis: %s (TTL=%ds)", redis_key, self._ttl)
        except Exception as exc:
            logger.warning("Error storing in semantic cache: %s", exc)


_cache_instance: SemanticCache | None = None


def get_semantic_cache(settings: Settings | None = None) -> SemanticCache:
    """Return a singleton instance of SemanticCache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(settings)
    return _cache_instance
