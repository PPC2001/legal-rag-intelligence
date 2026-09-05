"""Application settings powered by pydantic-settings (v2).

Settings are loaded from environment variables and `.env` file.
Supports environment profiles (DEV, PROD, LOCAL, TEST).
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported application runtime environments."""

    DEV = "DEV"
    PROD = "PROD"
    LOCAL = "LOCAL"
    TEST = "TEST"


class Settings(BaseSettings):
    """Centralised, validated application configuration.

    Values are resolved in order:
      1. Explicit environment variables
      2. `.env` file
      3. Field defaults defined here
    """

    # ── Application & Environment ────────────────────────────────
    env: Environment = Field(
        default=Environment.DEV, description="Runtime environment: DEV | PROD"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    app_name: str = Field(default="Legal RAG QA", description="Display name")
    app_version: str = Field(default="0.1.0", description="Semantic version")
    root_path: str = Field(
        default="",
        description="FastAPI root_path for reverse proxy / API Gateway",
    )

    # ── Database (Neon PostgreSQL + pgvector) ─────────────────────
    database_url: str = Field(
        ...,
        description="PostgreSQL connection string (Neon)",
    )
    collection_name: str = Field(
        default="legal_documents",
        description="PGVector collection / table name",
    )

    # ── LLM Configuration ────────────────────────────────────────
    default_llm_provider: str = Field(
        default="groq",
        description="Default LLM provider name",
    )
    default_llm_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="Default model identifier",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0 = deterministic)",
    )
    llm_max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens in LLM response",
    )

    # ── API Keys (all optional — system uses whichever is set) ───
    openai_api_key: SecretStr | None = Field(default=None)
    google_api_key: SecretStr | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)
    mistral_api_key: SecretStr | None = Field(default=None)

    # ── Embedding Configuration ──────────────────────────────────
    embedding_provider: str = Field(
        default="google",
        description="Embedding provider: google | openai",
    )
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Embedding model identifier",
    )
    embedding_dimensions: int = Field(
        default=768,
        gt=0,
        description="Embedding vector dimensionality",
    )

    # ── Retrieval Configuration ──────────────────────────────────
    chunk_size: int = Field(default=1000, gt=0, description="Chunk size (chars)")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    retrieval_top_k: int = Field(default=5, gt=0, description="Docs to retrieve")
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="BM25 fusion weight")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector fusion weight")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance to include a chunk",
    )

    # ── Server ───────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    uvicorn_reload: bool = Field(default=False, description="Enable auto-reload in Uvicorn")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # ── Upload & Storage ─────────────────────────────────────────
    upload_dir: str = Field(default="data/uploads", description="Local upload directory")
    max_file_size_mb: int = Field(default=50, gt=0, description="Max file size (MB)")
    s3_endpoint_url: str | None = Field(default=None, description="MinIO/S3 endpoint URL")
    s3_access_key: SecretStr | None = Field(default=None, description="MinIO/S3 access key")
    s3_secret_key: SecretStr | None = Field(default=None, description="MinIO/S3 secret key")
    s3_bucket_name: str = Field(default="legal-documents", description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="S3 region")

    # ── Scalability & Open-Source Stack (Redis, Celery, Rerank) ──
    redis_url: str | None = Field(default=None, description="Redis connection URL")
    celery_broker_url: str | None = Field(default=None, description="Celery broker URL")
    sparse_search_backend: str = Field(
        default="bm25",
        description="Sparse search backend: bm25 (in-memory) | postgres (native tsvector)",
    )
    enable_reranker: bool = Field(
        default=False,
        description="Enable FlashRank neural cross-encoder reranking",
    )
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank ONNX reranker model identifier",
    )
    reranker_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to retain after reranking",
    )
    enable_semantic_cache: bool = Field(
        default=False,
        description="Enable Redis semantic caching for repeated queries",
    )
    cache_similarity_threshold: float = Field(
        default=0.96,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=86400,
        gt=0,
        description="Semantic cache entry TTL in seconds",
    )

    # ── Observability (Langfuse) ─────────────────────────────────
    langfuse_public_key: str | None = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: SecretStr | None = Field(default=None, description="Langfuse secret key")
    langfuse_host: str = Field(default="http://localhost:3000", description="Langfuse host URL")

    # ── Pydantic‑settings wiring ─────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: Any) -> Any:
        """Ensure PostgreSQL connection strings use psycopg (v3) driver dialect.

        Neon and standard Postgres URLs start with 'postgresql://' or 'postgres://',
        which SQLAlchemy defaults to the 'psycopg2' driver. Since this project uses
        psycopg v3, normalize to 'postgresql+psycopg://'.
        """
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("postgresql://"):
                return "postgresql+psycopg://" + value[len("postgresql://") :]
            if value.startswith("postgres://"):
                return "postgresql+psycopg://" + value[len("postgres://") :]
        return value

    @field_validator("env", mode="before")
    @classmethod
    def normalize_env(cls, value: Any) -> Any:
        """Normalize env string to upper-case enum value."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, value: Any) -> list[str]:
        """Support both JSON list '["http://..."]' and comma-separated string 'http://a,http://b'."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        """Ensure log level is uppercase and valid."""
        if isinstance(value, str):
            val = value.strip().upper()
            allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if val not in allowed:
                raise ValueError(f"Invalid log_level '{value}'. Allowed: {sorted(allowed)}")
            return val
        return value

    @model_validator(mode="after")
    def set_environment_defaults(self) -> Settings:
        """Derive reload defaults based on environment if not explicitly set."""
        if self.env in (Environment.DEV, Environment.LOCAL):
            if not self.uvicorn_reload:
                self.uvicorn_reload = True
        elif self.env == Environment.PROD:
            self.debug = False
            self.uvicorn_reload = False
        return self

    # ── Helpers ──────────────────────────────────────────────────

    def get_available_providers(self) -> list[str]:
        """Return the list of LLM providers whose API key is configured."""
        mapping: dict[str, SecretStr | None] = {
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "mistral": self.mistral_api_key,
        }
        return [name for name, key in mapping.items() if key]

    def get_api_key(self, provider: str) -> str:
        """Resolve the secret string for a provider, or raise."""
        mapping: dict[str, SecretStr | None] = {
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "mistral": self.mistral_api_key,
        }
        key = mapping.get(provider)
        if not key:
            raise ValueError(
                f"API key for provider '{provider}' is not configured. "
                f"Set the corresponding environment variable."
            )
        return key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
