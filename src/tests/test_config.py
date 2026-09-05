"""Tests for configuration and settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import Environment, Settings


class TestSettings:
    """Verify settings load, validate, and resolve providers correctly."""

    def test_settings_load_from_env(self, settings):
        """Settings should load the test values we injected."""
        assert settings.env == Environment.DEV
        assert settings.env == "DEV"
        assert settings.debug is True
        assert settings.database_url == "postgresql+psycopg://test:test@localhost/test_db"
        assert settings.collection_name == "test_collection"
        assert settings.chunk_size == 500

    def test_get_available_providers(self, settings):
        """Only providers with keys should be listed."""
        providers = settings.get_available_providers()
        assert "gemini" in providers
        # openai key was not set
        assert "openai" not in providers

    def test_get_api_key_valid(self, settings):
        """get_api_key should return the secret value."""
        key = settings.get_api_key("gemini")
        assert key == "test-google-key"

    def test_get_api_key_missing_raises(self, settings):
        """get_api_key should raise for unconfigured providers."""
        with pytest.raises(ValueError, match="API key for provider 'openai'"):
            settings.get_api_key("openai")

    def test_validation_constraints(self):
        """Temperature and max_tokens should enforce bounds."""
        with pytest.raises(ValidationError):
            Settings(
                database_url="postgresql://x:x@localhost/db",
                llm_temperature=5.0,  # > 2.0
            )

    def test_env_normalization(self):
        """Lower-case env strings should normalize to upper-case enum."""
        dev_settings = Settings(
            database_url="postgresql://x:x@localhost/db",
            env="dev",
        )
        assert dev_settings.env == Environment.DEV
        assert dev_settings.env == "DEV"
        assert dev_settings.uvicorn_reload is True

        prod_settings = Settings(
            database_url="postgresql://x:x@localhost/db",
            env="prod",
        )
        assert prod_settings.env == Environment.PROD
        assert prod_settings.env == "PROD"
        assert prod_settings.debug is False
        assert prod_settings.uvicorn_reload is False

    def test_cors_origins_parsing(self):
        """Should support comma-separated strings as well as JSON lists."""
        s = Settings(
            database_url="postgresql://x:x@localhost/db",
            cors_origins="http://localhost:3000, http://localhost:8000",
        )
        assert s.cors_origins == ["http://localhost:3000", "http://localhost:8000"]

    def test_log_level_normalization(self):
        """Log level should be capitalized and validated."""
        s = Settings(
            database_url="postgresql://x:x@localhost/db",
            log_level="debug",
        )
        assert s.log_level == "DEBUG"

        with pytest.raises(ValidationError):
            Settings(
                database_url="postgresql://x:x@localhost/db",
                log_level="INVALID_LEVEL",
            )

    def test_database_url_normalization(self):
        """Standard postgresql:// and postgres:// URLs should normalize to postgresql+psycopg://."""
        s1 = Settings(database_url="postgresql://user:pass@host/db")
        assert s1.database_url == "postgresql+psycopg://user:pass@host/db"

        s2 = Settings(database_url="postgres://user:pass@host/db")
        assert s2.database_url == "postgresql+psycopg://user:pass@host/db"

        s3 = Settings(database_url="postgresql+psycopg://user:pass@host/db")
        assert s3.database_url == "postgresql+psycopg://user:pass@host/db"

