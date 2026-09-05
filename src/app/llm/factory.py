"""Multi-provider LLM factory.

Instantiates LangChain chat model objects for any supported provider
using a simple registry pattern. Provider-specific packages are
imported lazily so only the dependencies for the chosen provider
need to be installed.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Provider registry ────────────────────────────────────────────

# Maps provider name → (import_path, class_name, model_kwarg_name).
_PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "openai": ("langchain_openai", "ChatOpenAI", "model"),
    "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI", "model"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "model"),
    "groq": ("langchain_groq", "ChatGroq", "model"),
    "mistral": ("langchain_mistralai", "ChatMistralAI", "model"),
}

# Maps provider → the env-var key name used to pass the API key.
_API_KEY_KWARG: dict[str, str] = {
    "openai": "api_key",
    "gemini": "google_api_key",
    "anthropic": "api_key",
    "groq": "api_key",
    "mistral": "api_key",
}


class LLMFactory:
    """Create LangChain chat models for any supported provider."""

    @staticmethod
    def create(
        provider: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        **overrides: Any,
    ) -> BaseChatModel:
        """Build a chat model instance.

        Parameters
        ----------
        provider:
            Provider name (``openai``, ``gemini``, ``anthropic``,
            ``groq``, ``mistral``).  Falls back to settings default.
        model:
            Model identifier (e.g. ``gpt-4o``).  Falls back to settings.
        settings:
            Explicit settings; defaults to the global singleton.
        **overrides:
            Extra kwargs forwarded to the model constructor
            (e.g. ``temperature``, ``max_tokens``).
        """
        cfg = settings or get_settings()
        provider = (provider or cfg.default_llm_provider).lower()
        model_name = model or cfg.default_llm_model

        if provider not in _PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. Supported: {sorted(_PROVIDER_REGISTRY)}"
            )

        # Validate API key availability
        api_key = cfg.get_api_key(provider)

        # Lazy-import the provider class
        module_path, class_name, model_kwarg = _PROVIDER_REGISTRY[provider]
        import importlib

        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Build kwargs
        kwargs: dict[str, Any] = {
            model_kwarg: model_name,
            _API_KEY_KWARG[provider]: api_key,
            "temperature": overrides.pop("temperature", cfg.llm_temperature),
            "max_tokens": overrides.pop("max_tokens", cfg.llm_max_tokens),
            **overrides,
        }

        logger.info("Creating LLM: provider=%s, model=%s", provider, model_name)
        return model_class(**kwargs)

    @staticmethod
    def get_available_providers(settings: Settings | None = None) -> list[str]:
        """Return providers whose API keys are configured."""
        cfg = settings or get_settings()
        return cfg.get_available_providers()
