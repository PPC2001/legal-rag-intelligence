"""Legal RAG QA — Production-grade document intelligence system."""

__version__ = "0.1.0"


def main() -> None:
    """CLI entry-point (registered in pyproject.toml)."""
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
