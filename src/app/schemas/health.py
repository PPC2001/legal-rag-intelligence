"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = Field(description="ok | degraded | error")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current environment")
    available_providers: list[str] = Field(description="LLM providers with keys set")
    database_connected: bool = Field(description="PostgreSQL reachable")
