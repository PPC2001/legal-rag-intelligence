# ─────────────────────────────────────────────────────────────
#  Legal RAG QA — Multi-stage Docker build
# ─────────────────────────────────────────────────────────────

# Stage 1: Build dependencies with UV
FROM python:3.12-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first (layer caching)
COPY pyproject.toml uv.lock* README.md ./
COPY src/ src/

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-editable

# ─────────────────────────────────────────────────────────────
# Stage 2: Slim runtime
FROM python:3.12-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ src/
COPY pyproject.toml README.md ./

# Create data directory
RUN mkdir -p data/uploads && chown -R appuser:appuser /app

# Set PATH so the venv python is used
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["python", "-m", "app.main"]
