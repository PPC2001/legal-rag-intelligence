#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Legal RAG QA — Startup Script
#
# Usage:
#   ./start.sh                  # Start server using settings from .env
#   ./start.sh --reload         # Start with auto-reload enabled
#   ./start.sh --port 8080      # Start on port 8080
#   ./start.sh --help           # Show uvicorn CLI options
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Ensure .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found in $SCRIPT_DIR!"
    if [ -f .env.example ]; then
        echo "📋 Creating .env from .env.example..."
        cp .env.example .env
        echo "👉 Please edit .env with your DATABASE_URL and API keys, then re-run ./start.sh."
        exit 1
    else
        echo "❌ .env.example not found. Please create .env before starting."
        exit 1
    fi
fi

# 2. Select execution environment (prefer uv, fallback to venv or python3)
if command -v uv >/dev/null 2>&1; then
    UV_CMD="uv run"
elif [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
    UV_CMD=".venv/bin/"
else
    UV_CMD=""
fi

echo "=================================================="
echo "  ⚖️   Legal RAG QA — Document Intelligence"
echo "=================================================="
echo "  📁 Root:   $SCRIPT_DIR"
echo "  ⚙️  Config: .env"
echo "=================================================="

# 3. Launch application
if [ $# -eq 0 ]; then
    # Default launch using application settings from .env
    if [ -n "$UV_CMD" ]; then
        exec $UV_CMD python -m app.main
    else
        exec python3 -m app.main
    fi
else
    # Forward custom flags (e.g. --port, --reload, --workers) to uvicorn
    if [ -n "$UV_CMD" ]; then
        exec $UV_CMD uvicorn app.main:create_app --factory "$@"
    else
        exec uvicorn app.main:create_app --factory "$@"
    fi
fi
