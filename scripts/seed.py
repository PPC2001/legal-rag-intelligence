"""Seed script — ingest sample documents for development testing.

Usage:
    uv run python scripts/seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    """Ingest all sample documents from data/samples/."""
    from app.config import get_settings
    from app.document_processing.ingestion import IngestionService
    from app.retrieval.vector_store import VectorStoreManager

    settings = get_settings()
    samples_dir = project_root / "data" / "samples"

    if not samples_dir.exists():
        print(f"❌  Samples directory not found: {samples_dir}")
        sys.exit(1)

    sample_files = sorted(samples_dir.iterdir())
    if not sample_files:
        print("❌  No sample files found.")
        sys.exit(1)

    print(f"📂  Found {len(sample_files)} sample file(s) in {samples_dir}\n")

    # Initialise services
    print("🔌  Connecting to vector store...")
    vector_store = VectorStoreManager(settings)
    ingestion = IngestionService(vector_store)

    total_chunks = 0
    for path in sample_files:
        if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            print(f"⏭️   Skipping unsupported file: {path.name}")
            continue

        print(f"📄  Ingesting: {path.name}")
        result = ingestion.ingest_file(path)
        print(f"    Status: {result.status} | Chunks: {result.chunks_created}")
        if result.message:
            print(f"    Message: {result.message}")
        total_chunks += result.chunks_created

    print(f"\n✅  Seeding complete! Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    main()
