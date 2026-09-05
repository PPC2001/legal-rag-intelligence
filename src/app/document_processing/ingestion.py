"""Document ingestion service.

Orchestrates the full pipeline: load → chunk → embed → store in PGVector.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.document_processing.chunker import DocumentChunker
from app.document_processing.loaders import load_document
from app.schemas import IngestionResponse

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: Path) -> str:
    """SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


class IngestionService:
    """End-to-end document ingestion into the vector store."""

    def __init__(self, vector_store_manager) -> None:  # noqa: ANN001 — avoid circular import
        self._vector_store = vector_store_manager
        self._chunker = DocumentChunker()
        self._ingested_hashes: set[str] = set()

    def ingest_file(self, file_path: str | Path) -> IngestionResponse:
        """Load, chunk, and store a single document.

        Returns an ``IngestionResponse`` with status and chunk count.
        Skips files that have already been ingested (by content hash).
        """
        path = Path(file_path)
        settings = get_settings()
        doc_hash = _compute_file_hash(path)

        # Deduplication
        if doc_hash in self._ingested_hashes:
            logger.info("Skipping duplicate document: %s", path.name)
            return IngestionResponse(
                document_id=doc_hash,
                filename=path.name,
                chunks_created=0,
                status="skipped",
                message="Document already ingested (identical content hash).",
            )

        start = time.perf_counter()

        try:
            # 1. Load
            raw_docs: list[Document] = load_document(path)
            if not raw_docs:
                return IngestionResponse(
                    document_id=doc_hash,
                    filename=path.name,
                    chunks_created=0,
                    status="error",
                    message="No readable text found in the document.",
                )

            # 2. Chunk
            chunks = self._chunker.chunk_documents(raw_docs)

            # 3. Add document_id to every chunk's metadata
            for chunk in chunks:
                chunk.metadata["document_id"] = doc_hash
                chunk.metadata["collection"] = settings.collection_name

            # 4. Store in PGVector
            self._vector_store.add_documents(chunks)

            self._ingested_hashes.add(doc_hash)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Ingested %s → %d chunks in %.1f ms",
                path.name,
                len(chunks),
                elapsed_ms,
            )

            return IngestionResponse(
                document_id=doc_hash,
                filename=path.name,
                chunks_created=len(chunks),
                status="success",
                message=f"Ingested in {elapsed_ms:.0f} ms.",
            )

        except Exception as exc:
            logger.exception("Failed to ingest %s", path.name)
            return IngestionResponse(
                document_id=doc_hash,
                filename=path.name,
                chunks_created=0,
                status="error",
                message=str(exc),
            )
