"""Intelligent document chunking for legal / policy text.

Uses ``RecursiveCharacterTextSplitter`` with legal-aware separators
that respect section headers, numbered clauses, and paragraph breaks.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)

# Separators ordered from most to least preferred split points.
# Legal documents are structured: articles → sections → paragraphs.
_LEGAL_SEPARATORS: list[str] = [
    "\n\n\n",  # Major section break
    "\n\n",  # Paragraph break
    "\n",  # Line break
    ". ",  # Sentence boundary
    "; ",  # Clause boundary
    ", ",  # Sub-clause
    " ",  # Word boundary (fallback)
]


class DocumentChunker:
    """Split documents into overlapping chunks with enriched metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=_LEGAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split *documents* into chunks and enrich metadata.

        Each chunk receives:
        - ``chunk_index``: position within the parent document
        - ``total_chunks``: how many chunks the parent produced
        - All original metadata (``source``, ``page``, ``file_type``)
        """
        all_chunks: list[Document] = []

        for doc in documents:
            splits = self._splitter.split_documents([doc])
            total = len(splits)
            for idx, chunk in enumerate(splits):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = total
            all_chunks.extend(splits)

        logger.info(
            "Chunked %d document(s) → %d chunks (size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return all_chunks
