"""Tests for document loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.document_processing.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileTypeError,
    load_document,
)


class TestDocumentLoader:
    """Verify document loading across file types."""

    def test_load_txt_file(self, sample_txt_file: Path):
        """TXT files should produce one Document with correct metadata."""
        docs = load_document(sample_txt_file)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["source"] == sample_txt_file.name
        assert "Working Hours" in docs[0].page_content

    def test_load_nonexistent_file(self):
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/doc.txt")

    def test_load_unsupported_extension(self, tmp_path: Path):
        """Unsupported extensions should raise UnsupportedFileTypeError."""
        bad_file = tmp_path / "notes.md"
        bad_file.write_text("some markdown")
        with pytest.raises(UnsupportedFileTypeError):
            load_document(bad_file)

    def test_load_empty_txt(self, tmp_path: Path):
        """Empty files should return an empty list."""
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        docs = load_document(empty)
        assert docs == []

    def test_supported_extensions_set(self):
        """All three formats should be in the supported set."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS


class TestDocumentChunker:
    """Verify chunking behaviour."""

    def test_chunks_have_metadata(self, sample_txt_file: Path):
        """Each chunk should carry chunk_index and total_chunks metadata."""
        from app.document_processing.chunker import DocumentChunker

        docs = load_document(sample_txt_file)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "source" in chunk.metadata

    def test_chunk_size_respected(self, sample_txt_file: Path):
        """No chunk should exceed chunk_size (with some tolerance for separators)."""
        from app.document_processing.chunker import DocumentChunker

        docs = load_document(sample_txt_file)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_documents(docs)

        for chunk in chunks:
            # Allow small overshoot from separator-aware splitting
            assert len(chunk.page_content) <= 250

    def test_empty_input(self):
        """Chunking an empty list should return an empty list."""
        from app.document_processing.chunker import DocumentChunker

        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_documents([]) == []
