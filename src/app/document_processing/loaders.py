"""Document loaders for PDF, DOCX, and plain-text files.

Each loader normalises the output into LangChain ``Document`` objects with
consistent metadata (``source``, ``page``, ``file_type``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported file extensions ──────────────────────────────────────
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension is not in ``SUPPORTED_EXTENSIONS``."""


# ── Individual loaders ──────────────────────────────────────────


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF file, one ``Document`` per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "file_type": "pdf",
                    },
                )
            )
    return docs


def _load_docx(path: Path) -> list[Document]:
    """Load a DOCX file.  Each paragraph becomes metadata-tagged text
    but the full document is returned as a single ``Document`` so the
    chunker can split it intelligently.
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    if not full_text.strip():
        return []
    return [
        Document(
            page_content=full_text,
            metadata={"source": path.name, "page": 0, "file_type": "docx"},
        )
    ]


def _load_txt(path: Path) -> list[Document]:
    """Load a plain-text file as a single ``Document``."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [
        Document(
            page_content=text,
            metadata={"source": path.name, "page": 0, "file_type": "txt"},
        )
    ]


# ── Factory ─────────────────────────────────────────────────────

_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
}


def load_document(file_path: str | Path) -> list[Document]:
    """Load a document from *file_path* and return LangChain ``Document`` list.

    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    loader_fn = _LOADER_MAP.get(ext)
    if loader_fn is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Loading %s document: %s", ext.upper(), path.name)
    documents = loader_fn(path)
    logger.info("Loaded %d raw section(s) from %s", len(documents), path.name)
    return documents
