"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

# Force test environment before any settings import
os.environ["ENV"] = "dev"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test_db"
os.environ["GOOGLE_API_KEY"] = "test-google-key"


@pytest.fixture()
def settings():
    """Return a Settings instance with test overrides."""
    from app.config.settings import Settings

    return Settings(
        env="DEV",
        debug=True,
        database_url="postgresql://test:test@localhost/test_db",
        google_api_key="test-google-key",
        default_llm_provider="gemini",
        default_llm_model="gemini-2.0-flash",
        collection_name="test_collection",
        chunk_size=500,
        chunk_overlap=100,
    )


@pytest.fixture()
def sample_documents() -> list[Document]:
    """A small set of test documents with metadata."""
    return [
        Document(
            page_content=(
                "Employees wishing to resign must provide written notice. "
                "Junior staff requires thirty (30) calendar days. "
                "Senior staff requires ninety (90) calendar days."
            ),
            metadata={"source": "handbook.txt", "page": 1, "file_type": "txt"},
        ),
        Document(
            page_content=(
                "Female employees are entitled to twenty-six (26) weeks of "
                "paid maternity leave at full salary."
            ),
            metadata={"source": "handbook.txt", "page": 3, "file_type": "txt"},
        ),
        Document(
            page_content=(
                "The Insured must notify GlobalShield Insurance of any claim "
                "within forty-eight (48) hours of discovery."
            ),
            metadata={"source": "insurance.txt", "page": 5, "file_type": "txt"},
        ),
    ]


@pytest.fixture()
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file for testing."""
    content = (
        "SECTION 1: Working Hours\n\n"
        "Standard working hours are Monday through Friday, 9:00 AM to 6:00 PM.\n"
        "The standard work week is forty (40) hours.\n\n"
        "SECTION 2: Overtime\n\n"
        "Overtime must be pre-approved by the direct manager.\n"
        "Overtime is compensated at 1.5 times the regular hourly rate.\n"
    )
    file_path = tmp_path / "test_policy.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a sample text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test document.", encoding="utf-8")
    return tmp_path
