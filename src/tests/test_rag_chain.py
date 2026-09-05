"""Tests for the RAG chain (with mocked LLM and retriever)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.schemas import QuestionRequest


class TestRAGChain:
    """Verify end-to-end RAG behaviour with mocked dependencies."""

    def _make_chain(self, retrieved_docs: list[Document]):
        """Create a RAGChain with a mocked retriever."""
        from langchain_core.output_parsers import StrOutputParser

        from app.rag.chain import RAGChain

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = retrieved_docs

        mock_settings = MagicMock()
        mock_settings.retrieval_top_k = 5
        mock_settings.default_llm_provider = "gemini"
        mock_settings.default_llm_model = "gemini-2.0-flash"
        mock_settings.llm_temperature = 0.0
        mock_settings.llm_max_tokens = 2048
        mock_settings.get_api_key.return_value = "test-key"

        chain = RAGChain.__new__(RAGChain)
        chain._retriever = mock_retriever
        chain._settings = mock_settings
        chain._parser = StrOutputParser()

        return chain

    def test_hallucination_guard_no_docs(self):
        """If no documents are retrieved, the chain should refuse to answer."""
        chain = self._make_chain(retrieved_docs=[])
        request = QuestionRequest(question="What is the notice period?")
        response = chain.ask(request)

        assert response.sufficient_context is False
        assert "cannot find" in response.answer.lower()
        assert response.sources == []

    @patch("app.rag.chain.LLMFactory")
    @patch("app.rag.chain.QA_PROMPT")
    def test_successful_answer_with_sources(self, mock_prompt, mock_factory):
        """A successful retrieval should produce an answer with sources."""
        docs = [
            Document(
                page_content="The notice period for junior staff is 30 days.",
                metadata={
                    "source": "handbook.txt",
                    "page": 2,
                    "file_type": "txt",
                    "rrf_score": 0.5,
                },
            )
        ]
        chain = self._make_chain(retrieved_docs=docs)

        answer_text = (
            "The notice period for junior staff is 30 days. [Source: handbook.txt, Page 2]"
        )

        # Set up the chain pipeline mock: QA_PROMPT | llm | parser → answer_text
        mock_chain_end = MagicMock()
        mock_chain_end.invoke.return_value = answer_text

        # QA_PROMPT | llm → intermediate
        mock_intermediate = MagicMock()
        mock_intermediate.__or__ = MagicMock(return_value=mock_chain_end)

        # QA_PROMPT.__or__(llm) → intermediate
        mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)

        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        request = QuestionRequest(question="What is the notice period?")
        response = chain.ask(request)

        assert response.sufficient_context is True
        assert len(response.sources) == 1
        assert response.sources[0].document == "handbook.txt"
        assert response.sources[0].page == 2
        assert response.answer == answer_text


class TestSourceBuilding:
    """Verify source reference extraction."""

    def test_build_sources_deduplicates(self):
        """Duplicate content should be deduplicated."""
        from app.rag.chain import _build_sources

        docs = [
            Document(
                page_content="Same content here.",
                metadata={"source": "a.txt", "page": 1, "rrf_score": 0.5},
            ),
            Document(
                page_content="Same content here.",
                metadata={"source": "a.txt", "page": 1, "rrf_score": 0.4},
            ),
            Document(
                page_content="Different content.",
                metadata={"source": "b.txt", "page": 3, "rrf_score": 0.3},
            ),
        ]
        sources = _build_sources(docs)
        assert len(sources) == 2

    def test_build_sources_preserves_metadata(self):
        """Source references should carry document name and page."""
        from app.rag.chain import _build_sources

        docs = [
            Document(
                page_content="Policy content.",
                metadata={"source": "policy.pdf", "page": 7, "rrf_score": 0.8},
            ),
        ]
        sources = _build_sources(docs)
        assert sources[0].document == "policy.pdf"
        assert sources[0].page == 7
        assert sources[0].relevance_score == 0.8
