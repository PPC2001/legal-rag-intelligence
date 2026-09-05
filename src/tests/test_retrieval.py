"""Tests for BM25 and hybrid retrieval."""

from __future__ import annotations

from langchain_core.documents import Document

from app.retrieval.bm25_retriever import BM25RetrieverService


class TestBM25Retriever:
    """Verify BM25 keyword retrieval."""

    def test_basic_retrieval(self, sample_documents: list[Document]):
        """BM25 should rank relevant documents higher."""
        retriever = BM25RetrieverService(documents=sample_documents, k=2)
        results = retriever.invoke("maternity leave")

        assert len(results) >= 1
        # The maternity leave document should be top-ranked
        assert "maternity" in results[0].page_content.lower()

    def test_no_results_for_irrelevant_query(self, sample_documents):
        """An irrelevant query should return empty or low-score results."""
        retriever = BM25RetrieverService(documents=sample_documents, k=2)
        results = retriever.invoke("quantum computing algorithms")
        # BM25 may still return docs, but scores should be 0
        # Our implementation filters out score <= 0
        for doc in results:
            assert doc.metadata.get("bm25_score", 0) >= 0

    def test_empty_corpus(self):
        """An empty corpus should return no results."""
        retriever = BM25RetrieverService(documents=[], k=5)
        results = retriever.invoke("any query")
        assert results == []

    def test_update_documents(self, sample_documents):
        """Updating documents should rebuild the index."""
        retriever = BM25RetrieverService(documents=[], k=5)
        assert retriever.invoke("maternity") == []

        retriever.update_documents(sample_documents)
        results = retriever.invoke("maternity")
        assert len(results) >= 1

    def test_add_documents(self, sample_documents):
        """Adding documents should extend the corpus."""
        # Use all 3 sample docs so BM25 IDF is not degenerate
        retriever = BM25RetrieverService(documents=sample_documents, k=5)

        new_doc = Document(
            page_content="The annual bonus is 15% of base salary for all employees.",
            metadata={"source": "test.txt", "page": 1, "file_type": "txt"},
        )
        retriever.add_documents([new_doc])
        assert len(retriever.documents) == 4
        results = retriever.invoke("annual bonus salary")
        assert any("bonus" in r.page_content.lower() for r in results)


class TestRRF:
    """Verify Reciprocal Rank Fusion logic."""

    def test_rrf_merges_results(self):
        """RRF should merge and deduplicate results from two lists."""
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        list1 = [
            Document(page_content="Doc A", metadata={}),
            Document(page_content="Doc B", metadata={}),
        ]
        list2 = [
            Document(page_content="Doc B", metadata={}),
            Document(page_content="Doc C", metadata={}),
        ]

        fused = _reciprocal_rank_fusion(
            ranked_lists=[list1, list2],
            weights=[0.7, 0.3],
        )

        contents = [d.page_content for d in fused]
        # Doc B appears in both lists, so it should score highest
        assert contents[0] == "Doc B"
        # All three docs should appear
        assert set(contents) == {"Doc A", "Doc B", "Doc C"}

    def test_rrf_respects_weights(self):
        """Higher weight should boost the corresponding list's rankings."""
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        list1 = [Document(page_content="Only in list 1", metadata={})]
        list2 = [Document(page_content="Only in list 2", metadata={})]

        fused = _reciprocal_rank_fusion(
            ranked_lists=[list1, list2],
            weights=[0.9, 0.1],
        )

        # list1's item should rank higher due to 0.9 weight
        assert fused[0].page_content == "Only in list 1"
