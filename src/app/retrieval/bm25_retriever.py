"""BM25 keyword retriever.

Implements a LangChain-compatible retriever that uses BM25Okapi from
``rank_bm25`` for sparse keyword matching.  The index is held in-memory
and rebuilt on ingestion.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Simple tokeniser: lowercase, split on non-alphanumeric, drop short tokens.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenisation with basic normalisation."""
    return [tok for tok in _TOKEN_PATTERN.findall(text.lower()) if len(tok) > 1]


class BM25RetrieverService(BaseRetriever):
    """In-memory BM25 retriever backed by ``rank_bm25``.

    Attributes:
        documents: The corpus of ``Document`` objects.
        k: Number of results to return.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document] = []
    k: int = 5

    # Private attributes (Pydantic v2 style)
    _bm25: BM25Okapi | None = PrivateAttr(default=None)
    _tokenized_corpus: list[list[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Build the BM25 index after model initialisation."""
        super().model_post_init(__context)
        self._rebuild_index()

    # ── Public helpers ───────────────────────────────────────────

    def update_documents(self, documents: list[Document]) -> None:
        """Replace the corpus and rebuild the BM25 index."""
        self.documents = list(documents)
        self._rebuild_index()

    def add_documents(self, new_docs: list[Document]) -> None:
        """Append documents and rebuild the index."""
        self.documents = list(self.documents) + list(new_docs)
        self._rebuild_index()

    # ── LangChain retriever interface ────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Return the top-k documents by BM25 score."""
        if not self._bm25 or not self.documents:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Pair each document with its score, sort descending
        scored = sorted(
            zip(self.documents, scores, strict=False),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[Document] = []
        for doc, score in scored[: self.k]:
            if score <= 0:
                break
            # Attach score to metadata for downstream fusion
            enriched = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(score)},
            )
            results.append(enriched)

        return results

    # ── Internal ─────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """(Re)build the BM25 index from the current corpus."""
        if not self.documents:
            self._bm25 = None
            self._tokenized_corpus = []
            return

        self._tokenized_corpus = [_tokenize(doc.page_content) for doc in self.documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(self.documents))
