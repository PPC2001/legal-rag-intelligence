"""Question and Answer schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming natural-language question from the user."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question to answer from the documents",
        examples=["What is the notice period for junior staff?"],
    )


class SourceReference(BaseModel):
    """A single cited source chunk."""

    document: str = Field(description="Source document filename")
    page: int = Field(description="Page number (0 if not applicable)")
    chunk_text: str = Field(description="Relevant excerpt from the chunk")
    relevance_score: float = Field(description="Retrieval relevance score")


class RAGResponse(BaseModel):
    """Full response returned by the /ask endpoint."""

    answer: str = Field(description="Generated answer grounded in the documents")
    sources: list[SourceReference] = Field(description="Cited source chunks")
    sufficient_context: bool = Field(
        description="Whether enough context was found to answer",
    )
