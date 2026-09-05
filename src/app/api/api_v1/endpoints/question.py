"""Question-answering endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_chain
from app.rag.chain import RAGChain
from app.schemas import QuestionRequest, RAGResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/ask",
    response_model=RAGResponse,
    summary="Ask a question",
)
def ask_question(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> RAGResponse:
    """Submit a natural-language question and receive a cited answer.

    The answer is grounded strictly in ingested documents.
    If the system cannot find relevant context, it will refuse
    to answer rather than fabricate information.
    """
    try:
        return rag_chain.ask(request)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error answering question")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        ) from exc


@router.post(
    "/ask/stream",
    summary="Ask a question (streaming)",
)
async def ask_question_stream(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> StreamingResponse:
    """Stream the answer token-by-token via Server-Sent Events.

    Useful for real-time UIs that want to display the answer as it
    is being generated.
    """

    async def _event_generator():
        try:
            async for token in rag_chain.ask_stream(request):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Streaming error")
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
