"""Query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from rag_engine.api.dependencies import get_query_pipeline
from rag_engine.api.rate_limiter import rate_limit
from rag_engine.exceptions import RAGEngineError
from rag_engine.models.schemas import QueryRequest, QueryResponse
from rag_engine.pipeline.query_pipeline import QueryPipeline

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    _auth: dict = Depends(rate_limit),
) -> QueryResponse:
    try:
        return await pipeline.execute(request)
    except RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    _auth: dict = Depends(rate_limit),
):
    """Stream query results via Server-Sent Events."""

    async def event_generator():
        try:
            async for event in pipeline.execute_stream(request):
                event_type = event["event"]
                data = event["data"]
                yield f"event: {event_type}\ndata: {data}\n\n"
        except RAGEngineError as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
