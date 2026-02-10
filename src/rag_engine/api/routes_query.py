"""Query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rag_engine.api.dependencies import get_query_pipeline
from rag_engine.exceptions import RAGEngineError
from rag_engine.models.schemas import QueryRequest, QueryResponse
from rag_engine.pipeline.query_pipeline import QueryPipeline

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
) -> QueryResponse:
    try:
        return await pipeline.execute(request)
    except RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))
