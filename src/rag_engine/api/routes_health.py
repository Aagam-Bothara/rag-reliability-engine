"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_engine.api.dependencies import get_doc_store, get_vector_store
from rag_engine.models.schemas import HealthResponse
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    doc_store: SQLiteDocStore = Depends(get_doc_store),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        doc_count=await doc_store.count_documents(),
        chunk_count=await doc_store.count_chunks(),
        index_size=vector_store.size,
    )
