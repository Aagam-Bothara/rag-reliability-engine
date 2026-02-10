"""FastAPI dependency injection helpers."""

from __future__ import annotations

from fastapi import Request

from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.pipeline.query_pipeline import QueryPipeline
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore


def get_query_pipeline(request: Request) -> QueryPipeline:
    return request.app.state.query_pipeline


def get_ingest_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.ingest_pipeline


def get_doc_store(request: Request) -> SQLiteDocStore:
    return request.app.state.doc_store


def get_vector_store(request: Request) -> FAISSVectorStore:
    return request.app.state.vector_store
