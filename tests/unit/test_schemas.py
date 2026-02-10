"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from rag_engine.models.schemas import (
    Citation,
    DebugInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


def test_query_request_defaults():
    req = QueryRequest(query="What is AI?")
    assert req.mode == "normal"
    assert req.latency_budget_ms == 5000
    assert req.context is None


def test_query_request_strict():
    req = QueryRequest(query="What is AI?", mode="strict")
    assert req.mode == "strict"


def test_query_response_serialization():
    resp = QueryResponse(
        answer="AI is artificial intelligence.",
        citations=[Citation(doc_id="d1", chunk_id="c1", text_snippet="snippet")],
        confidence=0.85,
        decision="answer",
        reasons=[],
        debug=DebugInfo(
            retrieval_quality=0.7,
            rerank_top_scores=[0.9, 0.8],
            trace_id="abc123",
            latency_ms=150.5,
        ),
    )
    data = resp.model_dump()
    assert data["answer"] == "AI is artificial intelligence."
    assert data["confidence"] == 0.85
    assert len(data["citations"]) == 1


def test_query_request_invalid_mode():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", mode="invalid")


def test_ingest_response():
    resp = IngestResponse(doc_id="d1", chunks_created=5, status="indexed")
    assert resp.chunks_created == 5


def test_health_response():
    resp = HealthResponse(status="ok", doc_count=10, chunk_count=100, index_size=100)
    assert resp.status == "ok"
