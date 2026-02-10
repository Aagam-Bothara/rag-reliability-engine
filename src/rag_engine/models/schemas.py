"""Pydantic models for API request/response serialization."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    context: str | None = None
    mode: Literal["strict", "normal"] = "normal"
    latency_budget_ms: int = 5000


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    text_snippet: str | None = None


class DebugInfo(BaseModel):
    retrieval_quality: float
    rerank_top_scores: list[float]
    trace_id: str
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    decision: Literal["answer", "clarify", "abstain"]
    reasons: list[str]
    debug: DebugInfo


class IngestRequest(BaseModel):
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    status: str


class HealthResponse(BaseModel):
    status: str
    doc_count: int
    chunk_count: int
    index_size: int
