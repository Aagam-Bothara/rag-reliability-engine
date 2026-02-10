"""Core domain objects used throughout the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Document:
    doc_id: str
    source: str
    content_type: str
    metadata: dict
    raw_text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict
    token_count: int
    embedding: list[float] | None = None


@dataclass
class RetrievalCandidate:
    chunk: Chunk
    score: float
    source_method: str  # "bm25", "vector", "hybrid", "reranked"


@dataclass
class RetrievalResult:
    candidates: list[RetrievalCandidate]
    quality_score: float
    reason_codes: list[str]
    decision: str  # "proceed", "fallback", "abstain"


@dataclass
class DecomposedQuery:
    original: str
    sub_questions: list[str]
    synthesis_instruction: str


@dataclass
class ProcessedQuery:
    normalized: str
    language: str
    intent: str
    constraints: dict


@dataclass
class GenerationResult:
    answer: str
    cited_chunks: list[Chunk]
    cited_spans: list[dict]  # {chunk_id, text}


@dataclass
class VerificationResult:
    groundedness_score: float
    contradiction_rate: float
    self_consistency_score: float | None
    decision: str  # "pass", "warn", "abstain"
    reason_codes: list[str]


@dataclass
class Trace:
    trace_id: str
    query: str
    timestamp: datetime
    latency_ms: float
    rq_score: float
    confidence: float
    decision: str
    reason_codes: list[str]
    spans: list[dict]
