"""Metric recording helpers for traces."""

from __future__ import annotations

from rag_engine.observability.logger import get_logger

logger = get_logger("metrics")


def log_retrieval_metrics(
    trace_id: str,
    rq_score: float,
    top_scores: list[float],
    num_candidates: int,
    unique_docs: int,
) -> None:
    logger.info(
        "retrieval_metrics",
        trace_id=trace_id,
        rq_score=round(rq_score, 4),
        top_scores=[round(s, 4) for s in top_scores[:5]],
        num_candidates=num_candidates,
        unique_docs=unique_docs,
    )


def log_generation_metrics(
    trace_id: str,
    groundedness: float,
    contradiction_rate: float,
    confidence: float,
    decision: str,
) -> None:
    logger.info(
        "generation_metrics",
        trace_id=trace_id,
        groundedness=round(groundedness, 4),
        contradiction_rate=round(contradiction_rate, 4),
        confidence=round(confidence, 4),
        decision=decision,
    )


def log_latency(trace_id: str, stage: str, duration_ms: float) -> None:
    logger.info(
        "latency",
        trace_id=trace_id,
        stage=stage,
        duration_ms=round(duration_ms, 2),
    )
