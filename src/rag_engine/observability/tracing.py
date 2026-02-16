"""Lightweight request tracing with spans."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from rag_engine.models.domain import Trace


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class TraceContext:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or str(uuid4())
        self.spans: list[Span] = []
        self.start_time = time.monotonic()
        self._epoch = time.time()

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(
            name=name,
            start_ms=(time.monotonic() - self.start_time) * 1000,
            metadata=metadata,
        )
        try:
            yield s
        finally:
            s.end_ms = (time.monotonic() - self.start_time) * 1000
            self.spans.append(s)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000

    def to_trace(
        self, query: str, rq_score: float, confidence: float, decision: str, reason_codes: list[str]
    ) -> Trace:
        return Trace(
            trace_id=self.trace_id,
            query=query,
            timestamp=datetime.fromtimestamp(self._epoch, tz=timezone.utc),
            latency_ms=self.elapsed_ms,
            rq_score=rq_score,
            confidence=confidence,
            decision=decision,
            reason_codes=reason_codes,
            spans=[
                {
                    "name": s.name,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "duration_ms": s.duration_ms,
                    **s.metadata,
                }
                for s in self.spans
            ],
        )
