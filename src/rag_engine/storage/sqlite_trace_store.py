"""SQLite-backed query trace store for observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from rag_engine.models.domain import Trace
from rag_engine.storage.migrations import initialize_trace_db


class SQLiteTraceStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        await initialize_trace_db(self._db_path)

    async def save_trace(self, trace: Trace) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, query, timestamp, latency_ms, rq_score, confidence, decision, reason_codes, spans) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.query,
                    trace.timestamp.isoformat(),
                    trace.latency_ms,
                    trace.rq_score,
                    trace.confidence,
                    trace.decision,
                    json.dumps(trace.reason_codes),
                    json.dumps(trace.spans),
                ),
            )
            await db.commit()

    async def get_trace(self, trace_id: str) -> Trace | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_trace(row)

    async def get_recent_traces(self, limit: int = 100) -> list[Trace]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces ORDER BY timestamp DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_trace(row) for row in rows]

    @staticmethod
    def _row_to_trace(row: aiosqlite.Row) -> Trace:
        return Trace(
            trace_id=row["trace_id"],
            query=row["query"],
            timestamp=datetime.fromisoformat(row["timestamp"]).replace(tzinfo=timezone.utc),
            latency_ms=row["latency_ms"],
            rq_score=row["rq_score"],
            confidence=row["confidence"],
            decision=row["decision"],
            reason_codes=json.loads(row["reason_codes"]),
            spans=json.loads(row["spans"]),
        )
