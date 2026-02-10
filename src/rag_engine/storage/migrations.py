"""Idempotent database schema creation."""

from __future__ import annotations

import aiosqlite

DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    raw_text TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    token_count INTEGER NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
)
"""

CHUNKS_DOC_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)
"""

TRACES_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    rq_score REAL NOT NULL,
    confidence REAL NOT NULL,
    decision TEXT NOT NULL,
    reason_codes TEXT NOT NULL DEFAULT '[]',
    spans TEXT NOT NULL DEFAULT '[]'
)
"""

TRACES_TIMESTAMP_INDEX = """
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp)
"""


async def initialize_doc_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(DOCUMENTS_TABLE)
        await db.execute(CHUNKS_TABLE)
        await db.execute(CHUNKS_DOC_INDEX)
        await db.commit()


async def initialize_trace_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(TRACES_TABLE)
        await db.execute(TRACES_TIMESTAMP_INDEX)
        await db.commit()
