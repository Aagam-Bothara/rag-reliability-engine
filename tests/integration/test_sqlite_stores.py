"""Integration tests for SQLite document and trace stores."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from rag_engine.models.domain import Chunk, Document, Trace
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.storage.sqlite_trace_store import SQLiteTraceStore


@pytest.fixture
async def doc_store():
    tmp = tempfile.mkdtemp()
    store = SQLiteDocStore(str(Path(tmp) / "test.db"))
    await store.initialize()
    return store


@pytest.fixture
async def trace_store():
    tmp = tempfile.mkdtemp()
    store = SQLiteTraceStore(str(Path(tmp) / "test_traces.db"))
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_save_and_get_document(doc_store):
    doc = Document(
        doc_id=str(uuid4()),
        source="test.txt",
        content_type=".txt",
        metadata={"title": "Test"},
        raw_text="Hello world",
    )
    await doc_store.save_document(doc)
    retrieved = await doc_store.get_document(doc.doc_id)
    assert retrieved is not None
    assert retrieved.source == "test.txt"
    assert retrieved.raw_text == "Hello world"


@pytest.mark.asyncio
async def test_save_and_get_chunks(doc_store):
    doc_id = str(uuid4())
    chunks = [
        Chunk(
            chunk_id=str(uuid4()),
            doc_id=doc_id,
            text=f"Chunk {i}",
            index=i,
            metadata={},
            token_count=2,
        )
        for i in range(3)
    ]
    await doc_store.save_chunks(chunks)

    retrieved = await doc_store.get_chunks_by_doc(doc_id)
    assert len(retrieved) == 3
    assert retrieved[0].index == 0


@pytest.mark.asyncio
async def test_get_chunks_by_ids(doc_store):
    doc_id = str(uuid4())
    chunks = [
        Chunk(
            chunk_id=str(uuid4()),
            doc_id=doc_id,
            text=f"Chunk {i}",
            index=i,
            metadata={},
            token_count=2,
        )
        for i in range(3)
    ]
    await doc_store.save_chunks(chunks)

    ids = [chunks[0].chunk_id, chunks[2].chunk_id]
    result = await doc_store.get_chunks_by_ids(ids)
    assert len(result) == 2
    assert chunks[0].chunk_id in result


@pytest.mark.asyncio
async def test_count_documents(doc_store):
    assert await doc_store.count_documents() == 0
    doc = Document(
        doc_id=str(uuid4()),
        source="test.txt",
        content_type=".txt",
        metadata={},
        raw_text="test",
    )
    await doc_store.save_document(doc)
    assert await doc_store.count_documents() == 1


@pytest.mark.asyncio
async def test_save_and_get_trace(trace_store):
    trace = Trace(
        trace_id=str(uuid4()),
        query="What is AI?",
        timestamp=datetime.now(timezone.utc),
        latency_ms=150.0,
        rq_score=0.7,
        confidence=0.85,
        decision="answer",
        reason_codes=[],
        spans=[{"name": "retrieval", "duration_ms": 50.0}],
    )
    await trace_store.save_trace(trace)
    retrieved = await trace_store.get_trace(trace.trace_id)
    assert retrieved is not None
    assert retrieved.query == "What is AI?"
    assert retrieved.confidence == 0.85


@pytest.mark.asyncio
async def test_recent_traces(trace_store):
    for i in range(5):
        trace = Trace(
            trace_id=str(uuid4()),
            query=f"Query {i}",
            timestamp=datetime.now(timezone.utc),
            latency_ms=100.0,
            rq_score=0.5,
            confidence=0.6,
            decision="answer",
            reason_codes=[],
            spans=[],
        )
        await trace_store.save_trace(trace)

    recent = await trace_store.get_recent_traces(limit=3)
    assert len(recent) == 3
