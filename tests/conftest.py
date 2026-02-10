"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from rag_engine.config.settings import Settings
from rag_engine.models.domain import Chunk, Document, RetrievalCandidate


@pytest.fixture
def settings():
    """Test settings with temp paths."""
    tmp = tempfile.mkdtemp()
    return Settings(
        openai_api_key="test-key",
        google_api_key="test-key",
        sqlite_doc_db_path=str(Path(tmp) / "test_rag.db"),
        sqlite_trace_db_path=str(Path(tmp) / "test_traces.db"),
        faiss_index_path=str(Path(tmp) / "faiss_index"),
        bm25_index_path=str(Path(tmp) / "bm25_index"),
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    doc_id = str(uuid4())
    return [
        Chunk(
            chunk_id=str(uuid4()),
            doc_id=doc_id,
            text=f"This is sample chunk number {i} with some text content about topic {i}.",
            index=i,
            metadata={"heading_path": ["Section"]},
            token_count=20,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_candidates(sample_chunks):
    """Create sample retrieval candidates."""
    return [
        RetrievalCandidate(
            chunk=chunk,
            score=1.0 - (i * 0.1),
            source_method="hybrid",
        )
        for i, chunk in enumerate(sample_chunks)
    ]


@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(
        doc_id=str(uuid4()),
        source="test.txt",
        content_type=".txt",
        metadata={"title": "Test Document"},
        raw_text="This is a test document with some content.\n\nIt has multiple paragraphs.\n\nAnd a third one.",
    )


@pytest.fixture
def tmp_dir():
    """Create a temporary directory."""
    return tempfile.mkdtemp()
