"""Tests for chunk quality checks."""

from uuid import uuid4

from rag_engine.chunking.quality import (
    compute_coverage,
    detect_near_duplicates,
    filter_garbage_chunks,
)
from rag_engine.models.domain import Chunk


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid4()),
        doc_id="doc1",
        text=text,
        index=0,
        metadata={},
        token_count=len(text.split()),
    )


def test_filter_garbage_short():
    chunks = [_make_chunk("hi"), _make_chunk("This is a normal chunk with enough text.")]
    filtered = filter_garbage_chunks(chunks)
    assert len(filtered) == 1
    assert "normal" in filtered[0].text


def test_filter_garbage_non_alpha():
    chunks = [_make_chunk("12345 67890 !@#$% ^&*()"), _make_chunk("This is normal text.")]
    filtered = filter_garbage_chunks(chunks)
    assert len(filtered) == 1


def test_filter_garbage_repetitive():
    chunks = [_make_chunk("word word word word word word word word word word"),
              _make_chunk("Each word here is unique and different text.")]
    filtered = filter_garbage_chunks(chunks)
    assert len(filtered) == 1


def test_near_duplicate_detection():
    chunks = [
        _make_chunk("The quick brown fox jumps over the lazy dog."),
        _make_chunk("The quick brown fox jumps over the lazy dog."),
        _make_chunk("Something completely different about cats and mice."),
    ]
    duplicates = detect_near_duplicates(chunks)
    assert len(duplicates) >= 1


def test_coverage():
    text = "the quick brown fox jumps over the lazy dog"
    chunks = [_make_chunk("the quick brown fox"), _make_chunk("jumps over the lazy dog")]
    coverage = compute_coverage(chunks, text)
    assert coverage > 0.8


def test_coverage_empty():
    assert compute_coverage([], "") == 0.0
