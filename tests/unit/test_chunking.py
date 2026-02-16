"""Tests for structure-aware chunking."""

from rag_engine.chunking.structure_chunker import StructureChunker


def test_chunk_simple_text():
    chunker = StructureChunker(max_tokens=50, overlap_pct=0.0)
    text = "This is a simple paragraph."
    chunks = chunker.chunk(text, {"doc_id": "test"})
    assert len(chunks) >= 1
    assert chunks[0].text.strip() == text


def test_chunk_by_headings():
    chunker = StructureChunker(max_tokens=100, overlap_pct=0.0)
    text = """# Introduction
This is the introduction.

## Methods
This describes the methods.

## Results
These are the results."""
    chunks = chunker.chunk(text, {"doc_id": "test"})
    assert len(chunks) >= 3
    # Each section should be a separate chunk
    texts = [c.text.strip() for c in chunks]
    assert any("introduction" in t.lower() for t in texts)
    assert any("methods" in t.lower() for t in texts)
    assert any("results" in t.lower() for t in texts)


def test_chunk_with_overlap():
    chunker = StructureChunker(max_tokens=8, overlap_pct=0.15)
    text = (
        "First paragraph with some words here today.\n\nSecond paragraph with more words here now."
    )
    chunks = chunker.chunk(text, {"doc_id": "test"})
    assert len(chunks) >= 2
    # Second chunk should contain overlap from first
    assert chunks[1].index > chunks[0].index


def test_chunk_assigns_metadata():
    chunker = StructureChunker(max_tokens=100, overlap_pct=0.0)
    text = "# Title\nSome content."
    chunks = chunker.chunk(text, {"doc_id": "doc1", "source": "test.md"})
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc1"
    assert "heading_path" in chunks[0].metadata


def test_chunk_empty_text():
    chunker = StructureChunker(max_tokens=100, overlap_pct=0.0)
    chunks = chunker.chunk("", {"doc_id": "test"})
    assert len(chunks) == 0


def test_chunk_token_limit():
    chunker = StructureChunker(max_tokens=10, overlap_pct=0.0)
    text = "This is a very long paragraph. " * 50
    chunks = chunker.chunk(text, {"doc_id": "test"})
    # Should split into multiple chunks
    assert len(chunks) > 1
    # Each chunk should respect the token limit (approximately)
    for chunk in chunks:
        assert chunk.token_count <= 15  # Allow some slack for sentence boundaries
