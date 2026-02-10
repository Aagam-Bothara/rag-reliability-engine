"""Tests for BM25 tokenizer."""

from rag_engine.keyword_search.tokenizer import tokenize


def test_tokenize_basic():
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    # Stopwords removed
    assert "the" not in tokens
    assert "over" not in tokens


def test_tokenize_lowercase():
    tokens = tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_punctuation():
    tokens = tokenize("Hello, world! How are you?")
    assert "hello" in tokens
    assert "world" in tokens
    # Single-char tokens are removed
    assert "," not in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_all_stopwords():
    tokens = tokenize("the a an is are")
    assert tokens == []
