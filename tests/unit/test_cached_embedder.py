"""Tests for CachedEmbedder wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_engine.embeddings.cache import EmbeddingCache
from rag_engine.embeddings.cached_embedder import CachedEmbedder


class FakeEmbedder:
    """Fake embedder that tracks call counts."""

    def __init__(self) -> None:
        self.embed_texts_calls = 0
        self.embed_query_calls = 0
        self._dimensions = 3

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls += 1
        return [[float(i + 1)] * 3 for i in range(len(texts))]

    async def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls += 1
        return [1.0, 2.0, 3.0]


@pytest.fixture
async def embedder_pair():
    tmp = tempfile.mkdtemp()
    cache = EmbeddingCache(str(Path(tmp) / "cache.db"))
    await cache.initialize()
    delegate = FakeEmbedder()
    embedder = CachedEmbedder(delegate=delegate, cache=cache)
    return embedder, delegate


async def test_embed_query_caches(embedder_pair):
    embedder, delegate = embedder_pair
    result1 = await embedder.embed_query("hello")
    result2 = await embedder.embed_query("hello")
    assert result1 == result2
    assert delegate.embed_query_calls == 1


async def test_embed_query_different_queries(embedder_pair):
    embedder, delegate = embedder_pair
    await embedder.embed_query("hello")
    await embedder.embed_query("world")
    assert delegate.embed_query_calls == 2


async def test_embed_texts_caches(embedder_pair):
    embedder, delegate = embedder_pair
    texts = ["a", "b", "c"]
    result1 = await embedder.embed_texts(texts)
    result2 = await embedder.embed_texts(texts)
    assert result1 == result2
    assert delegate.embed_texts_calls == 1


async def test_embed_texts_partial_cache(embedder_pair):
    embedder, delegate = embedder_pair
    await embedder.embed_texts(["a", "b"])
    assert delegate.embed_texts_calls == 1
    await embedder.embed_texts(["a", "b", "c"])
    assert delegate.embed_texts_calls == 2


async def test_embed_texts_empty(embedder_pair):
    embedder, delegate = embedder_pair
    result = await embedder.embed_texts([])
    assert result == []
    assert delegate.embed_texts_calls == 0


async def test_dimensions_passthrough(embedder_pair):
    embedder, delegate = embedder_pair
    assert embedder.dimensions == 3
