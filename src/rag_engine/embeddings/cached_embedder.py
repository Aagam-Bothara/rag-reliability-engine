"""Caching wrapper around an Embedder that stores results in SQLite."""

from __future__ import annotations

from rag_engine.embeddings.cache import EmbeddingCache
from rag_engine.observability.logger import get_logger

logger = get_logger("cached_embedder")


class CachedEmbedder:
    """Wraps any Embedder, checks EmbeddingCache first, calls delegate for misses."""

    def __init__(self, delegate, cache: EmbeddingCache) -> None:
        self._delegate = delegate
        self._cache = cache

    @property
    def dimensions(self) -> int:
        return self._delegate.dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Batch-check cache
        cached = await self._cache.get_batch(texts)

        # Identify misses
        miss_indices = [i for i in range(len(texts)) if i not in cached]

        if not miss_indices:
            logger.info("embed_texts_all_cached", count=len(texts))
            return [cached[i] for i in range(len(texts))]

        # Embed misses via delegate
        miss_texts = [texts[i] for i in miss_indices]
        miss_embeddings = await self._delegate.embed_texts(miss_texts)

        # Store new embeddings in cache
        await self._cache.put_batch(miss_texts, miss_embeddings)

        # Merge results in original order
        result: list[list[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            if i in cached:
                result[i] = cached[i]
        for idx, emb in zip(miss_indices, miss_embeddings):
            result[idx] = emb

        logger.info(
            "embed_texts_with_cache",
            total=len(texts),
            hits=len(texts) - len(miss_indices),
            misses=len(miss_indices),
        )
        return result

    async def embed_query(self, query: str) -> list[float]:
        cached = await self._cache.get(query)
        if cached is not None:
            logger.debug("embed_query_cache_hit", query_len=len(query))
            return cached

        embedding = await self._delegate.embed_query(query)
        await self._cache.put(query, embedding)
        logger.debug("embed_query_cache_miss", query_len=len(query))
        return embedding
