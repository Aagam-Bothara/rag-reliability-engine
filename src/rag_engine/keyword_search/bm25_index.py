"""BM25 keyword search index using rank_bm25."""

from __future__ import annotations

import asyncio
import os
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag_engine.keyword_search.tokenizer import tokenize
from rag_engine.models.domain import Chunk
from rag_engine.observability.logger import get_logger

logger = get_logger("bm25_index")


class BM25Index:
    def __init__(self, index_path: str | None = None) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index_path = index_path
        self._write_lock = asyncio.Lock()

        if index_path:
            self._try_load(index_path)

    def _try_load(self, path: str) -> None:
        index_file = os.path.join(path, "bm25.pkl")
        if os.path.exists(index_file):
            with open(index_file, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            self._tokenized_corpus = data["tokenized_corpus"]
            logger.info("bm25_loaded", size=len(self._chunk_ids), path=path)

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks. Replaces existing index."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [tokenize(c.text) for c in chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
        logger.info("bm25_built", size=len(self._chunk_ids))

    async def rebuild(self, chunks: list[Chunk]) -> None:
        """Thread-safe rebuild of the BM25 index."""
        async with self._write_lock:
            await asyncio.to_thread(self.build, chunks)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search the BM25 index. Returns list of (chunk_id, score)."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str | None = None) -> None:
        path = path or self._index_path
        if not path:
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "chunk_ids": self._chunk_ids,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        logger.info("bm25_saved", path=path, size=len(self._chunk_ids))

    @property
    def size(self) -> int:
        return len(self._chunk_ids)
