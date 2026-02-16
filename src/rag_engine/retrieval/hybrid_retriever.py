"""Hybrid retriever combining BM25 + FAISS vector search with RRF fusion."""

from __future__ import annotations

import asyncio

import numpy as np

from rag_engine.protocols.embedder import Embedder
from rag_engine.keyword_search.bm25_index import BM25Index
from rag_engine.models.domain import Chunk, RetrievalCandidate
from rag_engine.observability.logger import get_logger
from rag_engine.retrieval.rrf import reciprocal_rank_fusion
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger("hybrid_retriever")


class HybridRetrieverImpl:
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
        doc_store: SQLiteDocStore,
        embedder: Embedder,
        rrf_k: int = 60,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._doc_store = doc_store
        self._embedder = embedder
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k_bm25: int = 50,
        top_k_vector: int = 50,
    ) -> list[RetrievalCandidate]:
        # 1. Embed query
        query_embedding = await self._embedder.embed_query(query)
        query_array = np.array(query_embedding, dtype=np.float32)

        # 2. Concurrent retrieval from both sources
        vector_results, bm25_results = await asyncio.gather(
            asyncio.to_thread(self._vector_store.search, query_array, top_k_vector),
            asyncio.to_thread(self._bm25_index.search, query, top_k_bm25),
        )

        logger.info(
            "retrieval_results",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
        )

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], k=self._rrf_k
        )

        if not fused:
            return []

        # 4. Load chunk objects from doc store
        chunk_ids = [cid for cid, _ in fused]
        chunks_map = await self._doc_store.get_chunks_by_ids(chunk_ids)

        # 5. Build candidates preserving RRF order
        candidates = []
        for chunk_id, score in fused:
            chunk = chunks_map.get(chunk_id)
            if chunk:
                candidates.append(
                    RetrievalCandidate(chunk=chunk, score=score, source_method="hybrid")
                )

        return candidates
