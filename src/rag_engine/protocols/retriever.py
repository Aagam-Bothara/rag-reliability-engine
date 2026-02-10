"""Protocol for retrieval providers."""

from __future__ import annotations

from typing import Protocol

from rag_engine.models.domain import RetrievalCandidate


class Retriever(Protocol):
    async def retrieve(
        self, query: str, top_k: int = 50
    ) -> list[RetrievalCandidate]: ...


class HybridRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        top_k_bm25: int = 50,
        top_k_vector: int = 50,
    ) -> list[RetrievalCandidate]: ...
