"""Protocol for reranking providers."""

from __future__ import annotations

from typing import Protocol

from rag_engine.models.domain import RetrievalCandidate


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_n: int = 10,
    ) -> list[RetrievalCandidate]: ...
