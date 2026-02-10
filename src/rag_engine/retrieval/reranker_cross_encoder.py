"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import asyncio

from sentence_transformers import CrossEncoder

from rag_engine.models.domain import RetrievalCandidate
from rag_engine.observability.logger import get_logger

logger = get_logger("reranker")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_n: int = 10,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]

        # CrossEncoder.predict is synchronous — run in thread pool
        scores = await asyncio.to_thread(self._model.predict, pairs)

        # Assign new scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for candidate, score in scored[:top_n]:
            result.append(
                RetrievalCandidate(
                    chunk=candidate.chunk,
                    score=float(score),
                    source_method="reranked",
                )
            )

        logger.info(
            "reranked",
            input_count=len(candidates),
            output_count=len(result),
            top_score=round(result[0].score, 4) if result else 0.0,
        )

        return result
