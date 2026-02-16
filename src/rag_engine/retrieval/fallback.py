"""Fallback retrieval strategies when initial retrieval quality is low."""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag_engine.models.domain import RetrievalCandidate, RetrievalResult
from rag_engine.observability.logger import get_logger

logger = get_logger("fallback")


class RewriteResponse(BaseModel):
    rewrites: list[str]


class FallbackManager:
    def __init__(self, hybrid_retriever, reranker, rq_scorer, settings) -> None:
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._rq_scorer = rq_scorer
        self._settings = settings

    async def expanded_retrieval(self, query: str) -> list[RetrievalCandidate]:
        """Try retrieval with larger k values."""
        expand_k = self._settings.retrieval_fallback_expand_k
        candidates = await self._retriever.retrieve(
            query, top_k_bm25=expand_k, top_k_vector=expand_k
        )
        reranked = await self._reranker.rerank(query, candidates, top_n=self._settings.rerank_top_n)
        logger.info("expanded_retrieval", candidates=len(reranked))
        return reranked

    async def query_rewrite(self, query: str, llm) -> list[str]:
        """Use LLM to generate alternative query formulations."""
        from rag_engine.generation.prompt_templates import QUERY_REWRITE_PROMPT

        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        try:
            result = await llm.generate_structured(prompt, RewriteResponse)
            rewrites = result.rewrites[:3]
            logger.info("query_rewrites", count=len(rewrites))
            return rewrites
        except Exception:
            # Fallback: try plain generation and parse JSON manually
            try:
                raw = await llm.generate(prompt)
                data = json.loads(raw)
                return data.get("rewrites", [])[:3]
            except Exception:
                logger.warning("query_rewrite_failed")
                return []

    async def fallback_retrieve(self, query: str, llm) -> RetrievalResult:
        """Execute fallback strategy: expand k, then try query rewrites."""
        # Step 1: Expanded retrieval
        candidates = await self.expanded_retrieval(query)
        rq_score, reason_codes = self._rq_scorer.score(candidates)

        if rq_score >= self._settings.rq_proceed_threshold:
            return RetrievalResult(
                candidates=candidates,
                quality_score=rq_score,
                reason_codes=reason_codes,
                decision="proceed",
            )

        # Step 2: Query rewriting
        rewrites = await self.query_rewrite(query, llm)
        best_candidates = candidates
        best_rq = rq_score
        best_reasons = reason_codes

        for rewrite in rewrites:
            new_candidates = await self._retriever.retrieve(rewrite)
            new_reranked = await self._reranker.rerank(
                query, new_candidates, top_n=self._settings.rerank_top_n
            )
            new_rq, new_reasons = self._rq_scorer.score(new_reranked)
            if new_rq > best_rq:
                best_candidates = new_reranked
                best_rq = new_rq
                best_reasons = new_reasons

        if best_rq >= self._settings.rq_fallback_threshold:
            return RetrievalResult(
                candidates=best_candidates,
                quality_score=best_rq,
                reason_codes=best_reasons,
                decision="proceed",
            )

        return RetrievalResult(
            candidates=best_candidates,
            quality_score=best_rq,
            reason_codes=best_reasons,
            decision="abstain",
        )
