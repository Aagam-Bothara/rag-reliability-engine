"""Retrieval Quality (RQ) scoring: RQ = w1*rel + w2*margin + w3*coverage + w4*consistency."""

from __future__ import annotations

import math

import numpy as np

from rag_engine.config.settings import Settings
from rag_engine.models.domain import RetrievalCandidate
from rag_engine.scoring.reason_codes import ReasonCode


class RetrievalQualityScorer:
    def __init__(self, settings: Settings) -> None:
        self.w1 = settings.rq_w_relevance
        self.w2 = settings.rq_w_margin
        self.w3 = settings.rq_w_coverage
        self.w4 = settings.rq_w_consistency

    def score(
        self, candidates: list[RetrievalCandidate]
    ) -> tuple[float, list[str]]:
        if not candidates:
            return 0.0, [ReasonCode.NO_RESULTS]

        scores = [c.score for c in candidates]

        # Relevance: sigmoid-normalized top score
        rel = self._sigmoid_normalize(scores[0])

        # Margin: how much the top result stands out
        if len(scores) > 1:
            margin = (scores[0] - scores[1]) / (abs(scores[0]) + 1e-8)
            margin = max(0.0, min(1.0, margin))
        else:
            margin = 1.0

        # Coverage: unique doc_ids / total candidates
        unique_docs = len(set(c.chunk.doc_id for c in candidates))
        coverage = min(unique_docs / max(len(candidates), 1), 1.0)

        # Consistency: low variance among top scores = good
        top_scores = scores[:5]
        if len(top_scores) > 1:
            mean_s = float(np.mean(top_scores))
            std_s = float(np.std(top_scores))
            consistency = 1.0 - (std_s / (mean_s + 1e-8))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 1.0

        rq = self.w1 * rel + self.w2 * margin + self.w3 * coverage + self.w4 * consistency
        rq = max(0.0, min(1.0, rq))

        reason_codes: list[str] = []
        if rel < 0.4:
            reason_codes.append(ReasonCode.LOW_RELEVANCE)
        if margin < 0.1:
            reason_codes.append(ReasonCode.LOW_MARGIN)
        if coverage < 0.3:
            reason_codes.append(ReasonCode.LOW_COVERAGE)
        if consistency < 0.3:
            reason_codes.append(ReasonCode.LOW_CONSISTENCY)

        return rq, reason_codes

    @staticmethod
    def _sigmoid_normalize(x: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
        """Sigmoid normalization to map arbitrary scores to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
