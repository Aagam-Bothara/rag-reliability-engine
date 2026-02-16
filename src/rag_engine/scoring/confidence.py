"""Final confidence scoring: CONF = alpha*RQ + beta*groundedness - gamma*contradiction_rate."""

from __future__ import annotations

from rag_engine.config.settings import Settings


class ConfidenceScorer:
    def __init__(self, settings: Settings) -> None:
        self.alpha = settings.conf_alpha
        self.beta = settings.conf_beta
        self.gamma = settings.conf_gamma

    def score(self, rq: float, groundedness: float, contradiction_rate: float) -> float:
        conf = self.alpha * rq + self.beta * groundedness - self.gamma * contradiction_rate
        return max(0.0, min(1.0, conf))
