"""Groundedness check: does the answer align with the evidence?"""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag_engine.generation.prompt_templates import (
    GROUNDEDNESS_CHECK_PROMPT,
    format_evidence_block,
)
from rag_engine.models.domain import Chunk
from rag_engine.observability.logger import get_logger

logger = get_logger("groundedness")


class GroundednessResponse(BaseModel):
    score: float
    unsupported_claims: list[str] = []


class GroundednessChecker:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def check(self, answer: str, evidence: list[Chunk], query: str = "") -> float:
        evidence_block = format_evidence_block(evidence)
        prompt = GROUNDEDNESS_CHECK_PROMPT.format(
            query=query, answer=answer, evidence_block=evidence_block
        )

        try:
            result = await self._llm.generate_structured(
                prompt, GroundednessResponse
            )
            score = max(0.0, min(1.0, result.score))
        except Exception:
            try:
                raw = await self._llm.generate(prompt)
                data = json.loads(raw)
                score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            except Exception:
                logger.warning("groundedness_check_failed")
                score = 0.5  # Neutral fallback

        logger.info("groundedness", score=round(score, 4))
        return score
