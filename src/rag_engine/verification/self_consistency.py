"""Self-consistency check: regenerate a brief answer and compare."""

from __future__ import annotations

from difflib import SequenceMatcher

from rag_engine.config.constants import SELF_CONSISTENCY_TEMPERATURE
from rag_engine.generation.prompt_templates import (
    SELF_CONSISTENCY_PROMPT,
    format_evidence_block,
)
from rag_engine.models.domain import Chunk
from rag_engine.observability.logger import get_logger

logger = get_logger("self_consistency")


class SelfConsistencyChecker:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def check(
        self, query: str, evidence: list[Chunk], original_answer: str
    ) -> float:
        """Regenerate a brief answer and compare with the original. Returns similarity 0-1."""
        evidence_block = format_evidence_block(evidence)
        prompt = SELF_CONSISTENCY_PROMPT.format(
            query=query, evidence_block=evidence_block
        )

        try:
            brief_answer = await self._llm.generate(
                prompt, temperature=SELF_CONSISTENCY_TEMPERATURE
            )
            similarity = self._compare(original_answer, brief_answer)
        except Exception:
            logger.warning("self_consistency_check_failed")
            similarity = 0.5  # Neutral fallback

        logger.info("self_consistency", score=round(similarity, 4))
        return similarity

    @staticmethod
    def _compare(answer_a: str, answer_b: str) -> float:
        """Simple text similarity using SequenceMatcher."""
        a = answer_a.lower().strip()
        b = answer_b.lower().strip()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
