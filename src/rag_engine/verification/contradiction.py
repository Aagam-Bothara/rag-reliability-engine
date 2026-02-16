"""Contradiction detection: doc-vs-doc and answer-vs-evidence."""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag_engine.generation.prompt_templates import (
    ANSWER_CONTRADICTION_PROMPT,
    CONTRADICTION_DETECTION_PROMPT,
    format_evidence_block,
)
from rag_engine.models.domain import Chunk
from rag_engine.observability.logger import get_logger

logger = get_logger("contradiction")


class ContradictionResponse(BaseModel):
    contradictions: list[dict] = []
    contradiction_rate: float = 0.0


class ContradictionDetector:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def detect_doc_conflicts(self, chunks: list[Chunk]) -> list[dict]:
        """Detect contradictions between the top evidence chunks."""
        if len(chunks) < 2:
            return []

        passages = "\n\n".join(f"Passage {i + 1}: {c.text}" for i, c in enumerate(chunks[:5]))
        prompt = CONTRADICTION_DETECTION_PROMPT.format(passages=passages)

        try:
            result = await self._llm.generate_structured(prompt, ContradictionResponse)
            return result.contradictions
        except Exception:
            try:
                raw = await self._llm.generate(prompt)
                data = json.loads(raw)
                return data.get("contradictions", [])
            except Exception:
                logger.warning("doc_conflict_detection_failed")
                return []

    async def detect_answer_conflicts(self, answer: str, chunks: list[Chunk]) -> float:
        """Check if the answer contradicts the evidence. Returns contradiction rate 0-1."""
        evidence_block = format_evidence_block(chunks)
        prompt = ANSWER_CONTRADICTION_PROMPT.format(answer=answer, evidence_block=evidence_block)

        try:
            result = await self._llm.generate_structured(prompt, ContradictionResponse)
            rate = max(0.0, min(1.0, result.contradiction_rate))
        except Exception:
            try:
                raw = await self._llm.generate(prompt)
                data = json.loads(raw)
                rate = max(0.0, min(1.0, float(data.get("contradiction_rate", 0.0))))
            except Exception:
                logger.warning("answer_conflict_detection_failed")
                rate = 0.0

        logger.info("answer_contradiction_rate", rate=round(rate, 4))
        return rate
