"""Multi-hop query decomposition via LLM."""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag_engine.config.constants import MAX_SUB_QUESTIONS
from rag_engine.generation.prompt_templates import QUERY_DECOMPOSITION_PROMPT
from rag_engine.models.domain import DecomposedQuery
from rag_engine.observability.logger import get_logger

logger = get_logger("decomposition")


class DecompositionResponse(BaseModel):
    sub_questions: list[str]
    synthesis_instruction: str


class QueryDecomposer:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def decompose(self, query: str) -> DecomposedQuery:
        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)

        try:
            result = await self._llm.generate_structured(prompt, DecompositionResponse)
            sub_questions = result.sub_questions[:MAX_SUB_QUESTIONS]
            synthesis = result.synthesis_instruction
        except Exception:
            # Fallback: try plain generation and parse JSON
            try:
                raw = await self._llm.generate(prompt)
                data = json.loads(raw)
                sub_questions = data.get("sub_questions", [query])[:MAX_SUB_QUESTIONS]
                synthesis = data.get("synthesis_instruction", "Combine the answers.")
            except Exception:
                logger.warning("decomposition_failed", query=query)
                sub_questions = [query]
                synthesis = ""

        if not sub_questions:
            sub_questions = [query]

        logger.info(
            "decomposed",
            original=query,
            sub_questions=len(sub_questions),
        )

        return DecomposedQuery(
            original=query,
            sub_questions=sub_questions,
            synthesis_instruction=synthesis,
        )
