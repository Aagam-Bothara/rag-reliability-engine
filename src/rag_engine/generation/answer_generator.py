"""Answer generation using evidence chunks and LLM."""

from __future__ import annotations

import re

from rag_engine.generation.prompt_templates import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_GENERATION_STRICT_SYSTEM,
    ANSWER_GENERATION_SYSTEM,
    format_decomposition_context,
    format_evidence_block,
)
from rag_engine.models.domain import (
    DecomposedQuery,
    GenerationResult,
    RetrievalCandidate,
)
from rag_engine.observability.logger import get_logger

logger = get_logger("generation")


class AnswerGenerator:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def generate(
        self,
        query: str,
        evidence: list[RetrievalCandidate],
        decomposition: DecomposedQuery | None = None,
        mode: str = "normal",
    ) -> GenerationResult:
        evidence_block = format_evidence_block(evidence)
        decomp_context = ""
        if decomposition and len(decomposition.sub_questions) > 1:
            decomp_context = format_decomposition_context(
                decomposition.sub_questions, decomposition.synthesis_instruction
            )

        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            evidence_block=evidence_block,
            decomposition_context=decomp_context,
        )

        system = ANSWER_GENERATION_STRICT_SYSTEM if mode == "strict" else ANSWER_GENERATION_SYSTEM

        answer = await self._llm.generate(prompt, system=system)

        # Parse cited chunk references from answer
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        cited_chunks = []
        cited_spans = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(evidence):
                chunk = evidence[idx - 1].chunk
                cited_chunks.append(chunk)
                cited_spans.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text[:200],
                    }
                )

        logger.info(
            "generated_answer",
            query_len=len(query),
            answer_len=len(answer),
            citations=len(cited_chunks),
        )

        return GenerationResult(
            answer=answer,
            cited_chunks=cited_chunks,
            cited_spans=cited_spans,
        )

    async def generate_stream(
        self,
        query: str,
        evidence: list[RetrievalCandidate],
        decomposition: DecomposedQuery | None = None,
        mode: str = "normal",
    ):
        """Yield (chunk_text, None) during streaming, then (None, GenerationResult) at end."""
        evidence_block = format_evidence_block(evidence)
        decomp_context = ""
        if decomposition and len(decomposition.sub_questions) > 1:
            decomp_context = format_decomposition_context(
                decomposition.sub_questions, decomposition.synthesis_instruction
            )

        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            evidence_block=evidence_block,
            decomposition_context=decomp_context,
        )

        system = ANSWER_GENERATION_STRICT_SYSTEM if mode == "strict" else ANSWER_GENERATION_SYSTEM

        full_answer = ""
        async for chunk in self._llm.generate_stream(prompt, system=system):
            full_answer += chunk
            yield chunk, None

        # Parse citations from full answer
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", full_answer))
        cited_chunks = []
        cited_spans = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(evidence):
                chunk = evidence[idx - 1].chunk
                cited_chunks.append(chunk)
                cited_spans.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text[:200],
                    }
                )

        logger.info(
            "generated_answer_stream",
            query_len=len(query),
            answer_len=len(full_answer),
            citations=len(cited_chunks),
        )

        yield (
            None,
            GenerationResult(
                answer=full_answer,
                cited_chunks=cited_chunks,
                cited_spans=cited_spans,
            ),
        )
