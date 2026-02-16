"""Master query pipeline orchestrator — the heart of the online path."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from rag_engine.config.settings import Settings
from rag_engine.generation.answer_generator import AnswerGenerator
from rag_engine.models.domain import GenerationResult, RetrievalCandidate
from rag_engine.models.schemas import Citation, DebugInfo, QueryRequest
from rag_engine.models.schemas import QueryResponse as QueryResponseSchema
from rag_engine.observability.logger import get_logger
from rag_engine.observability.metrics import log_generation_metrics, log_retrieval_metrics
from rag_engine.observability.tracing import TraceContext
from rag_engine.query.decomposition import QueryDecomposer
from rag_engine.query.understanding import QueryUnderstanding
from rag_engine.retrieval.fallback import FallbackManager
from rag_engine.retrieval.hybrid_retriever import HybridRetrieverImpl
from rag_engine.retrieval.reranker_cross_encoder import CrossEncoderReranker
from rag_engine.scoring.confidence import ConfidenceScorer
from rag_engine.scoring.reason_codes import ReasonCode
from rag_engine.scoring.retrieval_quality import RetrievalQualityScorer
from rag_engine.storage.sqlite_trace_store import SQLiteTraceStore
from rag_engine.verification.contradiction import ContradictionDetector
from rag_engine.verification.decision import VerificationDecisionMaker
from rag_engine.verification.groundedness import GroundednessChecker
from rag_engine.verification.self_consistency import SelfConsistencyChecker

logger = get_logger("query_pipeline")


class QueryPipeline:
    def __init__(
        self,
        query_understanding: QueryUnderstanding,
        decomposer: QueryDecomposer,
        hybrid_retriever: HybridRetrieverImpl,
        reranker: CrossEncoderReranker,
        rq_scorer: RetrievalQualityScorer,
        fallback_manager: FallbackManager,
        answer_generator: AnswerGenerator,
        groundedness_checker: GroundednessChecker,
        contradiction_detector: ContradictionDetector,
        self_consistency_checker: SelfConsistencyChecker,
        verification_decider: VerificationDecisionMaker,
        confidence_scorer: ConfidenceScorer,
        trace_store: SQLiteTraceStore,
        settings: Settings,
    ) -> None:
        self._qu = query_understanding
        self._decomposer = decomposer
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._rq_scorer = rq_scorer
        self._fallback = fallback_manager
        self._generator = answer_generator
        self._groundedness = groundedness_checker
        self._contradiction = contradiction_detector
        self._self_consistency = self_consistency_checker
        self._verification = verification_decider
        self._confidence = confidence_scorer
        self._trace_store = trace_store
        self._settings = settings

    async def execute(self, request: QueryRequest) -> QueryResponseSchema:
        trace = TraceContext()
        deadline = time.monotonic() + request.latency_budget_ms / 1000

        # STEP 1: Query Understanding
        with trace.span("query_understanding"):
            processed = await self._qu.process(request.query)

        # STEP 2: Query Decomposition
        with trace.span("decomposition"):
            decomposed = await self._decomposer.decompose(processed.normalized)

        # STEP 3: Hybrid Retrieval (for each sub-question)
        all_candidates: list[RetrievalCandidate] = []
        with trace.span("retrieval"):
            for sq in decomposed.sub_questions:
                candidates = await self._retriever.retrieve(
                    sq,
                    top_k_bm25=self._settings.bm25_top_k,
                    top_k_vector=self._settings.vector_top_k,
                )
                all_candidates.extend(candidates)
            all_candidates = self._deduplicate(all_candidates)

        # STEP 4: Reranking
        with trace.span("reranking"):
            reranked = await self._reranker.rerank(
                processed.normalized, all_candidates, top_n=self._settings.rerank_top_n
            )

        # STEP 5: Retrieval Quality Assessment
        with trace.span("rq_scoring"):
            rq_score, rq_reasons = self._rq_scorer.score(reranked)

        log_retrieval_metrics(
            trace.trace_id,
            rq_score,
            [c.score for c in reranked],
            len(reranked),
            len(set(c.chunk.doc_id for c in reranked)),
        )

        # STEP 6: Decision gate
        proceed_threshold = (
            self._settings.strict_rq_proceed_threshold
            if request.mode == "strict"
            else self._settings.rq_proceed_threshold
        )

        if rq_score < self._settings.rq_fallback_threshold:
            return self._build_abstain_response(rq_score, rq_reasons, trace, request)

        if rq_score < proceed_threshold:
            with trace.span("fallback"):
                fallback_result = await self._fallback.fallback_retrieve(
                    processed.normalized, self._generator._llm
                )
                if fallback_result.decision == "abstain":
                    reasons = rq_reasons + [ReasonCode.FALLBACK_FAILED]
                    return self._build_abstain_response(
                        rq_score, reasons, trace, request
                    )
                reranked = fallback_result.candidates
                rq_score = fallback_result.quality_score
                rq_reasons.append(ReasonCode.FALLBACK_USED)

        # STEP 7: Answer Generation
        with trace.span("generation"):
            gen_result = await self._generator.generate(
                processed.normalized, reranked, decomposed, request.mode
            )

        # STEP 7.5: Detect self-admitted insufficient evidence (RQ-aware)
        if self._answer_admits_ignorance(gen_result.answer):
            reasons = rq_reasons + [ReasonCode.LOW_GROUNDEDNESS]
            if rq_score >= self._settings.rq_proceed_threshold:
                # High-RQ ignorance: evidence was good, LLM hedged — clarify
                logger.info(
                    "answer_admits_ignorance_clarify",
                    query=processed.normalized,
                    rq=round(rq_score, 4),
                )
                return self._build_clarify_response(
                    gen_result, rq_score, reasons, trace, request
                )
            else:
                # Low-RQ ignorance: evidence was poor — hard abstain
                logger.info(
                    "answer_admits_ignorance_abstain",
                    query=processed.normalized,
                    rq=round(rq_score, 4),
                )
                return self._build_abstain_response(rq_score, reasons, trace, request)

        # STEP 8: Verification
        with trace.span("verification"):
            remaining_ms = (deadline - time.monotonic()) * 1000
            evidence_chunks = [c.chunk for c in reranked]

            groundedness_score, contradiction_rate = await asyncio.gather(
                self._groundedness.check(gen_result.answer, evidence_chunks, processed.normalized),
                self._contradiction.detect_answer_conflicts(
                    gen_result.answer, evidence_chunks
                ),
            )

            sc_score = None
            if remaining_ms > 1500:
                sc_score = await self._self_consistency.check(
                    processed.normalized, evidence_chunks, gen_result.answer
                )

            verification = self._verification.decide(
                groundedness_score, contradiction_rate, sc_score, request.mode
            )

        # STEP 9: Final Confidence
        confidence = self._confidence.score(
            rq_score, groundedness_score, contradiction_rate
        )

        log_generation_metrics(
            trace.trace_id,
            groundedness_score,
            contradiction_rate,
            confidence,
            verification.decision,
        )

        # STEP 10: Build Response
        all_reasons = rq_reasons + verification.reason_codes
        decision = self._map_decision(verification.decision, request.mode)

        if decision == "abstain":
            answer_text = (
                "I cannot provide a reliable answer to this question. "
                "The evidence is insufficient or contradictory."
            )
        elif decision == "clarify":
            answer_text = (
                gen_result.answer
                + "\n\nNote: This answer has moderate uncertainty. "
                "Some claims may not be fully supported by the available evidence."
            )
        else:
            answer_text = gen_result.answer

        response = QueryResponseSchema(
            answer=answer_text,
            citations=[
                Citation(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    text_snippet=c.text[:200],
                )
                for c in gen_result.cited_chunks
            ],
            confidence=round(confidence, 4),
            decision=decision,
            reasons=[str(r) for r in all_reasons],
            debug=DebugInfo(
                retrieval_quality=round(rq_score, 4),
                rerank_top_scores=[round(c.score, 4) for c in reranked[:5]],
                trace_id=trace.trace_id,
                latency_ms=round(trace.elapsed_ms, 2),
            ),
        )

        # Save trace (fire and forget)
        trace_obj = trace.to_trace(
            query=request.query,
            rq_score=rq_score,
            confidence=confidence,
            decision=decision,
            reason_codes=[str(r) for r in all_reasons],
        )
        asyncio.create_task(self._trace_store.save_trace(trace_obj))

        return response

    async def execute_stream(
        self, request: QueryRequest
    ) -> AsyncGenerator[dict, None]:
        """Execute the pipeline with streaming. Yields SSE-formatted dicts:
        - {"event": "token", "data": "<text chunk>"}
        - {"event": "metadata", "data": "<json payload>"}
        - {"event": "done", "data": ""}
        """
        trace = TraceContext()
        deadline = time.monotonic() + request.latency_budget_ms / 1000

        # STEPS 1-6: Same as execute() — non-streaming
        with trace.span("query_understanding"):
            processed = await self._qu.process(request.query)

        with trace.span("decomposition"):
            decomposed = await self._decomposer.decompose(processed.normalized)

        all_candidates: list[RetrievalCandidate] = []
        with trace.span("retrieval"):
            for sq in decomposed.sub_questions:
                candidates = await self._retriever.retrieve(
                    sq,
                    top_k_bm25=self._settings.bm25_top_k,
                    top_k_vector=self._settings.vector_top_k,
                )
                all_candidates.extend(candidates)
            all_candidates = self._deduplicate(all_candidates)

        with trace.span("reranking"):
            reranked = await self._reranker.rerank(
                processed.normalized, all_candidates, top_n=self._settings.rerank_top_n
            )

        with trace.span("rq_scoring"):
            rq_score, rq_reasons = self._rq_scorer.score(reranked)

        log_retrieval_metrics(
            trace.trace_id,
            rq_score,
            [c.score for c in reranked],
            len(reranked),
            len(set(c.chunk.doc_id for c in reranked)),
        )

        proceed_threshold = (
            self._settings.strict_rq_proceed_threshold
            if request.mode == "strict"
            else self._settings.rq_proceed_threshold
        )

        # Early exit: abstain
        if rq_score < self._settings.rq_fallback_threshold:
            response = self._build_abstain_response(rq_score, rq_reasons, trace, request)
            yield {"event": "metadata", "data": response.model_dump_json()}
            yield {"event": "done", "data": ""}
            return

        # Fallback if needed
        if rq_score < proceed_threshold:
            with trace.span("fallback"):
                fallback_result = await self._fallback.fallback_retrieve(
                    processed.normalized, self._generator._llm
                )
                if fallback_result.decision == "abstain":
                    reasons = rq_reasons + [ReasonCode.FALLBACK_FAILED]
                    response = self._build_abstain_response(rq_score, reasons, trace, request)
                    yield {"event": "metadata", "data": response.model_dump_json()}
                    yield {"event": "done", "data": ""}
                    return
                reranked = fallback_result.candidates
                rq_score = fallback_result.quality_score
                rq_reasons.append(ReasonCode.FALLBACK_USED)

        # STEP 7: Stream generation
        gen_result = None
        with trace.span("generation"):
            async for chunk_text, result in self._generator.generate_stream(
                processed.normalized, reranked, decomposed, request.mode
            ):
                if chunk_text is not None:
                    yield {"event": "token", "data": chunk_text}
                if result is not None:
                    gen_result = result

        # STEP 7.5: Check for self-admitted ignorance
        if self._answer_admits_ignorance(gen_result.answer):
            reasons = rq_reasons + [ReasonCode.LOW_GROUNDEDNESS]
            if rq_score >= self._settings.rq_proceed_threshold:
                response = self._build_clarify_response(
                    gen_result, rq_score, reasons, trace, request
                )
            else:
                response = self._build_abstain_response(rq_score, reasons, trace, request)
            yield {"event": "metadata", "data": response.model_dump_json()}
            yield {"event": "done", "data": ""}
            return

        # STEPS 8-10: Verification, confidence, response building
        with trace.span("verification"):
            remaining_ms = (deadline - time.monotonic()) * 1000
            evidence_chunks = [c.chunk for c in reranked]

            groundedness_score, contradiction_rate = await asyncio.gather(
                self._groundedness.check(
                    gen_result.answer, evidence_chunks, processed.normalized
                ),
                self._contradiction.detect_answer_conflicts(
                    gen_result.answer, evidence_chunks
                ),
            )

            sc_score = None
            if remaining_ms > 1500:
                sc_score = await self._self_consistency.check(
                    processed.normalized, evidence_chunks, gen_result.answer
                )

            verification = self._verification.decide(
                groundedness_score, contradiction_rate, sc_score, request.mode
            )

        confidence = self._confidence.score(
            rq_score, groundedness_score, contradiction_rate
        )

        log_generation_metrics(
            trace.trace_id,
            groundedness_score,
            contradiction_rate,
            confidence,
            verification.decision,
        )

        all_reasons = rq_reasons + verification.reason_codes
        decision = self._map_decision(verification.decision, request.mode)

        metadata = QueryResponseSchema(
            answer=gen_result.answer,
            citations=[
                Citation(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    text_snippet=c.text[:200],
                )
                for c in gen_result.cited_chunks
            ],
            confidence=round(confidence, 4),
            decision=decision,
            reasons=[str(r) for r in all_reasons],
            debug=DebugInfo(
                retrieval_quality=round(rq_score, 4),
                rerank_top_scores=[round(c.score, 4) for c in reranked[:5]],
                trace_id=trace.trace_id,
                latency_ms=round(trace.elapsed_ms, 2),
            ),
        )

        # Save trace
        trace_obj = trace.to_trace(
            query=request.query,
            rq_score=rq_score,
            confidence=confidence,
            decision=decision,
            reason_codes=[str(r) for r in all_reasons],
        )
        asyncio.create_task(self._trace_store.save_trace(trace_obj))

        yield {"event": "metadata", "data": metadata.model_dump_json()}
        yield {"event": "done", "data": ""}

    def _build_abstain_response(
        self,
        rq_score: float,
        reasons: list,
        trace: TraceContext,
        request: QueryRequest,
    ) -> QueryResponseSchema:
        trace_obj = trace.to_trace(
            query=request.query,
            rq_score=rq_score,
            confidence=0.0,
            decision="abstain",
            reason_codes=[str(r) for r in reasons],
        )
        asyncio.create_task(self._trace_store.save_trace(trace_obj))

        return QueryResponseSchema(
            answer="I cannot provide a reliable answer. The retrieved evidence is insufficient for this question.",
            citations=[],
            confidence=0.0,
            decision="abstain",
            reasons=[str(r) for r in reasons],
            debug=DebugInfo(
                retrieval_quality=round(rq_score, 4),
                rerank_top_scores=[],
                trace_id=trace.trace_id,
                latency_ms=round(trace.elapsed_ms, 2),
            ),
        )

    def _build_clarify_response(
        self,
        gen_result: GenerationResult,
        rq_score: float,
        reasons: list,
        trace: TraceContext,
        request: QueryRequest,
    ) -> QueryResponseSchema:
        """Build a clarify response: answer + caveat + citations."""
        answer_text = (
            gen_result.answer
            + "\n\nNote: This answer has moderate uncertainty. "
            "Some claims may not be fully supported by the available evidence."
        )
        confidence = round(rq_score * 0.5, 4)

        trace_obj = trace.to_trace(
            query=request.query,
            rq_score=rq_score,
            confidence=confidence,
            decision="clarify",
            reason_codes=[str(r) for r in reasons],
        )
        asyncio.create_task(self._trace_store.save_trace(trace_obj))

        return QueryResponseSchema(
            answer=answer_text,
            citations=[
                Citation(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    text_snippet=c.text[:200],
                )
                for c in gen_result.cited_chunks
            ],
            confidence=confidence,
            decision="clarify",
            reasons=[str(r) for r in reasons],
            debug=DebugInfo(
                retrieval_quality=round(rq_score, 4),
                rerank_top_scores=[],
                trace_id=trace.trace_id,
                latency_ms=round(trace.elapsed_ms, 2),
            ),
        )

    @staticmethod
    def _deduplicate(
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """Deduplicate by chunk_id, keeping the highest score."""
        seen: dict[str, RetrievalCandidate] = {}
        for c in candidates:
            existing = seen.get(c.chunk.chunk_id)
            if existing is None or c.score > existing.score:
                seen[c.chunk.chunk_id] = c
        return list(seen.values())

    @staticmethod
    def _answer_admits_ignorance(answer: str) -> bool:
        """Detect when the generated answer itself says the evidence is insufficient.

        Only matches explicit refusal patterns — avoids false positives from
        legitimate phrases like 'not contained in the model weights'.
        """
        lower = answer.lower()
        # Full refusal patterns that unambiguously indicate the LLM can't answer
        refusal_patterns = [
            "do not contain information",
            "does not contain information",
            "do not contain the answer",
            "does not contain the answer",
            "do not contain the necessary",
            "do not contain the coordinates",
            "don't contain information",
            "doesn't contain information",
            "cannot answer the question",
            "cannot answer this question",
            "unable to answer",
            "i cannot provide an answer",
            "i am unable to",
            "no relevant information",
            "outside the scope of",
            "is not discussed in",
            "are not discussed in",
            "not contain any information",
            "do not address",
            "does not address",
            "not provided in the evidence",
        ]
        return any(phrase in lower for phrase in refusal_patterns)

    @staticmethod
    def _map_decision(verification_decision: str, mode: str) -> str:
        """Map verification decision to API response decision."""
        if verification_decision == "pass":
            return "answer"
        elif verification_decision == "warn":
            return "clarify"
        else:
            return "abstain"
