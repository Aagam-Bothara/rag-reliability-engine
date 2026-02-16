"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from rag_engine.api.auth import router as auth_router
from rag_engine.api.middleware import RequestTimingMiddleware
from rag_engine.api.routes_health import router as health_router
from rag_engine.api.routes_ingest import router as ingest_router
from rag_engine.api.routes_query import router as query_router
from rag_engine.chunking.structure_chunker import StructureChunker
from rag_engine.config.settings import Settings
from rag_engine.embeddings.cache import EmbeddingCache
from rag_engine.embeddings.cached_embedder import CachedEmbedder
from rag_engine.embeddings.openai_embedder import OpenAIEmbedder
from rag_engine.generation.answer_generator import AnswerGenerator
from rag_engine.generation.gemini_provider import GeminiProvider
from rag_engine.ingestion.parser_registry import create_default_registry
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.keyword_search.bm25_index import BM25Index
from rag_engine.observability.logger import get_logger, setup_logging
from rag_engine.pipeline.query_pipeline import QueryPipeline
from rag_engine.query.decomposition import QueryDecomposer
from rag_engine.query.understanding import QueryUnderstanding
from rag_engine.retrieval.fallback import FallbackManager
from rag_engine.retrieval.hybrid_retriever import HybridRetrieverImpl
from rag_engine.retrieval.reranker_cross_encoder import CrossEncoderReranker
from rag_engine.scoring.confidence import ConfidenceScorer
from rag_engine.scoring.retrieval_quality import RetrievalQualityScorer
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.storage.sqlite_trace_store import SQLiteTraceStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore
from rag_engine.verification.contradiction import ContradictionDetector
from rag_engine.verification.decision import VerificationDecisionMaker
from rag_engine.verification.groundedness import GroundednessChecker
from rag_engine.verification.self_consistency import SelfConsistencyChecker

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    setup_logging()

    # Ensure data directories exist
    for path in [
        settings.sqlite_doc_db_path,
        settings.sqlite_trace_db_path,
        settings.embedding_cache_db_path,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Storage
    doc_store = SQLiteDocStore(settings.sqlite_doc_db_path)
    await doc_store.initialize()
    trace_store = SQLiteTraceStore(settings.sqlite_trace_db_path)
    await trace_store.initialize()

    # Embedding (with cache)
    raw_embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        _dimensions=settings.embedding_dimensions,
    )
    embedding_cache = EmbeddingCache(settings.embedding_cache_db_path)
    await embedding_cache.initialize()
    embedder = CachedEmbedder(delegate=raw_embedder, cache=embedding_cache)

    # Vector store
    vector_store = FAISSVectorStore(
        dimensions=settings.embedding_dimensions,
        index_path=settings.faiss_index_path,
    )

    # BM25 index
    bm25_index = BM25Index(index_path=settings.bm25_index_path)
    if bm25_index.size == 0:
        # Rebuild from doc store if not loaded from disk
        all_chunks = await doc_store.get_all_chunks()
        if all_chunks:
            bm25_index.build(all_chunks)

    # LLM
    llm = GeminiProvider(api_key=settings.google_api_key, model=settings.gemini_model)

    # Reranker
    reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model)

    # Chunker
    chunker = StructureChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_pct=settings.chunk_overlap_pct,
    )

    # Retrieval
    hybrid_retriever = HybridRetrieverImpl(
        vector_store=vector_store,
        bm25_index=bm25_index,
        doc_store=doc_store,
        embedder=embedder,
        rrf_k=settings.rrf_k,
    )

    # Scoring
    rq_scorer = RetrievalQualityScorer(settings)
    confidence_scorer = ConfidenceScorer(settings)

    # Fallback
    fallback_manager = FallbackManager(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        rq_scorer=rq_scorer,
        settings=settings,
    )

    # Generation
    answer_generator = AnswerGenerator(llm=llm)

    # Query understanding
    query_understanding = QueryUnderstanding()
    decomposer = QueryDecomposer(llm=llm)

    # Verification
    groundedness_checker = GroundednessChecker(llm=llm)
    contradiction_detector = ContradictionDetector(llm=llm)
    self_consistency_checker = SelfConsistencyChecker(llm=llm)
    verification_decider = VerificationDecisionMaker(settings)

    # Ingestion pipeline
    parser_registry = create_default_registry()
    ingest_pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        doc_store=doc_store,
    )

    # Query pipeline
    query_pipeline = QueryPipeline(
        query_understanding=query_understanding,
        decomposer=decomposer,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        rq_scorer=rq_scorer,
        fallback_manager=fallback_manager,
        answer_generator=answer_generator,
        groundedness_checker=groundedness_checker,
        contradiction_detector=contradiction_detector,
        self_consistency_checker=self_consistency_checker,
        verification_decider=verification_decider,
        confidence_scorer=confidence_scorer,
        trace_store=trace_store,
        settings=settings,
    )

    # Attach to app state
    app.state.query_pipeline = query_pipeline
    app.state.ingest_pipeline = ingest_pipeline
    app.state.doc_store = doc_store
    app.state.vector_store = vector_store
    app.state.settings = settings

    logger.info(
        "startup_complete",
        docs=await doc_store.count_documents(),
        chunks=await doc_store.count_chunks(),
        index_size=vector_store.size,
    )

    yield

    # Shutdown: persist indexes
    vector_store.save()
    bm25_index.save()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Reliability Engine",
        version="1.0.0",
        description="Production-grade, failure-aware RAG system",
        lifespan=lifespan,
    )
    app.add_middleware(RequestTimingMiddleware)
    app.include_router(health_router, tags=["health"])
    app.include_router(auth_router)
    app.include_router(ingest_router, tags=["ingest"])
    app.include_router(query_router, tags=["query"])
    return app
