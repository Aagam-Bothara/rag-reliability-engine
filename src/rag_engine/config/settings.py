"""Central configuration via Pydantic Settings. All values driven by env vars."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    google_api_key: str = ""

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100

    # LLM / Gemini
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 4096

    # Retrieval
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 10
    retrieval_fallback_expand_k: int = 100

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_pct: float = 0.15

    # Retrieval quality scoring weights
    rq_proceed_threshold: float = 0.55
    rq_fallback_threshold: float = 0.25
    rq_w_relevance: float = 0.45
    rq_w_margin: float = 0.20
    rq_w_coverage: float = 0.15
    rq_w_consistency: float = 0.20

    # Confidence scoring weights
    conf_alpha: float = 0.50
    conf_beta: float = 0.35
    conf_gamma: float = 0.15

    # Verification thresholds (normal mode)
    groundedness_pass_threshold: float = 0.7
    groundedness_warn_threshold: float = 0.5
    contradiction_pass_threshold: float = 0.2
    contradiction_warn_threshold: float = 0.4

    # Strict mode overrides
    strict_rq_proceed_threshold: float = 0.70
    strict_groundedness_pass_threshold: float = 0.85
    strict_contradiction_pass_threshold: float = 0.1

    # Storage paths
    sqlite_doc_db_path: str = "data/rag.db"
    sqlite_trace_db_path: str = "data/traces.db"
    embedding_cache_db_path: str = "data/embedding_cache.db"
    faiss_index_path: str = "data/faiss_index"
    bm25_index_path: str = "data/bm25_index"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    default_latency_budget_ms: int = 5000

    # Cross-encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Auth / JWT
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_keys: str = ""  # comma-separated list of valid API keys

    # Rate limiting
    rate_limit_requests_per_minute: int = 60

    model_config = {"env_file": ".env", "env_prefix": "RAG_"}
