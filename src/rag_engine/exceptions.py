"""Custom exception hierarchy for the RAG engine."""


class RAGEngineError(Exception):
    """Base exception for all RAG engine errors."""


class IngestionError(RAGEngineError):
    """Error during document ingestion."""


class ParsingError(IngestionError):
    """Error parsing a document file."""


class ChunkingError(IngestionError):
    """Error during text chunking."""


class EmbeddingError(RAGEngineError):
    """Error generating embeddings."""


class RetrievalError(RAGEngineError):
    """Error during retrieval."""


class GenerationError(RAGEngineError):
    """Error during answer generation."""


class VerificationError(RAGEngineError):
    """Error during answer verification."""


class ConfigurationError(RAGEngineError):
    """Error in system configuration."""


class LatencyBudgetExceeded(RAGEngineError):
    """Latency budget was exceeded."""
