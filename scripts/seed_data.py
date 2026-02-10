"""Seed the system with sample documents for development."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.chunking.structure_chunker import StructureChunker
from rag_engine.config.settings import Settings
from rag_engine.embeddings.openai_embedder import OpenAIEmbedder
from rag_engine.ingestion.parser_registry import create_default_registry
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.keyword_search.bm25_index import BM25Index
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore

SAMPLE_DOCS = [
    {
        "filename": "ai_overview.md",
        "content": """# Artificial Intelligence Overview

## What is AI?
Artificial Intelligence (AI) is the simulation of human intelligence in machines. These machines are programmed to think like humans and mimic their actions.

## Types of AI
There are three main types of AI:
1. **Narrow AI** - designed for specific tasks (e.g., Siri, chess engines)
2. **General AI** - hypothetical AI with human-level intelligence
3. **Super AI** - hypothetical AI surpassing human intelligence

## Machine Learning
Machine learning is a subset of AI that enables systems to learn from data. Key approaches include:
- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Deep Learning
Deep learning uses neural networks with many layers. It has been particularly successful in:
- Image recognition
- Natural language processing
- Speech recognition
""",
    },
    {
        "filename": "rag_systems.md",
        "content": """# Retrieval-Augmented Generation (RAG)

## What is RAG?
RAG is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating answers.

## Components of RAG
A typical RAG system includes:
1. **Document Store** - stores the knowledge base
2. **Retriever** - finds relevant documents for a query
3. **Generator** - produces answers using retrieved context

## Benefits of RAG
- Reduces hallucination by grounding answers in evidence
- Can be updated without retraining
- Provides citations for generated answers
- Works with domain-specific knowledge

## Challenges
- Retrieval quality directly impacts answer quality
- Latency can be high with large knowledge bases
- Chunk size and overlap affect performance
""",
    },
]


async def main():
    settings = Settings()

    # Ensure data directories exist
    Path(settings.sqlite_doc_db_path).parent.mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(settings.sqlite_doc_db_path)
    await doc_store.initialize()

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )

    vector_store = FAISSVectorStore(
        dimensions=settings.embedding_dimensions,
        index_path=settings.faiss_index_path,
    )

    bm25_index = BM25Index(index_path=settings.bm25_index_path)

    chunker = StructureChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_pct=settings.chunk_overlap_pct,
    )

    pipeline = IngestionPipeline(
        parser_registry=create_default_registry(),
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        doc_store=doc_store,
    )

    for doc in SAMPLE_DOCS:
        # Write to temp file
        tmp = Path(tempfile.mktemp(suffix=".md"))
        tmp.write_text(doc["content"])
        try:
            result = await pipeline.ingest_file(str(tmp), {"title": doc["filename"]})
            print(f"Ingested {doc['filename']}: {result.chunks_created} chunks")
        finally:
            tmp.unlink(missing_ok=True)

    print(f"\nTotal documents: {await doc_store.count_documents()}")
    print(f"Total chunks: {await doc_store.count_chunks()}")
    print(f"Vector index size: {vector_store.size}")


if __name__ == "__main__":
    asyncio.run(main())
