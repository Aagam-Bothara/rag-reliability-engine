"""Ingestion pipeline: parse -> chunk -> embed -> index -> store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np

from rag_engine.chunking.quality import (
    compute_coverage,
    detect_near_duplicates,
    filter_garbage_chunks,
)
from rag_engine.chunking.structure_chunker import StructureChunker
from rag_engine.protocols.embedder import Embedder
from rag_engine.ingestion.parser_registry import ParserRegistry
from rag_engine.keyword_search.bm25_index import BM25Index
from rag_engine.models.domain import Document
from rag_engine.models.schemas import IngestResponse
from rag_engine.observability.logger import get_logger
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger("ingestion")


class IngestionPipeline:
    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: StructureChunker,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
        doc_store: SQLiteDocStore,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._doc_store = doc_store

    async def ingest_file(
        self, file_path: str | Path, metadata: dict | None = None
    ) -> IngestResponse:
        file_path = Path(file_path)
        metadata = metadata or {}
        doc_id = str(uuid4())

        # 1. Parse file
        parser = self._parser_registry.get_parser(file_path.name)
        raw_text, enriched_metadata = await asyncio.to_thread(parser.parse, file_path, metadata)
        logger.info("parsed", doc_id=doc_id, source=str(file_path), chars=len(raw_text))

        # 2. Create and save document
        doc = Document(
            doc_id=doc_id,
            source=str(file_path),
            content_type=file_path.suffix.lower(),
            metadata=enriched_metadata,
            raw_text=raw_text,
        )
        await self._doc_store.save_document(doc)

        # 3. Chunk text
        chunks = await asyncio.to_thread(
            self._chunker.chunk, raw_text, {"doc_id": doc_id, **enriched_metadata}
        )

        # 4. Quality checks
        chunks = filter_garbage_chunks(chunks)
        detect_near_duplicates(chunks)
        compute_coverage(chunks, raw_text)

        if not chunks:
            return IngestResponse(doc_id=doc_id, chunks_created=0, status="no_chunks")

        # 5. Embed chunks
        texts = [c.text for c in chunks]
        embeddings = await self._embedder.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 6. Add to FAISS index
        emb_array = np.array(embeddings, dtype=np.float32)
        chunk_ids = [c.chunk_id for c in chunks]
        await self._vector_store.add_safe(chunk_ids, emb_array)

        # 7. Rebuild BM25 index (includes all existing chunks + new ones)
        await self._doc_store.save_chunks(chunks)
        all_chunks = await self._doc_store.get_all_chunks()
        await self._bm25_index.rebuild(all_chunks)

        # 8. Persist indexes
        await asyncio.to_thread(self._vector_store.save)
        await asyncio.to_thread(self._bm25_index.save)

        logger.info("ingested", doc_id=doc_id, chunks=len(chunks))
        return IngestResponse(doc_id=doc_id, chunks_created=len(chunks), status="indexed")
