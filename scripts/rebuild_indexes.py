"""Rebuild FAISS and BM25 indexes from the document store."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.config.settings import Settings
from rag_engine.embeddings.openai_embedder import OpenAIEmbedder
from rag_engine.keyword_search.bm25_index import BM25Index
from rag_engine.storage.sqlite_doc_store import SQLiteDocStore
from rag_engine.vectorstore.faiss_store import FAISSVectorStore


async def main():
    settings = Settings()

    doc_store = SQLiteDocStore(settings.sqlite_doc_db_path)
    await doc_store.initialize()

    all_chunks = await doc_store.get_all_chunks()
    print(f"Found {len(all_chunks)} chunks in doc store")

    if not all_chunks:
        print("No chunks to index.")
        return

    # Rebuild BM25
    print("Rebuilding BM25 index...")
    bm25_index = BM25Index(index_path=settings.bm25_index_path)
    bm25_index.build(all_chunks)
    bm25_index.save()
    print(f"BM25 index built: {bm25_index.size} entries")

    # Rebuild FAISS
    print("Rebuilding FAISS index...")
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    vector_store = FAISSVectorStore(
        dimensions=settings.embedding_dimensions,
        index_path=settings.faiss_index_path,
    )

    texts = [c.text for c in all_chunks]
    chunk_ids = [c.chunk_id for c in all_chunks]

    embeddings = await embedder.embed_texts(texts)
    emb_array = np.array(embeddings, dtype=np.float32)
    vector_store.add(chunk_ids, emb_array)
    vector_store.save()
    print(f"FAISS index built: {vector_store.size} vectors")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
