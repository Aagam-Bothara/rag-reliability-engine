"""FAISS vector store with ID mapping and persistence."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import faiss
import numpy as np

from rag_engine.observability.logger import get_logger

logger = get_logger("faiss_store")


class FAISSVectorStore:
    def __init__(self, dimensions: int, index_path: str | None = None) -> None:
        self._dimensions = dimensions
        self._index_path = index_path
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))
        self._id_to_chunk_id: dict[int, str] = {}
        self._chunk_id_to_int: dict[str, int] = {}
        self._next_id: int = 0
        self._write_lock = asyncio.Lock()

        if index_path:
            self._try_load(index_path)

    def _try_load(self, path: str) -> None:
        index_file = os.path.join(path, "index.faiss")
        mapping_file = os.path.join(path, "id_mapping.json")
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            self._index = faiss.read_index(index_file)
            with open(mapping_file) as f:
                data = json.load(f)
            self._id_to_chunk_id = {int(k): v for k, v in data["id_to_chunk_id"].items()}
            self._chunk_id_to_int = data["chunk_id_to_int"]
            self._next_id = data["next_id"]
            logger.info("faiss_loaded", size=self._index.ntotal, path=path)

    def add(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        int_ids = self._assign_int_ids(chunk_ids)
        self._index.add_with_ids(embeddings, np.array(int_ids, dtype=np.int64))
        logger.info("faiss_added", count=len(chunk_ids), total=self._index.ntotal)

    async def add_safe(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self.add, chunk_ids, embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self._index.search(query_embedding, min(top_k, self._index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            idx = int(idx)
            if idx == -1:
                continue
            chunk_id = self._id_to_chunk_id.get(idx)
            if chunk_id:
                results.append((chunk_id, float(score)))
        return results

    def save(self, path: str | None = None) -> None:
        path = path or self._index_path
        if not path:
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "id_mapping.json"), "w") as f:
            json.dump(
                {
                    "id_to_chunk_id": self._id_to_chunk_id,
                    "chunk_id_to_int": self._chunk_id_to_int,
                    "next_id": self._next_id,
                },
                f,
            )
        logger.info("faiss_saved", path=path, size=self._index.ntotal)

    @property
    def size(self) -> int:
        return self._index.ntotal

    def _assign_int_ids(self, chunk_ids: list[str]) -> list[int]:
        int_ids = []
        for cid in chunk_ids:
            if cid in self._chunk_id_to_int:
                int_ids.append(self._chunk_id_to_int[cid])
            else:
                self._id_to_chunk_id[self._next_id] = cid
                self._chunk_id_to_int[cid] = self._next_id
                int_ids.append(self._next_id)
                self._next_id += 1
        return int_ids
