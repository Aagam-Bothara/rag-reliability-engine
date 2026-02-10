"""SQLite-backed embedding cache to avoid re-embedding identical text."""

from __future__ import annotations

import hashlib
import json

import aiosqlite

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding TEXT NOT NULL
)
"""


class EmbeddingCache:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(CREATE_CACHE_TABLE)
            await db.commit()

    async def get(self, text: str) -> list[float] | None:
        text_hash = self._hash(text)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                (text_hash,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return json.loads(row[0])

    async def get_batch(self, texts: list[str]) -> dict[int, list[float]]:
        """Return {index: embedding} for texts that are cached."""
        if not texts:
            return {}
        hashes = [(i, self._hash(t)) for i, t in enumerate(texts)]
        hash_to_idx = {h: i for i, h in hashes}
        placeholders = ",".join("?" for _ in hashes)
        hash_values = [h for _, h in hashes]

        result: dict[int, list[float]] = {}
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders})",
                hash_values,
            ) as cursor:
                async for row in cursor:
                    idx = hash_to_idx.get(row[0])
                    if idx is not None:
                        result[idx] = json.loads(row[1])
        return result

    async def put(self, text: str, embedding: list[float]) -> None:
        text_hash = self._hash(text)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding) VALUES (?, ?)",
                (text_hash, json.dumps(embedding)),
            )
            await db.commit()

    async def put_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        if not texts:
            return
        rows = [
            (self._hash(t), json.dumps(e)) for t, e in zip(texts, embeddings)
        ]
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding) VALUES (?, ?)",
                rows,
            )
            await db.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
