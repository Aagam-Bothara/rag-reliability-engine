"""SQLite-backed document and chunk store."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from rag_engine.models.domain import Chunk, Document
from rag_engine.storage.migrations import initialize_doc_db


class SQLiteDocStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        await initialize_doc_db(self._db_path)

    async def save_document(self, doc: Document) -> str:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO documents (doc_id, source, content_type, metadata, raw_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    doc.source,
                    doc.content_type,
                    json.dumps(doc.metadata),
                    doc.raw_text,
                    doc.created_at.isoformat(),
                ),
            )
            await db.commit()
        return doc.doc_id

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, chunk_index, metadata, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.text,
                        c.index,
                        json.dumps(c.metadata),
                        c.token_count,
                    )
                    for c in chunks
                ],
            )
            await db.commit()

    async def get_document(self, doc_id: str) -> Document | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return Document(
                    doc_id=row["doc_id"],
                    source=row["source"],
                    content_type=row["content_type"],
                    metadata=json.loads(row["metadata"]),
                    raw_text=row["raw_text"],
                    created_at=datetime.fromisoformat(row["created_at"]).replace(
                        tzinfo=timezone.utc
                    ),
                )

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_chunk(row)

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ) as cursor:
                rows = await cursor.fetchall()
                return {row["chunk_id"]: self._row_to_chunk(row) for row in rows}

    async def get_chunks_by_doc(self, doc_id: str) -> list[Chunk]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_chunk(row) for row in rows]

    async def get_all_chunks(self) -> list[Chunk]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index") as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_chunk(row) for row in rows]

    async def count_documents(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM documents") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def count_chunks(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM chunks") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    @staticmethod
    def _row_to_chunk(row: aiosqlite.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            index=row["chunk_index"],
            metadata=json.loads(row["metadata"]),
            token_count=row["token_count"],
        )
