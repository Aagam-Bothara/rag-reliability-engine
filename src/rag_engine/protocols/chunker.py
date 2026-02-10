"""Protocol for text chunking."""

from __future__ import annotations

from typing import Protocol

from rag_engine.models.domain import Chunk


class Chunker(Protocol):
    def chunk(self, text: str, metadata: dict) -> list[Chunk]: ...
