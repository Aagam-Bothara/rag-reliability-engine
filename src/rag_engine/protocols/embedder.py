"""Protocol for embedding providers."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...

    @property
    def dimensions(self) -> int: ...
