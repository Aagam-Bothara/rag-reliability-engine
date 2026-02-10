"""Protocol for LLM providers."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str: ...

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel: ...
