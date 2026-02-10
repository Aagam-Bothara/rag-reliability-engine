"""Google Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import json

from google import genai
from google.genai import types
from pydantic import BaseModel

from rag_engine.exceptions import GenerationError
from rag_engine.observability.logger import get_logger

logger = get_logger("gemini")


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if system:
                config.system_instruction = system

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            raise GenerationError(f"Gemini generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        system: str | None = None,
    ) -> BaseModel:
        try:
            config = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            if system:
                config.system_instruction = system

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            data = json.loads(response.text)
            return response_schema.model_validate(data)
        except Exception as e:
            raise GenerationError(f"Gemini structured generation failed: {e}") from e
