"""OpenAI embedding provider using text-embedding-3-small."""

from __future__ import annotations

from openai import AsyncOpenAI

from rag_engine.exceptions import EmbeddingError
from rag_engine.observability.logger import get_logger

logger = get_logger("embeddings")


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        _dimensions: int = 1536,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._dimensions = _dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                response = await self._client.embeddings.create(input=batch, model=self._model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            logger.info("embedded_texts", count=len(texts), model=self._model)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed {len(texts)} texts: {e}") from e
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(input=[query], model=self._model)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}") from e
