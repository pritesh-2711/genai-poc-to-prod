"""OpenAI embedding — calls the OpenAI Embeddings API.

Requires OPENAI_API_KEY in environment or passed explicitly.

Usage:
    from src.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder()                                       # text-embedding-3-small
    embedder = OpenAIEmbedder(model="text-embedding-3-large")        # higher quality
    vectors = embedder.embed(["hello world", "foo bar"])
"""

import os

from openai import OpenAI

from ..core.exceptions import ConfigurationError
from .base import BaseEmbedder

_BATCH_SIZE = 512   # OpenAI allows up to 2048 inputs per request; 512 is safe


class OpenAIEmbedder(BaseEmbedder):
    """Embeds text using the OpenAI Embeddings API.

    Supports batched requests — texts are sent in chunks of up to 512
    to stay well within OpenAI's per-request limits.

    Args:
        model:   OpenAI embedding model name.
                 Defaults to "text-embedding-3-small" (cheap, fast, good).
        api_key: Optional API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError(
                "OpenAIEmbedder requires an API key. "
                "Set OPENAI_API_KEY in your environment or pass api_key=."
            )
        self._client = OpenAI(api_key=key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the OpenAI Embeddings API.

        Splits large batches into chunks of 512 to respect API limits.

        Returns:
            List of float vectors in the same order as the input texts.
        """
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # response.data is sorted by index, so order is preserved
            vectors.extend(item.embedding for item in response.data)
        return vectors
