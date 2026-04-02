"""Ollama embedding — calls a locally running Ollama server.

Requires Ollama to be running: https://ollama.com
Pull an embedding model first:  ollama pull nomic-embed-text

Usage:
    from src.embedding.ollama import OllamaEmbedder
    embedder = OllamaEmbedder()                                  # nomic-embed-text
    embedder = OllamaEmbedder(model="nomic-embed-text-v2-moe:latest")        # swap model
    vectors = embedder.embed(["hello world", "foo bar"])
"""

import requests

from ..core.exceptions import ConfigurationError
from .base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Embeds text using a locally running Ollama server.

    Calls POST /api/embeddings once per text (Ollama does not support
    batched embedding requests in its REST API).

    Args:
        model:    Ollama model name that supports embeddings.
                  Defaults to "nomic-embed-text-v2-moe:latest".
        base_url: Base URL of the Ollama server.
                  Defaults to "http://localhost:11434".
        timeout:  Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text-v2-moe:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama (sequential requests).

        Raises:
            ConfigurationError: If the Ollama server is unreachable.
            RuntimeError: If the server returns an unexpected response.
        """
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_one(text))
        return vectors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        try:
            response = requests.post(
                self._url,
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConfigurationError(
                f"Cannot reach Ollama at {self._url}. "
                "Is the Ollama server running? (ollama serve)"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            ) from exc

        data = response.json()
        if "embedding" not in data:
            raise RuntimeError(
                f"Unexpected Ollama response (no 'embedding' key): {data}"
            )
        return data["embedding"]
