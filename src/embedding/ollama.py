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


_NOMIC_MODELS = {
    "nomic-embed-text",
    "nomic-embed-text-v2-moe",
    "nomic-embed-text:latest",
    "nomic-embed-text-v2-moe:latest",
}

_DEFAULT_DOC_PREFIXES: dict[str, str] = {
    k: "search_document: " for k in _NOMIC_MODELS
}
_DEFAULT_QUERY_PREFIXES: dict[str, str] = {
    k: "search_query: " for k in _NOMIC_MODELS
}


class OllamaEmbedder(BaseEmbedder):
    """Embeds text using a locally running Ollama server.

    For asymmetric retrieval models (nomic-embed-text-*) the correct task
    prefix is prepended automatically:
      - "search_document: " when indexing passages  (embed / embed_one)
      - "search_query: "    when embedding queries  (embed_query)

    Args:
        model:        Ollama model name.
        base_url:     Ollama server base URL.
        timeout:      Per-request timeout in seconds.
        doc_prefix:   Override the document prefix (empty string to disable).
        query_prefix: Override the query prefix (empty string to disable).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text-v2-moe:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        doc_prefix: str | None = None,
        query_prefix: str | None = None,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._timeout = timeout

        model_key = model.lower()
        self._doc_prefix = (
            doc_prefix if doc_prefix is not None
            else _DEFAULT_DOC_PREFIXES.get(model_key, "")
        )
        self._query_prefix = (
            query_prefix if query_prefix is not None
            else _DEFAULT_QUERY_PREFIXES.get(model_key, "")
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document passages for indexing (applies doc prefix).

        Raises:
            ConfigurationError: If the Ollama server is unreachable.
            RuntimeError: If the server returns an unexpected response.
        """
        return [self._call(self._doc_prefix + t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single document passage (applies doc prefix)."""
        return self._call(self._doc_prefix + text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (applies query prefix)."""
        return self._call(self._query_prefix + text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> list[float]:
        try:
            response = requests.post(
                self._url,
                json={"model": self._model, "prompt": prompt},
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
