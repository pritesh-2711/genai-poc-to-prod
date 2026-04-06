"""Base abstraction for embedding providers.

All embedders expose the same two-method interface:
    embed(texts)  → list of float vectors (batch)
    embed_one(text) → single float vector

This makes every downstream consumer (EmbeddingSemanticChunker, vector
stores, retrieval) fully provider-agnostic — swap LocalEmbedder for
OpenAIEmbedder or OllamaEmbedder without changing any calling code.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Plug-and-play interface for all embedding providers.

    Asymmetric retrieval models (e.g. nomic-embed-text-v2) require different
    task prefixes for documents vs. queries.  All callers should use:
      - embed()       for indexing passages (applies doc prefix if configured)
      - embed_query() for query-time embedding (applies query prefix if configured)
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document passages for indexing.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """

    def embed_one(self, text: str) -> list[float]:
        """Embed a single document string. Convenience wrapper over embed()."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Override in subclasses that use a different task prefix for queries
        (e.g. nomic-embed-text-v2: 'search_query: ').  Default falls back to
        embed_one() for symmetric models.
        """
        return self.embed_one(text)
