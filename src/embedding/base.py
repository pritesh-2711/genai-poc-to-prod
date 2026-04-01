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
    """Plug-and-play interface for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper over embed()."""
        return self.embed([text])[0]
