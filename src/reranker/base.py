"""Base abstraction for reranking providers.

All rerankers expose the same interface:
    rerank(query, chunks, top_k) -> list[dict]

This keeps every orchestrator consumer fully provider-agnostic —
swap CrossEncoderReranker for any other reranker without changing callers.
"""

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Plug-and-play interface for all reranking providers.

    A reranker takes a query and a list of candidate chunks (each a dict with
    at least a ``chunk_content`` key) and returns the top-K most relevant
    chunks sorted by relevance score, with a ``rerank_score`` key added.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Score and filter chunks by relevance to the query.

        Args:
            query:   The user's query string.
            chunks:  Candidate chunks; each dict must have ``chunk_content``.
            top_k:   Maximum number of chunks to return.

        Returns:
            Top-K chunks sorted descending by ``rerank_score``, with the
            ``rerank_score`` (float) key added to each dict.
        """
