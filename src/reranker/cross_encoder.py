"""Cross-encoder reranker using sentence-transformers.

Scores (query, passage) pairs with a cross-encoder model and returns the
top-K chunks sorted by relevance score. Model is configurable via config.yaml
(reranker.model). Recommended models:
  - BAAI/bge-reranker-base   (fast, good quality)
  - BAAI/bge-reranker-large  (slower, better quality)
  - BAAI/bge-reranker-v2-m3  (multilingual)
"""

import logging

from sentence_transformers import CrossEncoder

from .base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a sentence-transformers CrossEncoder model.

    The model is loaded once at construction time and reused across calls.
    For GPU acceleration set device="cuda" in config.yaml (reranker.device).
    """

    def __init__(self, model: str, device: str = "cpu") -> None:
        """Initialise and load the CrossEncoder model.

        Args:
            model:  HuggingFace model name or local path.
            device: Torch device string — "cpu" or "cuda".
        """
        logger.info(f"Loading CrossEncoder reranker: model={model}, device={device}")
        self._model = CrossEncoder(model, device=device)
        self._model_name = model
        logger.info("CrossEncoder reranker loaded.")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Score chunks against the query and return the top-K by score.

        Args:
            query:   User query string used as the cross-encoder left input.
            chunks:  Candidate chunks; each must have a ``chunk_content`` key.
            top_k:   Maximum number of results to return.

        Returns:
            Top-K chunks (copies) sorted descending by ``rerank_score``.
            Returns an empty list when ``chunks`` is empty.
        """
        if not chunks:
            return []

        passages = [c.get("chunk_content", "") for c in chunks]
        pairs = [(query, p) for p in passages]

        scores: list[float] = self._model.predict(pairs).tolist()

        scored = [
            {**chunk, "rerank_score": score}
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]
