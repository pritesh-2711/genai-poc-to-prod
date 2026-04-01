"""Local embedding using sentence-transformers (no external API needed).

Default model: all-MiniLM-L6-v2 — fast, small, good general-purpose quality.
Any sentence-transformers compatible model name works as a drop-in.

Usage:
    from src.embedding.local import LocalEmbedder
    embedder = LocalEmbedder()                                   # MiniLM-L6-v2
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")          # swap model
    vectors = embedder.embed(["hello world", "foo bar"])
"""

from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """Sentence-transformers embedder — runs fully locally, no API key needed.

    Args:
        model_name: Any sentence-transformers model name or local path.
                    Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        batch_size: Batch size for SentenceTransformer.encode().
        device:     "cpu", "cuda", or None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
