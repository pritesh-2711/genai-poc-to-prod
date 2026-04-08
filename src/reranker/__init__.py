"""Reranker providers — plug-and-play via BaseReranker.

All providers share the same interface:
    rerank(query, chunks, top_k) -> list[dict]

Swap providers without changing any consumer code:
    CrossEncoderReranker(model="BAAI/bge-reranker-base")
"""

from .base import BaseReranker
from .cross_encoder import CrossEncoderReranker

__all__ = [
    "BaseReranker",
    "CrossEncoderReranker",
]
