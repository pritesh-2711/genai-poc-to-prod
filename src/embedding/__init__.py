"""Embedding providers — plug-and-play via BaseEmbedder.

All providers share the same interface:
    embed(texts: list[str]) -> list[list[float]]
    embed_one(text: str)    -> list[float]

Swap providers without changing any consumer code:
    LocalEmbedder()                         # sentence-transformers, fully local
    OllamaEmbedder(model="nomic-embed-text") # local Ollama server
    OpenAIEmbedder(model="text-embedding-3-small")  # OpenAI API
"""

from .base import BaseEmbedder
from .local import LocalEmbedder
from .ollama import OllamaEmbedder
from .openai import OpenAIEmbedder

__all__ = [
    "BaseEmbedder",
    "LocalEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
]
