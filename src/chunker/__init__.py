"""Chunking pipeline — converts ExtractedRecord lists into LangChain Documents.

Three strategies, all sharing the BaseChunker interface:

    HierarchicalChunker       parent/child split — search small, read large
    TextTilingChunker         lexical topic-shift detection — no model needed
    EmbeddingSemanticChunker  embedding cosine similarity — plug in any embedder

All chunkers support:
    .chunk(text, metadata)              → list[Document]
    .chunk_records(records, source)     → list[Document]   (pipeline-aware)

Example:
    from src.chunker import HierarchicalChunker, EmbeddingSemanticChunker
    from src.embedding import LocalEmbedder

    # Hierarchical
    docs = HierarchicalChunker().chunk_records(text_records, source=pdf_path)

    # Semantic — swap embedder without changing chunker code
    chunker = EmbeddingSemanticChunker(embedder=LocalEmbedder())
    docs = chunker.chunk_records(text_records, source=pdf_path)
"""

from .base import BaseChunker
from .hierarchical import HierarchicalChunker
from .semantic import EmbeddingSemanticChunker, TextTilingChunker

__all__ = [
    "BaseChunker",
    "HierarchicalChunker",
    "TextTilingChunker",
    "EmbeddingSemanticChunker",
]
