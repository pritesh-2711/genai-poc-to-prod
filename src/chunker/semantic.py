"""Semantic chunkers — split on meaning, not character count.

Two implementations, both extending BaseChunker:

TextTilingChunker
    Lexical semantic chunking based on the TextTiling algorithm.
    No embeddings or external models required.
    Uses bag-of-words cosine similarity between sliding sentence windows
    to detect topic shifts.

EmbeddingSemanticChunker
    Embedding-based semantic chunking.
    Uses any BaseEmbedder (LocalEmbedder, OllamaEmbedder, OpenAIEmbedder)
    to embed each sentence, then splits where consecutive sentence similarity
    drops below a threshold.
    Plug-and-play: pass any BaseEmbedder instance at construction time.

Usage:
    from src.chunker.semantic import TextTilingChunker, EmbeddingSemanticChunker
    from src.embedding.local import LocalEmbedder

    # Lexical — no model needed
    chunker = TextTilingChunker(window_size=3, depth_threshold=0.3)

    # Embedding-based — plug in any embedder
    chunker = EmbeddingSemanticChunker(embedder=LocalEmbedder())
    chunker = EmbeddingSemanticChunker(embedder=OpenAIEmbedder())
    chunker = EmbeddingSemanticChunker(embedder=OllamaEmbedder())

    docs = chunker.chunk("long text...", metadata={"source": "doc.pdf"})
    docs = chunker.chunk_records(text_records, source=pdf_path)
"""

import re
from collections import Counter
from typing import Any, Optional

import numpy as np
from langchain_core.documents import Document

from ..embedding.base import BaseEmbedder
from .base import BaseChunker


# ---------------------------------------------------------------------------
# Shared sentence tokenizer
# ---------------------------------------------------------------------------

def _sentence_tokenize(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# TextTiling helpers (ported from notebook)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def _block_score(block_a: list[list[str]], block_b: list[list[str]]) -> float:
    """Bag-of-words cosine similarity between two sentence blocks."""
    counter_a: Counter = Counter(t for sent in block_a for t in sent)
    counter_b: Counter = Counter(t for sent in block_b for t in sent)

    vocab = set(counter_a) | set(counter_b)
    dot = sum(counter_a[w] * counter_b[w] for w in vocab)
    norm_a = sum(v ** 2 for v in counter_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in counter_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _depth_score(scores: list[float], i: int) -> float:
    """Valley depth at position i — captures how much the score dips
    relative to the nearest left and right peaks."""
    left_peak = scores[i]
    for j in range(i - 1, -1, -1):
        if scores[j] >= left_peak:
            left_peak = scores[j]
        else:
            break

    right_peak = scores[i]
    for j in range(i + 1, len(scores)):
        if scores[j] >= right_peak:
            right_peak = scores[j]
        else:
            break

    return (left_peak - scores[i]) + (right_peak - scores[i])


# ---------------------------------------------------------------------------
# Embedding similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ---------------------------------------------------------------------------
# TextTilingChunker
# ---------------------------------------------------------------------------

class TextTilingChunker(BaseChunker):
    """Lexical semantic chunker — no embeddings, no external models.

    Algorithm:
        1. Split text into sentences and tokenize each.
        2. Compute bag-of-words cosine similarity between adjacent sliding
           windows of sentences at every sentence boundary (gap score).
        3. Convert gap scores to depth scores (valley depth).
        4. Insert chunk boundaries at depth scores above depth_threshold.

    Args:
        window_size:          Number of sentences on each side of a boundary
                              used to compute the block similarity score.
        depth_threshold:      Depth score above which a boundary is inserted.
                              Higher = fewer, larger chunks.
        min_chunk_sentences:  Minimum sentences per chunk before a split is
                              allowed. Prevents over-fragmentation.
    """

    def __init__(
        self,
        window_size: int = 3,
        depth_threshold: float = 0.3,
        min_chunk_sentences: int = 2,
    ) -> None:
        self._window_size = window_size
        self._depth_threshold = depth_threshold
        self._min_chunk_sentences = min_chunk_sentences

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        chunks = self._split_text(text)
        meta = metadata or {}
        return [Document(page_content=c, metadata=dict(meta)) for c in chunks]

    # ------------------------------------------------------------------
    # Internal — TextTiling algorithm
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        sentences = _sentence_tokenize(text)
        if len(sentences) <= 1:
            return sentences

        token_seqs = [_tokenize(s) for s in sentences]
        n = len(sentences)
        w = self._window_size

        gap_scores: list[float] = []
        for i in range(n - 1):
            left_start = max(0, i - w + 1)
            right_end = min(n, i + w + 1)
            block_a = token_seqs[left_start: i + 1]
            block_b = token_seqs[i + 1: right_end]
            gap_scores.append(_block_score(block_a, block_b))

        depth_scores = [_depth_score(gap_scores, i) for i in range(len(gap_scores))]

        boundaries: list[int] = []
        for i, d in enumerate(depth_scores):
            if d >= self._depth_threshold:
                prev = boundaries[-1] if boundaries else -1
                if (i - prev) >= self._min_chunk_sentences:
                    boundaries.append(i)

        chunks: list[str] = []
        prev_idx = 0
        for b in boundaries:
            chunks.append(" ".join(sentences[prev_idx: b + 1]))
            prev_idx = b + 1

        if prev_idx < n:
            chunks.append(" ".join(sentences[prev_idx:]))

        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# EmbeddingSemanticChunker
# ---------------------------------------------------------------------------

class EmbeddingSemanticChunker(BaseChunker):
    """Embedding-based semantic chunker — provider-agnostic via BaseEmbedder.

    Splits text wherever consecutive sentence similarity drops below
    threshold, signalling a topic shift.

    Algorithm:
        1. Split text into sentences.
        2. Embed all sentences in one batch via the injected embedder.
        3. Compute cosine similarity between consecutive sentence embeddings.
        4. Insert a chunk boundary where similarity < threshold
           AND the current chunk has at least min_chunk_size sentences.
        5. Merge sentences within each chunk into one string.

    Args:
        embedder:        Any BaseEmbedder instance (LocalEmbedder,
                         OllamaEmbedder, OpenAIEmbedder). Injected at
                         construction time — fully plug-and-play.
        threshold:       Cosine similarity below which a boundary is inserted.
                         Range [0, 1]. Lower = fewer, larger chunks.
        min_chunk_size:  Minimum sentences per chunk. Prevents single-sentence
                         fragments when threshold is tight.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.5,
        min_chunk_size: int = 2,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._min_chunk_size = min_chunk_size

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        chunks = self._split_text(text)
        meta = metadata or {}
        return [Document(page_content=c, metadata=dict(meta)) for c in chunks]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        sentences = _sentence_tokenize(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        raw_vectors = self._embedder.embed(sentences)
        embeddings = np.array(raw_vectors)

        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        chunks: list[str] = []
        current: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            if sim < self._threshold and len(current) >= self._min_chunk_size:
                chunks.append(" ".join(current))
                current = [next_sentence]
            else:
                current.append(next_sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks
