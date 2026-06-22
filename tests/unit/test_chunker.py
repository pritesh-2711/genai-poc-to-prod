"""Unit tests for document chunkers."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.chunker.hierarchical import HierarchicalChunker
from src.chunker.semantic import EmbeddingSemanticChunker, TextTilingChunker


# ---------------------------------------------------------------------------
# HierarchicalChunker
# ---------------------------------------------------------------------------

@pytest.fixture
def hier() -> HierarchicalChunker:
    # Small sizes so tests don't need a large corpus
    return HierarchicalChunker(
        parent_chunk_size=200,
        parent_chunk_overlap=20,
        child_chunk_size=50,
        child_chunk_overlap=5,
    )


def test_hierarchical_splits_long_text(hier: HierarchicalChunker) -> None:
    text = "word " * 300  # 1500 chars
    children = hier.chunk(text)
    assert len(children) > 1


def test_each_child_has_parent_id(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    children = hier.chunk(text)
    for child in children:
        assert "parent_id" in child.metadata


def test_each_child_has_parent_text(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    children = hier.chunk(text)
    for child in children:
        assert "parent_text" in child.metadata
        assert child.metadata["parent_text"]


def test_child_content_fits_inside_parent(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    children = hier.chunk(text)
    for child in children:
        parent_text = child.metadata["parent_text"]
        assert child.page_content in parent_text


def test_metadata_is_forwarded_to_all_children(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    children = hier.chunk(text, metadata={"source": "paper.pdf", "page": 1})
    for child in children:
        assert child.metadata["source"] == "paper.pdf"
        assert child.metadata["page"] == 1


def test_empty_input_produces_no_chunks(hier: HierarchicalChunker) -> None:
    assert hier.chunk("") == []


def test_single_sentence_produces_one_child() -> None:
    chunker = HierarchicalChunker(parent_chunk_size=1000, child_chunk_size=500)
    children = chunker.chunk("Just one short sentence.")
    assert len(children) == 1


def test_chunk_with_parents_returns_both_lists(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    parents, children = hier.chunk_with_parents(text)
    assert len(parents) >= 1
    assert len(children) >= len(parents)


def test_children_reference_correct_parent_index(hier: HierarchicalChunker) -> None:
    text = "word " * 200
    parents, children = hier.chunk_with_parents(text)
    for child in children:
        pid = child.metadata["parent_id"]
        assert 0 <= pid < len(parents)
        assert child.metadata["parent_text"] == parents[pid].page_content


# ---------------------------------------------------------------------------
# TextTilingChunker
# ---------------------------------------------------------------------------

def test_text_tiling_single_sentence() -> None:
    chunker = TextTilingChunker()
    docs = chunker.chunk("Only one sentence here.")
    assert len(docs) == 1
    assert docs[0].page_content


def test_text_tiling_empty_produces_no_chunks() -> None:
    chunker = TextTilingChunker()
    assert chunker.chunk("") == []


def test_text_tiling_preserves_metadata() -> None:
    chunker = TextTilingChunker()
    docs = chunker.chunk("Sentence one. Sentence two.", metadata={"src": "test"})
    for doc in docs:
        assert doc.metadata.get("src") == "test"


def test_text_tiling_multi_paragraph_text() -> None:
    chunker = TextTilingChunker(window_size=2, depth_threshold=0.1)
    text = (
        "Cats eat fish. Cats drink milk. Cats sleep all day. "
        "Cars run on gasoline. Cars need maintenance. Cars have engines."
    )
    docs = chunker.chunk(text)
    # May or may not split — key assertion is it returns documents
    assert len(docs) >= 1
    assert all(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# EmbeddingSemanticChunker (mock embedder)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()

    def _fake_embed(sentences):
        # Return deterministic, distinct vectors per sentence
        rng = np.random.default_rng(42)
        return [rng.random(8).tolist() for _ in sentences]

    embedder.embed.side_effect = _fake_embed
    return embedder


def test_embedding_chunker_produces_chunks(mock_embedder: MagicMock) -> None:
    chunker = EmbeddingSemanticChunker(embedder=mock_embedder)
    text = "Sentence one. Sentence two. Sentence three."
    docs = chunker.chunk(text)
    assert len(docs) >= 1


def test_embedding_chunker_empty_returns_empty(mock_embedder: MagicMock) -> None:
    chunker = EmbeddingSemanticChunker(embedder=mock_embedder)
    assert chunker.chunk("") == []


def test_embedding_chunker_calls_embedder(mock_embedder: MagicMock) -> None:
    chunker = EmbeddingSemanticChunker(embedder=mock_embedder)
    chunker.chunk("One sentence. Another sentence.")
    mock_embedder.embed.assert_called_once()


def test_embedding_chunker_merges_similar_sentences(mock_embedder: MagicMock) -> None:
    # All identical vectors → similarity = 1.0 → no splits → one chunk
    mock_embedder.embed.side_effect = lambda sents: [[1.0, 0.0]] * len(sents)
    chunker = EmbeddingSemanticChunker(embedder=mock_embedder, threshold=0.5)
    docs = chunker.chunk("First sentence. Second sentence. Third sentence.")
    assert len(docs) == 1
