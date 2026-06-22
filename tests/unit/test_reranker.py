"""Unit tests for CrossEncoderReranker.

CrossEncoder is patched at import time so no model weights are downloaded.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def reranker():
    """CrossEncoderReranker with a mocked underlying model."""
    with patch("src.reranker.cross_encoder.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        from src.reranker.cross_encoder import CrossEncoderReranker
        r = CrossEncoderReranker(model="BAAI/bge-reranker-base", device="cpu")
        yield r


def _chunks(n: int) -> list[dict]:
    return [{"chunk_content": f"passage {i}", "chunk_id": i} for i in range(n)]


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_results_sorted_descending_by_score(reranker) -> None:
    chunks = _chunks(3)
    reranker._model.predict.return_value = np.array([0.1, 0.9, 0.5])
    results = reranker.rerank("query", chunks, top_k=3)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_highest_scoring_chunk_is_first(reranker) -> None:
    chunks = _chunks(3)
    reranker._model.predict.return_value = np.array([0.2, 0.8, 0.5])
    results = reranker.rerank("query", chunks, top_k=3)
    assert results[0]["chunk_id"] == 1  # index 1 has score 0.8


def test_rerank_score_added_to_each_chunk(reranker) -> None:
    chunks = _chunks(2)
    reranker._model.predict.return_value = np.array([0.6, 0.4])
    results = reranker.rerank("query", chunks, top_k=2)
    assert all("rerank_score" in r for r in results)


# ---------------------------------------------------------------------------
# top_k behaviour
# ---------------------------------------------------------------------------

def test_top_k_limits_result_count(reranker) -> None:
    chunks = _chunks(5)
    reranker._model.predict.return_value = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    results = reranker.rerank("query", chunks, top_k=2)
    assert len(results) == 2


def test_fewer_chunks_than_top_k_returns_all(reranker) -> None:
    chunks = _chunks(2)
    reranker._model.predict.return_value = np.array([0.7, 0.3])
    results = reranker.rerank("query", chunks, top_k=10)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_empty_list(reranker) -> None:
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
    reranker._model.predict.assert_not_called()


def test_original_chunk_data_preserved(reranker) -> None:
    chunks = [{"chunk_content": "hello", "chunk_id": 99, "extra": "data"}]
    reranker._model.predict.return_value = np.array([0.5])
    results = reranker.rerank("query", chunks, top_k=1)
    assert results[0]["extra"] == "data"
    assert results[0]["chunk_id"] == 99
