"""Unit tests for retrieval metrics (no external deps, no API key required)."""

import pytest
from rag_eval.metrics.retrieval import recall_at_k, mrr, ndcg_at_k, RetrievalMetrics


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0


def test_recall_at_k_zero():
    assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_position():
    assert mrr(["x", "a", "b"], ["a"]) == pytest.approx(0.5)


def test_mrr_not_found():
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_at_k_perfect():
    assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)


def test_ndcg_at_k_reversed():
    score = ndcg_at_k(["x", "a"], ["a"], k=2)
    assert score < 1.0


def test_ndcg_at_k_no_relevant_in_retrieved():
    assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_retrieval_metrics_aggregate():
    rm = RetrievalMetrics(k_values=[1, 3])
    retrieved = [["a", "b", "c"], ["x", "a", "b"]]
    relevant = [["a", "b"], ["a"]]
    scores = rm.compute(retrieved, relevant)
    assert "recall@1" in scores
    assert "recall@3" in scores
    assert "mrr" in scores
    assert "ndcg@1" in scores
    assert scores["recall@3"] == pytest.approx(1.0)
    assert scores["mrr"] == pytest.approx((1.0 + 0.5) / 2)
