"""Tests for RAGAS-style generation metrics (mocked Claude — no API key required)."""

import json
from unittest.mock import MagicMock, patch
import pytest

from rag_eval.llm.claude import ClaudeLLM
from rag_eval.metrics.generation import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    GenerationMetrics,
)


def _mock_llm(response_dict: dict) -> ClaudeLLM:
    llm = ClaudeLLM(api_key="sk-ant-test")
    llm.complete_json = MagicMock(return_value=response_dict)
    return llm


# ── faithfulness ──────────────────────────────────────────────────────────────

def test_faithfulness_fully_supported():
    llm = _mock_llm({
        "claims": [{"text": "Paris is in France.", "supported": True}],
        "faithfulness_score": 1.0,
    })
    assert faithfulness("Where is Paris?", "Paris is in France.", ["Paris is the capital of France."], llm) == 1.0


def test_faithfulness_partial():
    llm = _mock_llm({
        "claims": [
            {"text": "Paris is in France.", "supported": True},
            {"text": "Paris has 20 million people.", "supported": False},
        ],
        "faithfulness_score": 0.5,
    })
    score = faithfulness("Tell me about Paris.", "Paris is in France with 20M people.", ["Paris is in France."], llm)
    assert score == pytest.approx(0.5)


def test_faithfulness_no_claims():
    llm = _mock_llm({"claims": []})
    assert faithfulness("Q", "A", ["ctx"], llm) == 1.0


def test_faithfulness_derives_score_from_claims_when_missing():
    llm = _mock_llm({
        "claims": [
            {"text": "c1", "supported": True},
            {"text": "c2", "supported": True},
            {"text": "c3", "supported": False},
        ]
    })
    score = faithfulness("Q", "A", ["ctx"], llm)
    assert score == pytest.approx(2 / 3)


# ── answer_relevancy ──────────────────────────────────────────────────────────

def test_answer_relevancy_high():
    llm = _mock_llm({"score": 0.95, "reasoning": "Very relevant."})
    assert answer_relevancy("What is Paris?", "Paris is the capital of France.", llm) == pytest.approx(0.95)


def test_answer_relevancy_missing_score():
    llm = _mock_llm({})
    assert answer_relevancy("Q", "A", llm) == 0.0


# ── context_precision ─────────────────────────────────────────────────────────

def test_context_precision_computed():
    llm = _mock_llm({"scores": [0.9, 0.4], "context_precision": 0.65})
    assert context_precision("Q", ["ctx1", "ctx2"], llm) == pytest.approx(0.65)


def test_context_precision_derives_from_scores():
    llm = _mock_llm({"scores": [1.0, 0.0]})
    assert context_precision("Q", ["ctx1", "ctx2"], llm) == pytest.approx(0.5)


def test_context_precision_empty_contexts():
    llm = _mock_llm({})
    assert context_precision("Q", [], llm) == 0.0


# ── context_recall ────────────────────────────────────────────────────────────

def test_context_recall_score():
    llm = _mock_llm({"score": 0.8, "reasoning": "Most info present."})
    assert context_recall("Ground truth answer.", ["ctx1"], llm) == pytest.approx(0.8)


def test_context_recall_empty_ground_truth():
    llm = _mock_llm({})
    assert context_recall("", ["ctx"], llm) == 0.0


# ── GenerationMetrics aggregator ──────────────────────────────────────────────

def test_generation_metrics_compute():
    mock_llm = ClaudeLLM(api_key="sk-ant-test")
    mock_llm.complete_json = MagicMock(return_value={
        "faithfulness_score": 1.0,
        "score": 0.9,
        "context_precision": 0.8,
        "claims": [],
    })

    gm = GenerationMetrics(
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        llm=mock_llm,
    )
    scores = gm.compute(
        questions=["Q1", "Q2"],
        answers=["A1", "A2"],
        contexts=[["ctx1"], ["ctx2"]],
    )
    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "context_precision" in scores
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_generation_metrics_context_recall_skipped_without_ground_truth():
    mock_llm = ClaudeLLM(api_key="sk-ant-test")
    mock_llm.complete_json = MagicMock(return_value={"faithfulness_score": 1.0, "score": 0.9})

    gm = GenerationMetrics(metrics=["faithfulness", "context_recall"], llm=mock_llm)
    scores = gm.compute(["Q"], ["A"], [["ctx"]])  # no ground_truths
    assert "faithfulness" in scores
    assert "context_recall" not in scores
