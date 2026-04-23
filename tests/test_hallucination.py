"""Tests for the hallucination detector (mocked Claude — no API key required)."""

import json
from unittest.mock import MagicMock
import pytest

from rag_eval.llm.claude import ClaudeLLM
from rag_eval.metrics.hallucination import HallucinationDetector, HallucinationResult, ClaimVerification


def _mock_detector(claims_response: list[dict]) -> HallucinationDetector:
    llm = ClaudeLLM(api_key="sk-ant-test")
    llm.complete_json = MagicMock(return_value={"claims": claims_response})
    return HallucinationDetector(llm=llm)


# ── HallucinationResult ───────────────────────────────────────────────────────

def test_grounding_score_all_supported():
    result = HallucinationResult(
        answer="A",
        sources=["S"],
        claims=[
            ClaimVerification("c1", supported=True),
            ClaimVerification("c2", supported=True),
        ],
    )
    assert result.grounding_score == 1.0
    assert result.hallucination_score == 0.0


def test_grounding_score_none_supported():
    result = HallucinationResult(
        answer="A",
        sources=["S"],
        claims=[ClaimVerification("c1", False), ClaimVerification("c2", False)],
    )
    assert result.hallucination_score == 1.0
    assert result.grounding_score == 0.0


def test_grounding_score_partial():
    result = HallucinationResult(
        answer="A",
        sources=["S"],
        claims=[ClaimVerification("c1", True), ClaimVerification("c2", False)],
    )
    assert result.hallucination_score == pytest.approx(0.5)


def test_no_claims_means_no_hallucination():
    result = HallucinationResult(answer="A", sources=["S"], claims=[])
    assert result.hallucination_score == 0.0
    assert result.grounding_score == 1.0


# ── HallucinationDetector.check ───────────────────────────────────────────────

def test_check_fully_grounded():
    detector = _mock_detector([
        {"text": "Paris is in France.", "supported": True, "source_idx": 0},
    ])
    result = detector.check("Paris is in France.", ["Paris is the capital of France."])
    assert result.grounding_score == 1.0
    assert len(result.claims) == 1


def test_check_hallucination_detected():
    detector = _mock_detector([
        {"text": "Paris has 50 million people.", "supported": False, "source_idx": None},
    ])
    result = detector.check("Paris has 50M people.", ["Paris is the capital of France."])
    assert result.hallucination_score == 1.0


def test_check_mixed():
    detector = _mock_detector([
        {"text": "c1", "supported": True, "source_idx": 0},
        {"text": "c2", "supported": False, "source_idx": None},
        {"text": "c3", "supported": True, "source_idx": 0},
    ])
    result = detector.check("answer", ["source"])
    assert result.grounding_score == pytest.approx(2 / 3)


# ── check_batch + aggregate ───────────────────────────────────────────────────

def test_check_batch():
    detector = _mock_detector([{"text": "c1", "supported": True, "source_idx": 0}])
    results = detector.check_batch(["A1", "A2"], [["S1"], ["S2"]])
    assert len(results) == 2
    for r in results:
        assert isinstance(r, HallucinationResult)


def test_aggregate_mean_scores():
    results = [
        HallucinationResult("A1", ["S1"], [ClaimVerification("c", True)]),
        HallucinationResult("A2", ["S2"], [ClaimVerification("c", False)]),
    ]
    detector = HallucinationDetector.__new__(HallucinationDetector)
    agg = detector.aggregate(results)
    assert agg["hallucination_score"] == pytest.approx(0.5)
    assert agg["grounding_score"] == pytest.approx(0.5)
    assert agg["total_samples"] == 2
    assert agg["total_claims"] == 2


def test_aggregate_empty():
    detector = HallucinationDetector.__new__(HallucinationDetector)
    agg = detector.aggregate([])
    assert agg["hallucination_score"] == 0.0
    assert agg["grounding_score"] == 1.0


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_keys():
    result = HallucinationResult(
        answer="A",
        sources=["S"],
        claims=[
            ClaimVerification("c1", True, 0),
            ClaimVerification("c2", False, None),
        ],
    )
    d = result.to_dict()
    assert set(d.keys()) == {
        "hallucination_score",
        "grounding_score",
        "total_claims",
        "supported_claims",
        "unsupported_claims",
    }
    assert d["total_claims"] == 2
    assert "c1" in d["supported_claims"]
    assert "c2" in d["unsupported_claims"]
