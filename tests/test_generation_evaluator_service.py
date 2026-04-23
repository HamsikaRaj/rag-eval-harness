"""Tests for the GenerationEvaluatorService (mocked ModelRunner)."""

import json
from unittest.mock import MagicMock
import pytest

from services.model_runner.runner import ModelRunner, ModelResponse
from services.generation_evaluator.evaluator import GenerationEvaluatorService, GenerationEvalResult
from shared.logging.logger import FailureLogger


def _mock_runner(response_dict: dict, in_tok: int = 100, out_tok: int = 50) -> ModelRunner:
    runner = MagicMock(spec=ModelRunner)
    runner.model = "claude-test"
    runner.complete_json = MagicMock(return_value=response_dict)
    runner.get_usage = MagicMock(return_value={
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "call_count":         2,
        "estimated_cost_usd": 0.0,
    })
    runner.reset_usage = MagicMock()
    return runner


@pytest.fixture
def high_faith_service(tmp_path):
    runner = _mock_runner({
        "faithfulness_score": 0.9,
        "score": 0.85,
        "context_precision": 0.8,
        "claims": [{"text": "c1", "supported": True}],
    })
    logger = FailureLogger(run_id="gen-test", log_dir=str(tmp_path))
    svc = GenerationEvaluatorService(
        runner=runner,
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        faithfulness_threshold=0.5,
        failure_logger=logger,
    )
    return svc, logger


@pytest.fixture
def low_faith_service(tmp_path):
    runner = _mock_runner({
        "faithfulness_score": 0.3,   # below threshold
        "score": 0.4,
        "claims": [{"text": "c1", "supported": False}],
    })
    logger = FailureLogger(run_id="gen-fail-test", log_dir=str(tmp_path))
    svc = GenerationEvaluatorService(
        runner=runner,
        metrics=["faithfulness", "answer_relevancy"],
        faithfulness_threshold=0.5,
        failure_logger=logger,
    )
    return svc, logger


def test_evaluate_sample_returns_result(high_faith_service):
    svc, _ = high_faith_service
    result = svc.evaluate_sample(
        run_id="r1",
        query="What is RBAC?",
        answer="RBAC is role-based access control.",
        contexts=["RBAC stands for role-based access control."],
    )
    assert isinstance(result, GenerationEvalResult)
    assert "faithfulness" in result.metrics
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_high_faithfulness_not_flagged(high_faith_service):
    svc, logger = high_faith_service
    result = svc.evaluate_sample("r1", "Q?", "A", ["ctx"])
    assert result.is_faithfulness_failure is False
    assert logger.count == 0


def test_low_faithfulness_flagged_and_logged(low_faith_service):
    svc, logger = low_faith_service
    result = svc.evaluate_sample("r1", "Q?", "Hallucinated answer.", ["ctx"])
    assert result.is_faithfulness_failure is True
    assert logger.count == 1
    failure = logger.get_all()[0]
    assert "faithfulness_below_threshold" in failure.issue


def test_cost_estimated(high_faith_service):
    svc, _ = high_faith_service
    result = svc.evaluate_sample("r1", "Q?", "A", ["ctx"])
    assert isinstance(result.estimated_cost_usd, float)
    assert result.estimated_cost_usd >= 0.0


def test_evaluate_batch(high_faith_service):
    svc, _ = high_faith_service
    results = svc.evaluate_batch(
        run_id="r1",
        queries=["Q1", "Q2"],
        answers=["A1", "A2"],
        contexts=[["ctx1"], ["ctx2"]],
    )
    assert len(results) == 2


def test_aggregate_mean_metrics(high_faith_service):
    svc, _ = high_faith_service
    results = svc.evaluate_batch("r1", ["Q1", "Q2"], ["A1", "A2"], [["c1"], ["c2"]])
    agg = svc.aggregate(results)
    assert "faithfulness" in agg
    assert "total_input_tokens" in agg
    assert "total_output_tokens" in agg
    assert "total_cost_usd" in agg
    assert "faithfulness_failures" in agg


def test_aggregate_empty_returns_empty(high_faith_service):
    svc, _ = high_faith_service
    assert svc.aggregate([]) == {}


def test_get_cost_report(high_faith_service):
    svc, _ = high_faith_service
    results = svc.evaluate_batch("r1", ["Q"], ["A"], [["ctx"]])
    report = svc.get_cost_report("r1", results)
    assert report["run_id"] == "r1"
    assert "total_input_tokens" in report
    assert "breakdown" in report
    assert "generation_evaluator" in report["breakdown"]


def test_reset_usage_called_per_sample(high_faith_service):
    svc, _ = high_faith_service
    svc.evaluate_batch("r1", ["Q1", "Q2"], ["A1", "A2"], [["c1"], ["c2"]])
    # reset_usage should be called once per sample
    assert svc.runner.reset_usage.call_count == 2
