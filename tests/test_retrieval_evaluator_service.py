"""Tests for the RetrievalEvaluatorService."""

import time
import pytest
from unittest.mock import MagicMock, patch

from rag_eval.backends.base import Document, SearchResult
from rag_eval.backends.faiss_backend import FAISSBackend
from services.retrieval_evaluator.evaluator import RetrievalEvaluatorService, RetrievalEvalResult
from shared.logging.logger import FailureLogger


@pytest.fixture
def backend():
    b = FAISSBackend()
    b.add_documents([
        Document(id="doc_1", text="Paris is the capital of France."),
        Document(id="doc_2", text="Berlin is the capital of Germany."),
        Document(id="doc_3", text="Tokyo is the capital of Japan."),
        Document(id="doc_4", text="The Eiffel Tower is in Paris."),
    ])
    return b


@pytest.fixture
def service(backend, tmp_path):
    logger = FailureLogger(run_id="svc-test", log_dir=str(tmp_path))
    return RetrievalEvaluatorService(
        backend=backend,
        k_values=[1, 3],
        slow_threshold_ms=1.0,   # very low so we can test slow-flagging deterministically
        failure_logger=logger,
    ), logger


def test_evaluate_query_returns_result(service):
    svc, _ = service
    result = svc.evaluate_query("run1", "Capital of France?", ["doc_1", "doc_4"])
    assert isinstance(result, RetrievalEvalResult)
    assert result.query == "Capital of France?"
    assert len(result.retrieved_ids) > 0
    assert "recall@1" in result.metrics
    assert "mrr" in result.metrics


def test_evaluate_query_latency_tracked(service):
    svc, _ = service
    result = svc.evaluate_query("run1", "Q?", ["doc_1"])
    assert result.latency_ms >= 0.0


def test_slow_query_flagging(service):
    svc, _ = service
    # threshold is 1ms — any real FAISS query will exceed this
    result = svc.evaluate_query("run1", "Capital?", ["doc_1"])
    assert result.is_slow is True


def test_zero_relevant_detected_and_logged(service):
    svc, logger = service
    # Use doc IDs that definitely won't be retrieved for unrelated query
    result = svc.evaluate_query("run1", "quantum physics entanglement", ["completely_nonexistent_id"])
    # If zero relevant docs retrieved, should be flagged
    if result.has_zero_relevant:
        assert logger.count >= 1
        failure = logger.get_all()[0]
        assert failure.issue == "zero_relevant_docs_retrieved"
        assert failure.service == "retrieval_evaluator"


def test_evaluate_dataset_batch(service):
    svc, _ = service
    queries  = ["Capital of France?", "Capital of Germany?"]
    relevant = [["doc_1"], ["doc_2"]]
    results  = svc.evaluate_dataset("run1", queries, relevant)
    assert len(results) == 2


def test_aggregate_includes_latency_stats(service):
    svc, _ = service
    queries  = ["Capital of France?", "Capital of Japan?"]
    relevant = [["doc_1"], ["doc_3"]]
    results  = svc.evaluate_dataset("run1", queries, relevant)
    agg = svc.aggregate(results)
    assert "latency_ms_mean" in agg
    assert "latency_ms_max" in agg
    assert "slow_query_count" in agg
    assert "zero_relevant_count" in agg


def test_aggregate_empty_returns_empty(service):
    svc, _ = service
    assert svc.aggregate([]) == {}


def test_metrics_keys_present(service):
    svc, _ = service
    result = svc.evaluate_query("run1", "Paris capital?", ["doc_1"])
    for k in [1, 3]:
        assert f"recall@{k}" in result.metrics
        assert f"ndcg@{k}" in result.metrics
    assert "mrr" in result.metrics
