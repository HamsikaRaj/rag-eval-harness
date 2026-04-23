"""Tests for the structured failure logger."""

import json
import pytest
from pathlib import Path
from shared.logging.logger import FailureLogger
from shared.schemas.models import FailureLog


@pytest.fixture
def logger(tmp_path):
    return FailureLogger(run_id="test-run-001", log_dir=str(tmp_path))


def test_log_creates_record(logger):
    record = logger.log(
        service="retrieval_evaluator",
        query="What is RBAC?",
        issue="zero_relevant_docs_retrieved",
        retrieved=["doc_5"],
        expected=["doc_3"],
        latency_ms=340.5,
    )
    assert isinstance(record, FailureLog)
    assert record.run_id == "test-run-001"
    assert record.issue == "zero_relevant_docs_retrieved"
    assert record.latency_ms == 340.5


def test_count_increments(logger):
    assert logger.count == 0
    logger.log(service="svc", query="Q1", issue="err1")
    logger.log(service="svc", query="Q2", issue="err2")
    assert logger.count == 2


def test_get_all_returns_all(logger):
    logger.log(service="s1", query="Q1", issue="i1")
    logger.log(service="s2", query="Q2", issue="i2")
    all_logs = logger.get_all()
    assert len(all_logs) == 2


def test_get_by_service_filters(logger):
    logger.log(service="retrieval", query="Q1", issue="zero_docs")
    logger.log(service="generation", query="Q2", issue="low_faithfulness")
    logger.log(service="retrieval", query="Q3", issue="slow_query")

    ret_logs = logger.get_by_service("retrieval")
    gen_logs = logger.get_by_service("generation")

    assert len(ret_logs) == 2
    assert len(gen_logs) == 1


def test_persists_to_disk(tmp_path):
    logger = FailureLogger(run_id="persist-test", log_dir=str(tmp_path))
    logger.log(service="svc", query="Q", issue="err")

    expected_path = tmp_path / "failures_persist-test.json"
    assert expected_path.exists()
    data = json.loads(expected_path.read_text())
    assert len(data) == 1
    assert data[0]["issue"] == "err"


def test_load_existing_log(tmp_path):
    # Write first, then load
    logger1 = FailureLogger(run_id="reload-test", log_dir=str(tmp_path))
    logger1.log(service="svc", query="Q", issue="err")

    logger2 = FailureLogger.load(run_id="reload-test", log_dir=str(tmp_path))
    assert logger2.count == 1


def test_log_with_empty_lists(logger):
    record = logger.log(service="svc", query="Q", issue="err")
    assert record.retrieved == []
    assert record.expected == []


def test_failure_log_has_timestamp(logger):
    record = logger.log(service="svc", query="Q", issue="err")
    assert "T" in record.timestamp  # ISO 8601 contains 'T'
