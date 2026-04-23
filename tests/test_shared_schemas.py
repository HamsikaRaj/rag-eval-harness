"""Tests for shared Pydantic schemas."""

from shared.schemas.models import EvalQuery, EvalResult, FailureLog, CostReport, RunReport


def test_eval_query_defaults():
    q = EvalQuery(question="What is RBAC?")
    assert q.question == "What is RBAC?"
    assert q.relevant_doc_ids == []
    assert q.ground_truth_answer is None


def test_eval_query_full():
    q = EvalQuery(
        question="Q",
        relevant_doc_ids=["d1", "d2"],
        ground_truth_answer="A",
        metadata={"source": "test"},
    )
    assert len(q.relevant_doc_ids) == 2
    assert q.metadata["source"] == "test"


def test_eval_result_defaults():
    r = EvalResult(run_id="run1", question="Q?")
    assert r.retrieved_ids == []
    assert r.estimated_cost_usd == 0.0
    assert r.is_slow_query is False


def test_failure_log_timestamp_auto():
    f = FailureLog(run_id="r1", service="retrieval", query="Q", issue="zero_docs")
    assert f.timestamp  # auto-filled
    assert f.retrieved == []


def test_failure_log_serialization():
    f = FailureLog(
        run_id="r1", service="svc", query="Q", issue="err",
        retrieved=["d1"], expected=["d2"], latency_ms=123.4,
    )
    d = f.model_dump()
    assert d["latency_ms"] == 123.4
    assert d["retrieved"] == ["d1"]


def test_cost_report():
    c = CostReport(
        run_id="r1", model="claude-sonnet-4-20250514",
        total_input_tokens=1000, total_output_tokens=200,
        estimated_cost_usd=0.006,
        breakdown={"generation_evaluator": 0.006},
    )
    assert c.breakdown["generation_evaluator"] == 0.006


def test_run_report_defaults():
    r = RunReport(run_id="r1")
    assert r.status == "pending"
    assert r.per_sample == []
    assert r.failure_count == 0
    assert r.started_at  # auto-filled


def test_run_report_with_cost():
    cost = CostReport(run_id="r1", model="gpt-4o-mini",
                      total_input_tokens=500, total_output_tokens=100,
                      estimated_cost_usd=0.0001)
    r = RunReport(run_id="r1", cost_report=cost)
    assert r.cost_report.model == "gpt-4o-mini"
