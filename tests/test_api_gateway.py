"""Tests for the API Gateway (all new endpoints + verifying existing routes still work)."""

import json
import pytest
from fastapi.testclient import TestClient

from services.api_gateway.app import create_gateway_app


@pytest.fixture
def client(tmp_path):
    app = create_gateway_app(
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
    )
    return TestClient(app)


DOCS = [
    {"id": "doc_1", "text": "Paris is the capital of France."},
    {"id": "doc_2", "text": "Berlin is the capital of Germany."},
    {"id": "doc_3", "text": "Tokyo is the capital of Japan."},
]
SAMPLES = [
    {"question": "Capital of France?",  "relevant_doc_ids": ["doc_1"]},
    {"question": "Capital of Germany?", "relevant_doc_ids": ["doc_2"]},
]


# ── Existing routes still accessible ─────────────────────────────────────────

def test_health_still_works(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_index_still_works(client):
    resp = client.post("/index", json={"backend": "faiss", "documents": DOCS})
    assert resp.status_code == 200
    assert resp.json()["indexed"] == 3


def test_retrieval_metrics_still_works(client):
    resp = client.post("/metrics/retrieval", json={
        "retrieved_lists": [["doc_1", "doc_2"]],
        "relevant_lists":  [["doc_1"]],
        "k_values":        [1, 2],
    })
    assert resp.status_code == 200
    assert "recall@1" in resp.json()


# ── Dataset endpoints ─────────────────────────────────────────────────────────

def test_load_dataset(client):
    resp = client.post("/datasets/load", json={
        "name": "test_dataset",
        "samples": SAMPLES,
        "documents": DOCS,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "dataset_id" in data
    assert data["sample_count"] == 2


def test_list_datasets(client):
    client.post("/datasets/load", json={"name": "ds1", "samples": SAMPLES})
    resp = client.get("/datasets")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# ── Full eval endpoint ────────────────────────────────────────────────────────

def test_full_eval_inline_retrieval_only(client):
    resp = client.post("/run/full-eval", json={
        "samples":   SAMPLES,
        "documents": DOCS,
        "backend":   "faiss",
        "top_k":     2,
        "k_values":  [1, 2],
        "run_generation": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] in ("completed", "completed_partial")


def test_full_eval_stores_report(client):
    resp = client.post("/run/full-eval", json={
        "samples":   SAMPLES,
        "documents": DOCS,
        "backend":   "faiss",
        "k_values":  [1, 2],
    })
    run_id = resp.json()["run_id"]

    report_resp = client.get(f"/reports/{run_id}")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["run_id"] == run_id
    assert "aggregate_retrieval" in report
    assert len(report["per_sample"]) == 2


def test_full_eval_via_dataset_id(client):
    # First load dataset
    load_resp = client.post("/datasets/load", json={
        "name": "preloaded", "samples": SAMPLES, "documents": DOCS,
    })
    dataset_id = load_resp.json()["dataset_id"]

    # Then run eval referencing it
    resp = client.post("/run/full-eval", json={
        "dataset_id": dataset_id,
        "backend": "faiss",
    })
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == dataset_id


def test_full_eval_unknown_dataset_id_returns_404(client):
    resp = client.post("/run/full-eval", json={"dataset_id": "sha256:nonexistent"})
    assert resp.status_code == 404


def test_full_eval_no_data_returns_400(client):
    resp = client.post("/run/full-eval", json={"backend": "faiss"})
    assert resp.status_code == 400


# ── Report / failure / cost endpoints ────────────────────────────────────────

def test_get_report_not_found(client):
    resp = client.get("/reports/nonexistent-run-id")
    assert resp.status_code == 404


def test_get_failures_empty_run(client):
    resp = client.post("/run/full-eval", json={
        "samples": SAMPLES, "documents": DOCS, "backend": "faiss",
    })
    run_id = resp.json()["run_id"]

    fail_resp = client.get(f"/failures/{run_id}")
    assert fail_resp.status_code == 200
    assert "failures" in fail_resp.json()
    assert fail_resp.json()["run_id"] == run_id


def test_get_cost_no_generation(client):
    resp = client.post("/run/full-eval", json={
        "samples": SAMPLES, "documents": DOCS, "backend": "faiss",
    })
    run_id = resp.json()["run_id"]

    cost_resp = client.get(f"/cost/{run_id}")
    assert cost_resp.status_code == 200
    data = cost_resp.json()
    assert data["run_id"] == run_id
    assert data["cost_report"] is None  # no generation ran


def test_get_cost_not_found(client):
    resp = client.get("/cost/ghost-run")
    assert resp.status_code == 404


def test_full_eval_generation_skipped_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_gateway_app(
        log_dir=str(tmp_path / "logs2"),
        data_dir=str(tmp_path / "data2"),
    )
    c = TestClient(app)
    resp = c.post("/run/full-eval", json={
        "samples":        SAMPLES,
        "documents":      DOCS,
        "backend":        "faiss",
        "run_generation": True,
        "llm_provider":   "anthropic",
    })
    assert resp.status_code == 200
    data = resp.json()
    run_id = data["run_id"]

    report = c.get(f"/reports/{run_id}").json()
    assert "skipped" in report.get("error", "").lower() or report["aggregate_generation"] == {}
