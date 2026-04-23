"""FastAPI endpoint tests (no API key required — hallucination endpoint is skipped without key)."""

import pytest
from fastapi.testclient import TestClient
from rag_eval.api.app import create_app

client = TestClient(create_app())

DOCS = [
    {"id": "doc_1", "text": "Paris is the capital of France."},
    {"id": "doc_2", "text": "Berlin is the capital of Germany."},
]
SAMPLES = [
    {"question": "Capital of France?", "relevant_doc_ids": ["doc_1"]},
    {"question": "Capital of Germany?", "relevant_doc_ids": ["doc_2"]},
]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "claude_available" in resp.json()


def test_index_and_search():
    resp = client.post("/index", json={"backend": "faiss", "documents": DOCS})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert resp.json()["indexed"] == 2

    resp2 = client.post(f"/search/{session_id}", json={"query": "French capital", "k": 2})
    assert resp2.status_code == 200
    results = resp2.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] == "doc_1"


def test_search_unknown_session():
    resp = client.post("/search/nonexistent", json={"query": "test", "k": 1})
    assert resp.status_code == 404


def test_retrieval_metrics_endpoint():
    payload = {
        "retrieved_lists": [["a", "b", "c"], ["x", "a"]],
        "relevant_lists": [["a", "b"], ["a"]],
        "k_values": [1, 3],
    }
    resp = client.post("/metrics/retrieval", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "recall@1" in data
    assert "mrr" in data


def test_hallucination_endpoint_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Recreate app after env change so ClaudeLLM.is_available() re-evaluates
    c = TestClient(create_app())
    resp = c.post(
        "/metrics/hallucination",
        json={"answers": ["Paris is in France."], "sources_list": [["Paris is the capital of France."]]},
    )
    assert resp.status_code == 503


def test_evaluate_endpoint():
    resp = client.post(
        "/evaluate",
        json={"backend": "faiss", "documents": DOCS, "samples": SAMPLES, "k_values": [1, 2]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "retrieval" in data
    assert "recall@1" in data["retrieval"]


def test_evaluate_returns_zero_generation_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = TestClient(create_app())
    resp = c.post(
        "/evaluate",
        json={"backend": "faiss", "documents": DOCS, "samples": SAMPLES},
    )
    assert resp.status_code == 200
    assert resp.json()["generation"] == {}
