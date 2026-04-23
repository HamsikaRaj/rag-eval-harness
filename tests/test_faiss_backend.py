"""Integration tests for the FAISS backend (no network, no API key required)."""

import pytest
from rag_eval.backends.faiss_backend import FAISSBackend
from rag_eval.backends.base import Document


@pytest.fixture
def backend():
    b = FAISSBackend()
    docs = [
        Document(id="doc_1", text="Paris is the capital of France and home to the Eiffel Tower."),
        Document(id="doc_2", text="Berlin is the capital of Germany."),
        Document(id="doc_3", text="Tokyo is the capital of Japan."),
    ]
    b.add_documents(docs)
    return b


def test_search_returns_results(backend):
    results = backend.search("French capital", k=2)
    assert len(results) == 2


def test_search_top_result_relevant(backend):
    results = backend.search("What is the capital of France?", k=3)
    assert results[0].document.id == "doc_1"


def test_search_ids(backend):
    ids = backend.search_ids("German capital", k=2)
    assert "doc_2" in ids


def test_delete_and_search(backend):
    backend.delete(["doc_1"])
    ids = backend.search_ids("Eiffel Tower Paris", k=3)
    assert "doc_1" not in ids


def test_clear(backend):
    backend.clear()
    assert backend.search("capital", k=5) == []


def test_ranks_are_sequential(backend):
    results = backend.search("capital city", k=3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
