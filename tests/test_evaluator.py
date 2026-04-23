"""Integration tests for the RAGEvaluator pipeline (no API key required)."""

import pytest
from rag_eval import RAGEvaluator, EvalDataset
from rag_eval.backends.faiss_backend import FAISSBackend
from rag_eval.backends.base import Document
from rag_eval.pipeline.evaluator import EvalConfig


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
def dataset():
    return EvalDataset.from_dicts(
        [
            {"question": "What is the capital of France?", "relevant_doc_ids": ["doc_1", "doc_4"]},
            {"question": "Capital of Germany?", "relevant_doc_ids": ["doc_2"]},
        ],
        name="test_dataset",
    )


def test_retrieval_metrics_present(backend, dataset):
    ev = RAGEvaluator(backend=backend, config=EvalConfig(k_values=[1, 3]))
    report = ev.evaluate(dataset, top_k=3)
    for key in ("recall@1", "recall@3", "mrr", "ndcg@1", "ndcg@3"):
        assert key in report.retrieval


def test_per_sample_populated(backend, dataset):
    ev = RAGEvaluator(backend=backend)
    report = ev.evaluate(dataset, top_k=3)
    assert len(report.per_sample) == 2
    for s in report.per_sample:
        assert "question" in s and "retrieved_ids" in s


def test_no_generation_metrics_without_fn(backend, dataset):
    ev = RAGEvaluator(backend=backend)
    report = ev.evaluate(dataset)
    assert report.generation == {}
    assert report.hallucination == {}


def test_answer_recorded_when_rag_fn_provided(backend, dataset):
    ev = RAGEvaluator(backend=backend, config=EvalConfig(run_generation_metrics=False))
    report = ev.evaluate(dataset, rag_fn=lambda q, ctx: f"Answer: {q}", top_k=3)
    assert report.per_sample[0].get("answer") is not None


def test_report_summary_keys(backend, dataset):
    report = RAGEvaluator(backend=backend).evaluate(dataset)
    summary = report.summary()
    for key in ("dataset", "retrieval", "generation", "hallucination"):
        assert key in summary
