"""Main RAG evaluation pipeline — orchestrates retrieval metrics, generation metrics, and hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rag_eval.backends.base import VectorStoreBackend
from rag_eval.metrics.retrieval import RetrievalMetrics
from rag_eval.pipeline.dataset import EvalDataset


@dataclass
class EvalConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    run_generation_metrics: bool = False
    run_hallucination_detection: bool = False
    generation_metrics: list[str] | None = None


@dataclass
class EvalReport:
    dataset_name: str
    retrieval: dict[str, float] = field(default_factory=dict)
    generation: dict[str, float] = field(default_factory=dict)
    hallucination: dict[str, Any] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "hallucination": self.hallucination,
        }

    def print(self) -> None:
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            table = Table(title=f"RAG Eval Report — {self.dataset_name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right", style="green")

            for k, v in self.retrieval.items():
                table.add_row(k, f"{v:.4f}")
            for k, v in self.generation.items():
                table.add_row(k, f"{v:.4f}")
            for k, v in self.hallucination.items():
                if isinstance(v, float):
                    table.add_row(k, f"{v:.4f}")

            console.print(table)
        except ImportError:
            import json
            print(json.dumps(self.summary(), indent=2))


class RAGEvaluator:
    """
    End-to-end RAG evaluation pipeline.

    Usage::

        evaluator = RAGEvaluator(backend=FAISSBackend())
        evaluator.index_documents(documents)
        report = evaluator.evaluate(dataset, rag_fn=my_rag_function)
        report.print()

    ``rag_fn`` signature: ``(question: str, contexts: list[str]) -> str``
    It should call your RAG system and return a generated answer.
    Pass ``rag_fn=None`` to skip generation and hallucination metrics.
    """

    def __init__(
        self,
        backend: VectorStoreBackend,
        config: EvalConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or EvalConfig()
        self._retrieval_metrics = RetrievalMetrics(k_values=self.config.k_values)

    def index_documents(self, documents) -> None:
        """Add documents to the vector store backend."""
        self.backend.add_documents(documents)

    def evaluate(
        self,
        dataset: EvalDataset,
        rag_fn: Callable[[str, list[str]], str] | None = None,
        top_k: int = 10,
    ) -> EvalReport:
        """
        Run the full evaluation pipeline.

        Args:
            dataset:  Evaluation dataset with questions and ground-truth relevant doc IDs.
            rag_fn:   Optional RAG function ``(question, contexts) -> answer``.
                      Required for generation + hallucination metrics.
            top_k:    Number of documents to retrieve per query.
        """
        report = EvalReport(dataset_name=dataset.name)

        retrieved_lists: list[list[str]] = []
        relevant_lists: list[list[str]] = []
        questions: list[str] = []
        answers: list[str] = []
        contexts_list: list[list[str]] = []
        ground_truths: list[str] = []

        for sample in dataset:
            results = self.backend.search(sample.question, k=top_k)
            retrieved_ids = [r.document.id for r in results]
            retrieved_texts = [r.document.text for r in results]

            retrieved_lists.append(retrieved_ids)
            relevant_lists.append(sample.relevant_doc_ids)
            questions.append(sample.question)
            contexts_list.append(retrieved_texts)

            if sample.ground_truth_answer:
                ground_truths.append(sample.ground_truth_answer)

            per: dict[str, Any] = {
                "question": sample.question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sample.relevant_doc_ids,
            }

            if rag_fn is not None:
                answer = rag_fn(sample.question, retrieved_texts)
                answers.append(answer)
                per["answer"] = answer

            report.per_sample.append(per)

        report.retrieval = self._retrieval_metrics.compute(retrieved_lists, relevant_lists)

        if rag_fn is not None and self.config.run_generation_metrics and answers:
            from rag_eval.metrics.generation import GenerationMetrics

            default_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            gm = GenerationMetrics(metrics=self.config.generation_metrics or default_metrics)
            report.generation = gm.compute(
                questions=questions,
                answers=answers,
                contexts=contexts_list,
                ground_truths=ground_truths or None,
            )

        if rag_fn is not None and self.config.run_hallucination_detection and answers:
            from rag_eval.metrics.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            h_results = detector.check_batch(answers, contexts_list)
            report.hallucination = detector.aggregate(h_results)

            for i, hr in enumerate(h_results):
                report.per_sample[i]["hallucination"] = hr.to_dict()

        return report
