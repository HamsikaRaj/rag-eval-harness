"""
Retrieval Evaluator Service.

Extends the core retrieval metrics with:
  - Per-query latency tracking (ms)
  - Slow-query flagging (configurable threshold via SLOW_QUERY_THRESHOLD_MS env var)
  - Failure logging for queries that retrieved zero relevant documents
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.backends.base import VectorStoreBackend
from rag_eval.metrics.retrieval import RetrievalMetrics, recall_at_k, mrr, ndcg_at_k

if TYPE_CHECKING:
    from shared.logging.logger import FailureLogger


@dataclass
class RetrievalEvalResult:
    """Per-query retrieval evaluation result."""

    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float
    is_slow: bool
    has_zero_relevant: bool


class RetrievalEvaluatorService:
    """
    Evaluates retrieval quality for a set of queries against a vector store backend.

    Features beyond the core RetrievalMetrics:
      - Wall-clock latency per query
      - Flags queries slower than ``slow_threshold_ms``
      - Logs to ``failure_logger`` when zero relevant documents are retrieved

    Usage::

        service = RetrievalEvaluatorService(backend=FAISSBackend(), failure_logger=logger)
        results = service.evaluate_dataset(run_id, dataset, top_k=10)
        agg = service.aggregate(results)
    """

    SERVICE_NAME = "retrieval_evaluator"

    def __init__(
        self,
        backend: VectorStoreBackend,
        k_values: list[int] | None = None,
        slow_threshold_ms: float | None = None,
        failure_logger: "FailureLogger | None" = None,
    ) -> None:
        self.backend = backend
        self.k_values = k_values or [1, 3, 5, 10]
        self.slow_threshold_ms = slow_threshold_ms or float(
            os.getenv("SLOW_QUERY_THRESHOLD_MS", "500")
        )
        self._failure_logger = failure_logger
        self._retrieval_metrics = RetrievalMetrics(k_values=self.k_values)

    def evaluate_query(
        self,
        run_id: str,
        query: str,
        relevant_ids: list[str],
        top_k: int = 10,
    ) -> RetrievalEvalResult:
        """Evaluate a single query and return a detailed result."""
        t0 = time.perf_counter()
        search_results = self.backend.search(query, k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = [r.document.id for r in search_results]
        is_slow = latency_ms > self.slow_threshold_ms

        # Compute per-query metrics
        metrics: dict[str, float] = {}
        for k in self.k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"]   = ndcg_at_k(retrieved_ids, relevant_ids, k)
        metrics["mrr"] = mrr(retrieved_ids, relevant_ids)

        # Determine failure: none of the relevant docs were in top_k results
        retrieved_set  = set(retrieved_ids)
        relevant_set   = set(relevant_ids)
        has_zero_relevant = bool(relevant_ids) and retrieved_set.isdisjoint(relevant_set)

        if has_zero_relevant and self._failure_logger:
            self._failure_logger.log(
                service=self.SERVICE_NAME,
                query=query,
                issue="zero_relevant_docs_retrieved",
                retrieved=retrieved_ids,
                expected=relevant_ids,
                latency_ms=round(latency_ms, 2),
            )

        return RetrievalEvalResult(
            query=query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            metrics=metrics,
            latency_ms=round(latency_ms, 2),
            is_slow=is_slow,
            has_zero_relevant=has_zero_relevant,
        )

    def evaluate_dataset(
        self,
        run_id: str,
        queries: list[str],
        relevant_lists: list[list[str]],
        top_k: int = 10,
    ) -> list[RetrievalEvalResult]:
        """Evaluate all queries in a dataset."""
        assert len(queries) == len(relevant_lists)
        return [
            self.evaluate_query(run_id, q, rel, top_k)
            for q, rel in zip(queries, relevant_lists)
        ]

    def aggregate(self, results: list[RetrievalEvalResult]) -> dict[str, float]:
        """
        Compute mean metrics across all results plus latency statistics.
        """
        if not results:
            return {}

        n = len(results)
        agg: dict[str, float] = {}

        # Aggregate metric scores
        all_metric_keys = results[0].metrics.keys()
        for key in all_metric_keys:
            agg[key] = sum(r.metrics[key] for r in results) / n

        # Latency stats
        latencies = [r.latency_ms for r in results]
        agg["latency_ms_mean"]  = round(sum(latencies) / n, 2)
        agg["latency_ms_max"]   = round(max(latencies), 2)
        agg["latency_ms_min"]   = round(min(latencies), 2)
        agg["slow_query_count"] = float(sum(1 for r in results if r.is_slow))
        agg["zero_relevant_count"] = float(sum(1 for r in results if r.has_zero_relevant))

        return agg
