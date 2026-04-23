"""
Generation Evaluator Service.

Extends the core RAGAS-style generation metrics with:
  - Per-sample token count and cost estimation via ModelRunner
  - Failure logging when faithfulness score falls below the configured threshold
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.metrics.generation import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from shared.cost.estimator import estimate_cost

if TYPE_CHECKING:
    from shared.logging.logger import FailureLogger
    from services.model_runner.runner import ModelRunner


@dataclass
class GenerationEvalResult:
    """Per-sample generation evaluation result with cost metadata."""

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None
    metrics: dict[str, float]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    is_faithfulness_failure: bool


class GenerationEvaluatorService:
    """
    Evaluates generation quality using RAGAS-style metrics (faithfulness, answer relevancy,
    context precision, context recall) with a pluggable ModelRunner as the LLM judge.

    Extras beyond core GenerationMetrics:
      - Cumulative token and cost tracking via ModelRunner
      - Failure logging when faithfulness < ``faithfulness_threshold``

    Usage::

        runner  = ModelRunner(provider="anthropic")
        service = GenerationEvaluatorService(runner=runner, failure_logger=logger)
        results = service.evaluate_batch(run_id, queries, answers, contexts, ground_truths)
        cost    = service.get_cost_report(run_id, results)
    """

    SERVICE_NAME = "generation_evaluator"

    def __init__(
        self,
        runner: "ModelRunner",
        metrics: list[str] | None = None,
        faithfulness_threshold: float | None = None,
        failure_logger: "FailureLogger | None" = None,
    ) -> None:
        self.runner = runner
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        self.faithfulness_threshold = faithfulness_threshold or float(
            os.getenv("FAITHFULNESS_FAILURE_THRESHOLD", "0.5")
        )
        self._failure_logger = failure_logger

    def evaluate_sample(
        self,
        run_id: str,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> GenerationEvalResult:
        """Evaluate one question-answer-context triple."""
        self.runner.reset_usage()
        scores: dict[str, float] = {}

        # The ModelRunner is duck-type compatible with ClaudeLLM — same complete_json interface
        if "faithfulness" in self.metrics:
            scores["faithfulness"] = faithfulness(query, answer, contexts, self.runner)

        if "answer_relevancy" in self.metrics:
            scores["answer_relevancy"] = answer_relevancy(query, answer, self.runner)

        if "context_precision" in self.metrics:
            scores["context_precision"] = context_precision(query, contexts, self.runner)

        if "context_recall" in self.metrics and ground_truth:
            scores["context_recall"] = context_recall(ground_truth, contexts, self.runner)

        usage = self.runner.get_usage()
        in_tok  = usage["input_tokens"]
        out_tok = usage["output_tokens"]
        cost    = estimate_cost(self.runner.model, in_tok, out_tok)

        faith_score = scores.get("faithfulness", 1.0)
        is_failure  = faith_score < self.faithfulness_threshold

        if is_failure and self._failure_logger:
            self._failure_logger.log(
                service=self.SERVICE_NAME,
                query=query,
                issue=f"faithfulness_below_threshold: {faith_score:.3f} < {self.faithfulness_threshold}",
                retrieved=contexts[:3],
                expected=[ground_truth] if ground_truth else [],
            )

        return GenerationEvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=scores,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=round(cost, 6),
            is_faithfulness_failure=is_failure,
        )

    def evaluate_batch(
        self,
        run_id: str,
        queries: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | None] | None = None,
    ) -> list[GenerationEvalResult]:
        """Evaluate a list of samples."""
        gts = ground_truths or [None] * len(queries)
        return [
            self.evaluate_sample(run_id, q, a, ctx, gt)
            for q, a, ctx, gt in zip(queries, answers, contexts, gts)
        ]

    def aggregate(self, results: list[GenerationEvalResult]) -> dict[str, float]:
        """Mean scores + total token usage across all results."""
        if not results:
            return {}
        n = len(results)
        agg: dict[str, float] = {}

        all_metric_keys = {k for r in results for k in r.metrics}
        for key in all_metric_keys:
            vals = [r.metrics[key] for r in results if key in r.metrics]
            agg[key] = sum(vals) / len(vals) if vals else 0.0

        agg["total_input_tokens"]    = float(sum(r.input_tokens for r in results))
        agg["total_output_tokens"]   = float(sum(r.output_tokens for r in results))
        agg["total_cost_usd"]        = round(sum(r.estimated_cost_usd for r in results), 6)
        agg["faithfulness_failures"] = float(sum(1 for r in results if r.is_faithfulness_failure))

        return agg

    def get_cost_report(self, run_id: str, results: list[GenerationEvalResult]) -> dict:
        """Build the CostReport-compatible dict for this service."""
        from shared.schemas.models import CostReport

        total_in  = sum(r.input_tokens for r in results)
        total_out = sum(r.output_tokens for r in results)
        return CostReport(
            run_id=run_id,
            model=self.runner.model,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            estimated_cost_usd=round(estimate_cost(self.runner.model, total_in, total_out), 6),
            breakdown={self.SERVICE_NAME: round(estimate_cost(self.runner.model, total_in, total_out), 6)},
        ).model_dump()
