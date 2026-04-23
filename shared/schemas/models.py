"""Pydantic models shared across all services."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single evaluation query with ground-truth annotations."""

    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    ground_truth_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Per-sample evaluation result combining retrieval, generation, and system metrics."""

    run_id: str
    question: str
    retrieved_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    is_slow_query: bool = False
    is_faithfulness_failure: bool = False


class FailureLog(BaseModel):
    """Structured failure record written to logs/failures_{run_id}.json."""

    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    service: str
    query: str
    issue: str
    retrieved: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class CostReport(BaseModel):
    """Aggregated cost report for a single evaluation run."""

    run_id: str
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class RunReport(BaseModel):
    """Complete report for one full evaluation run."""

    run_id: str
    dataset_id: str = ""
    dataset_name: str = ""
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    status: str = "pending"  # pending | running | completed | failed
    config: dict[str, Any] = Field(default_factory=dict)
    aggregate_retrieval: dict[str, float] = Field(default_factory=dict)
    aggregate_generation: dict[str, float] = Field(default_factory=dict)
    aggregate_hallucination: dict[str, Any] = Field(default_factory=dict)
    cost_report: CostReport | None = None
    per_sample: list[EvalResult] = Field(default_factory=list)
    failure_count: int = 0
    error: str | None = None
