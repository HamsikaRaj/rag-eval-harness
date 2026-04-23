"""Retrieval quality metrics: Recall@K, MRR, NDCG@K."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in the top-K retrieved results."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for doc_id in relevant if doc_id in top_k)
    return hits / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant document in the retrieved list."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    By default uses binary relevance (1 if relevant, 0 otherwise).
    Pass ``relevance_scores`` for graded relevance: {doc_id: score}.
    """
    relevant_set = set(relevant)

    def gain(doc_id: str) -> float:
        if relevance_scores:
            return relevance_scores.get(doc_id, 0.0)
        return 1.0 if doc_id in relevant_set else 0.0

    dcg = sum(
        gain(doc_id) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
    )

    ideal_gains = sorted(
        (gain(d) for d in (relevant if not relevance_scores else relevance_scores)),
        reverse=True,
    )
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains[:k], start=1))

    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over a dataset of queries."""

    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    def compute(
        self,
        retrieved_lists: list[list[str]],
        relevant_lists: list[list[str]],
        relevance_scores: list[dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """
        Compute mean Recall@K, MRR, and NDCG@K across all queries.

        Args:
            retrieved_lists: Per-query ordered list of retrieved doc IDs.
            relevant_lists:  Per-query list of ground-truth relevant doc IDs.
            relevance_scores: Optional per-query graded relevance dicts.
        """
        n = len(retrieved_lists)
        assert n == len(relevant_lists), "retrieved and relevant lists must have equal length"

        results: dict[str, float] = {}

        for k in self.k_values:
            results[f"recall@{k}"] = (
                sum(recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_lists)) / n
            )
            results[f"ndcg@{k}"] = sum(
                ndcg_at_k(
                    r,
                    rel,
                    k,
                    relevance_scores[i] if relevance_scores else None,
                )
                for i, (r, rel) in enumerate(zip(retrieved_lists, relevant_lists))
            ) / n

        results["mrr"] = (
            sum(mrr(r, rel) for r, rel in zip(retrieved_lists, relevant_lists)) / n
        )

        return results
