"""Dataset schema and loaders for RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    relevant_doc_ids: list[str]
    ground_truth_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of evaluation samples."""

    samples: list[EvalSample] = field(default_factory=list)
    name: str = "dataset"

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        """Load from a JSON file with schema: [{"question": ..., "relevant_doc_ids": [...], ...}]"""
        data = json.loads(Path(path).read_text())
        samples = [
            EvalSample(
                question=item["question"],
                relevant_doc_ids=item["relevant_doc_ids"],
                ground_truth_answer=item.get("ground_truth_answer"),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        return cls(samples=samples, name=Path(path).stem)

    @classmethod
    def from_dicts(cls, records: list[dict], name: str = "dataset") -> "EvalDataset":
        return cls(
            samples=[
                EvalSample(
                    question=r["question"],
                    relevant_doc_ids=r["relevant_doc_ids"],
                    ground_truth_answer=r.get("ground_truth_answer"),
                    metadata=r.get("metadata", {}),
                )
                for r in records
            ],
            name=name,
        )

    def to_json(self, path: str | Path) -> None:
        records = [
            {
                "question": s.question,
                "relevant_doc_ids": s.relevant_doc_ids,
                "ground_truth_answer": s.ground_truth_answer,
                "metadata": s.metadata,
            }
            for s in self.samples
        ]
        Path(path).write_text(json.dumps(records, indent=2))

    @classmethod
    def from_hf(cls, dataset_name: str, split: str = "test", **kwargs) -> "EvalDataset":
        """Load from a HuggingFace dataset (must have 'question' and 'relevant_doc_ids' columns)."""
        from datasets import load_dataset

        hf_ds = load_dataset(dataset_name, split=split, **kwargs)
        return cls.from_dicts(list(hf_ds), name=dataset_name)
