"""
Dataset Manager Service.

Handles:
  - Loading datasets from JSON, CSV, and HuggingFace
  - Deterministic versioning via SHA-256 hash of dataset content
  - Configurable train / eval split
  - Persisting loaded datasets and run results keyed by (dataset_version, run_id)
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DatasetRecord:
    """An ingested and versioned dataset."""

    dataset_id: str           # SHA-256 hash of canonical content
    name: str
    source: str               # 'json' | 'csv' | 'huggingface'
    sample_count: int
    samples: list[dict[str, Any]]
    documents: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id":   self.dataset_id,
            "name":         self.name,
            "source":       self.source,
            "sample_count": self.sample_count,
            "samples":      self.samples,
            "documents":    self.documents,
            "created_at":   self.created_at,
            "metadata":     self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetRecord":
        return cls(**d)


class DatasetManager:
    """
    Manages dataset lifecycle: loading, versioning, splitting, and result storage.

    Directory layout (under ``data_dir``)::

        datasets/
            {dataset_id}.json          ← persisted DatasetRecord
        results/
            {dataset_id}/
                {run_id}.json          ← persisted run result

    Usage::

        mgr = DatasetManager()
        dataset_id = mgr.load_from_json("examples/sample_dataset.json", name="capitals")
        record = mgr.get(dataset_id)
        train, eval = mgr.train_eval_split(dataset_id, train_ratio=0.8)
    """

    def __init__(self, data_dir: str | None = None) -> None:
        root = Path(data_dir or os.getenv("DATA_DIR", "data"))
        self._datasets_dir = root / "datasets"
        self._results_dir  = root / "results"
        self._datasets_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache to avoid re-reading disk repeatedly
        self._cache: dict[str, DatasetRecord] = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_from_json(
        self,
        path: str | Path,
        name: str | None = None,
        documents: list[dict] | None = None,
    ) -> str:
        """
        Load an eval dataset from a JSON file.

        Expected format::

            [{"question": "...", "relevant_doc_ids": [...], "ground_truth_answer": "..."}]

        Returns the ``dataset_id`` (SHA-256 hash).
        """
        raw = json.loads(Path(path).read_text())
        samples = _normalise_samples(raw)
        return self._ingest(
            samples=samples,
            documents=documents or [],
            name=name or Path(path).stem,
            source="json",
        )

    def load_from_csv(
        self,
        path: str | Path,
        name: str | None = None,
        documents: list[dict] | None = None,
    ) -> str:
        """
        Load from a CSV file.

        Required columns: ``question``, ``relevant_doc_ids`` (comma-separated IDs).
        Optional columns: ``ground_truth_answer``.
        """
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        samples = _normalise_samples(rows)
        return self._ingest(
            samples=samples,
            documents=documents or [],
            name=name or Path(path).stem,
            source="csv",
        )

    def load_from_hf(
        self,
        dataset_name: str,
        split: str = "test",
        name: str | None = None,
        documents: list[dict] | None = None,
        **hf_kwargs,
    ) -> str:
        """
        Load from a HuggingFace datasets hub.

        The dataset must have ``question`` and ``relevant_doc_ids`` columns.
        """
        from datasets import load_dataset as hf_load

        hf_ds = hf_load(dataset_name, split=split, **hf_kwargs)
        samples = _normalise_samples(list(hf_ds))
        return self._ingest(
            samples=samples,
            documents=documents or [],
            name=name or dataset_name,
            source="huggingface",
            metadata={"hf_dataset": dataset_name, "split": split},
        )

    def load_inline(
        self,
        samples: list[dict],
        name: str = "inline",
        documents: list[dict] | None = None,
    ) -> str:
        """Ingest samples provided directly as a list of dicts."""
        return self._ingest(
            samples=_normalise_samples(samples),
            documents=documents or [],
            name=name,
            source="inline",
        )

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, dataset_id: str) -> DatasetRecord:
        """Retrieve a dataset record by its SHA-256 ID."""
        if dataset_id in self._cache:
            return self._cache[dataset_id]
        path = self._datasets_dir / f"{dataset_id}.json"
        if not path.exists():
            raise KeyError(f"Dataset '{dataset_id}' not found in {self._datasets_dir}")
        record = DatasetRecord.from_dict(json.loads(path.read_text()))
        self._cache[dataset_id] = record
        return record

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return summary metadata for all stored datasets."""
        summaries = []
        for p in self._datasets_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                summaries.append({
                    "dataset_id":   d["dataset_id"],
                    "name":         d["name"],
                    "source":       d["source"],
                    "sample_count": d["sample_count"],
                    "created_at":   d["created_at"],
                })
            except Exception:
                pass
        return summaries

    # ── Splitting ─────────────────────────────────────────────────────────────

    def train_eval_split(
        self,
        dataset_id: str,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[dict], list[dict]]:
        """
        Split dataset samples into train and eval subsets.

        Returns (train_samples, eval_samples).
        Splitting is deterministic for a given seed.
        """
        import random

        record = self.get(dataset_id)
        samples = list(record.samples)
        rng = random.Random(seed)
        rng.shuffle(samples)
        split_idx = max(1, int(len(samples) * train_ratio))
        return samples[:split_idx], samples[split_idx:]

    # ── Result storage ────────────────────────────────────────────────────────

    def store_result(self, dataset_id: str, run_id: str, result: dict) -> None:
        """Persist a run result keyed by (dataset_version, run_id)."""
        run_dir = self._results_dir / dataset_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2))

    def get_result(self, dataset_id: str, run_id: str) -> dict | None:
        """Load a previously stored run result. Returns None if not found."""
        path = self._results_dir / dataset_id / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_results(self, dataset_id: str) -> list[str]:
        """List all run IDs that have results for this dataset."""
        run_dir = self._results_dir / dataset_id
        if not run_dir.exists():
            return []
        return [p.stem for p in run_dir.glob("*.json")]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _ingest(
        self,
        samples: list[dict],
        documents: list[dict],
        name: str,
        source: str,
        metadata: dict | None = None,
    ) -> str:
        dataset_id = _compute_hash(samples)
        path = self._datasets_dir / f"{dataset_id}.json"

        if not path.exists():
            record = DatasetRecord(
                dataset_id=dataset_id,
                name=name,
                source=source,
                sample_count=len(samples),
                samples=samples,
                documents=documents,
                metadata=metadata or {},
            )
            path.write_text(json.dumps(record.to_dict(), indent=2))
            self._cache[dataset_id] = record

        return dataset_id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_samples(raw: list[dict]) -> list[dict]:
    """Ensure every sample has the required keys; normalise relevant_doc_ids field."""
    out = []
    for item in raw:
        # CSV rows may have doc IDs as a comma-separated string
        rel_ids = item.get("relevant_doc_ids", [])
        if isinstance(rel_ids, str):
            rel_ids = [r.strip() for r in rel_ids.split(",") if r.strip()]
        out.append({
            "question":            str(item.get("question", "")),
            "relevant_doc_ids":    rel_ids,
            "ground_truth_answer": item.get("ground_truth_answer") or item.get("answer"),
            "metadata":            item.get("metadata", {}),
        })
    return out


def _compute_hash(samples: list[dict]) -> str:
    """SHA-256 hash of the canonical JSON representation of the samples."""
    canonical = json.dumps(samples, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
