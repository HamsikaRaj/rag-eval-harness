"""
Structured failure logger.

Writes one JSON file per run to {log_dir}/failures_{run_id}.json.
Each entry follows the FailureLog schema from shared.schemas.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from shared.schemas.models import FailureLog


class FailureLogger:
    """
    Appends structured failure records to a per-run JSON log file.

    Usage::

        logger = FailureLogger(run_id="abc123")
        logger.log(
            service="retrieval_evaluator",
            query="What is RBAC?",
            issue="zero_relevant_docs_retrieved",
            retrieved=["doc_5"],
            expected=["doc_3"],
            latency_ms=340,
        )
        failures = logger.get_all()
    """

    def __init__(self, run_id: str, log_dir: str | None = None) -> None:
        self.run_id = run_id
        self._log_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / f"failures_{run_id}.json"
        self._records: list[FailureLog] = []
        # Load existing records if the file already exists (resumable runs)
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._records = [FailureLog(**r) for r in raw]
            except Exception:
                self._records = []

    def log(
        self,
        service: str,
        query: str,
        issue: str,
        retrieved: list[str] | None = None,
        expected: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> FailureLog:
        """Create and persist a failure record. Returns the created FailureLog."""
        record = FailureLog(
            run_id=self.run_id,
            timestamp=datetime.utcnow().isoformat(),
            service=service,
            query=query,
            issue=issue,
            retrieved=retrieved or [],
            expected=expected or [],
            latency_ms=latency_ms,
        )
        self._records.append(record)
        self._flush()
        return record

    def get_all(self) -> list[FailureLog]:
        """Return all failure records for this run."""
        return list(self._records)

    def get_by_service(self, service: str) -> list[FailureLog]:
        """Filter failures by service name."""
        return [r for r in self._records if r.service == service]

    @property
    def count(self) -> int:
        return len(self._records)

    def _flush(self) -> None:
        """Write all records to disk (overwrites — records are append-only)."""
        self._path.write_text(
            json.dumps([r.model_dump() for r in self._records], indent=2)
        )

    @classmethod
    def load(cls, run_id: str, log_dir: str | None = None) -> "FailureLogger":
        """Load an existing failure log from disk."""
        return cls(run_id=run_id, log_dir=log_dir)
