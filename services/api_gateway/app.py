"""
API Gateway — extends the base RAG eval FastAPI app with multi-service endpoints.

New routes added on top of the existing rag_eval.api.app routes:

  POST /run/full-eval          Run all evaluation layers; returns run_id
  GET  /reports/{run_id}       Full run report (retrieval + generation + hallucination + cost)
  GET  /failures/{run_id}      Failure log entries for a run
  GET  /cost/{run_id}          Cost breakdown for a run
  POST /datasets/load          Ingest and version a dataset
  GET  /datasets               List all versioned datasets
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Import base app (preserves all existing routes) ──────────────────────────
from rag_eval.api.app import create_app as _create_base_app
from rag_eval.backends import get_backend
from rag_eval.backends.base import Document
from rag_eval.llm.claude import ClaudeLLM
from shared.logging.logger import FailureLogger
from shared.schemas.models import CostReport, RunReport
from services.dataset_manager.manager import DatasetManager
from services.model_runner.runner import ModelRunner
from services.retrieval_evaluator.evaluator import RetrievalEvaluatorService
from services.generation_evaluator.evaluator import GenerationEvaluatorService


# ── Request / Response models ─────────────────────────────────────────────────

class FullEvalRequest(BaseModel):
    """Request body for POST /run/full-eval."""

    # Dataset — provide inline OR reference a pre-loaded dataset_id
    dataset_id: str | None = None
    samples: list[dict[str, Any]] | None = None
    documents: list[dict[str, Any]] | None = None

    # Retrieval config
    backend: str = "faiss"
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    k_values: list[int] = Field(default=[1, 3, 5, 10])

    # Generation config
    run_generation: bool = False
    run_hallucination: bool = False
    llm_provider: str = "anthropic"
    llm_model: str | None = None

    # Metadata
    name: str = "unnamed"


class DatasetLoadRequest(BaseModel):
    name: str = "dataset"
    samples: list[dict[str, Any]]
    documents: list[dict[str, Any]] = Field(default_factory=list)


# ── Gateway factory ───────────────────────────────────────────────────────────

def create_gateway_app(
    log_dir: str | None = None,
    data_dir: str | None = None,
) -> FastAPI:
    """
    Create the full API gateway.

    Starts from the base app (all /index, /search, /metrics/* routes included)
    and appends the multi-service orchestration endpoints.
    """
    app = _create_base_app()
    app.title = "RAG Evaluation Platform — API Gateway"
    app.description = (
        "Full-stack RAG evaluation: retrieval, generation, hallucination, cost tracking, "
        "failure logging, dataset versioning, and experiment comparison."
    )
    app.version = "0.3.0"

    _log_dir  = Path(log_dir  or os.getenv("LOG_DIR",  "logs"))
    _data_dir = Path(data_dir or os.getenv("DATA_DIR", "data"))
    _reports_dir = _log_dir / "reports"
    _reports_dir.mkdir(parents=True, exist_ok=True)

    _dataset_mgr = DatasetManager(data_dir=str(_data_dir))

    # ── Dataset endpoints ─────────────────────────────────────────────────────

    @app.post("/datasets/load")
    async def load_dataset(req: DatasetLoadRequest):
        """Ingest samples (and optional documents) into the dataset manager. Returns dataset_id."""
        dataset_id = _dataset_mgr.load_inline(
            samples=req.samples,
            name=req.name,
            documents=req.documents or [],
        )
        record = _dataset_mgr.get(dataset_id)
        return {
            "dataset_id":   dataset_id,
            "name":         record.name,
            "sample_count": record.sample_count,
            "created_at":   record.created_at,
        }

    @app.get("/datasets")
    async def list_datasets():
        """List all versioned datasets."""
        return _dataset_mgr.list_datasets()

    # ── Full evaluation endpoint ──────────────────────────────────────────────

    @app.post("/run/full-eval")
    async def full_eval(req: FullEvalRequest):
        """
        Run the complete evaluation pipeline:
          1. Index documents into the vector store
          2. Retrieval evaluation (Recall@K, MRR, NDCG, latency)
          3. (Optional) Generation evaluation + hallucination detection via Claude
          4. Persist RunReport and return run_id

        Provide either ``dataset_id`` (from /datasets/load) or inline ``samples`` + ``documents``.
        """
        run_id = str(uuid.uuid4())[:8]
        logger = FailureLogger(run_id=run_id, log_dir=str(_log_dir))

        # ── Resolve dataset ───────────────────────────────────────────────────
        if req.dataset_id:
            try:
                record = _dataset_mgr.get(req.dataset_id)
            except KeyError:
                raise HTTPException(404, f"Dataset '{req.dataset_id}' not found. Call /datasets/load first.")
            samples   = record.samples
            documents = record.documents
            dataset_id = req.dataset_id
        elif req.samples:
            dataset_id = _dataset_mgr.load_inline(
                samples=req.samples,
                name=req.name,
                documents=req.documents or [],
            )
            record    = _dataset_mgr.get(dataset_id)
            samples   = record.samples
            documents = record.documents
        else:
            raise HTTPException(400, "Provide either 'dataset_id' or 'samples'.")

        # ── Build report skeleton ─────────────────────────────────────────────
        report = RunReport(
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_name=record.name,
            status="running",
            config={
                "backend":       req.backend,
                "top_k":         req.top_k,
                "k_values":      req.k_values,
                "run_generation": req.run_generation,
                "llm_provider":  req.llm_provider,
            },
        )

        # ── Index documents ───────────────────────────────────────────────────
        backend = get_backend(req.backend, **req.backend_kwargs)
        if documents:
            docs = [Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
                    for d in documents]
            backend.add_documents(docs)

        queries       = [s["question"]            for s in samples]
        relevant_lists = [s.get("relevant_doc_ids", []) for s in samples]
        ground_truths  = [s.get("ground_truth_answer") for s in samples]

        # ── Retrieval evaluation ──────────────────────────────────────────────
        ret_service = RetrievalEvaluatorService(
            backend=backend,
            k_values=req.k_values,
            failure_logger=logger,
        )
        ret_results = ret_service.evaluate_dataset(run_id, queries, relevant_lists, top_k=req.top_k)
        report.aggregate_retrieval = ret_service.aggregate(ret_results)

        # ── Generation + hallucination (optional) ─────────────────────────────
        cost_report: CostReport | None = None
        gen_results = []

        if (req.run_generation or req.run_hallucination) and not ModelRunner.is_available(req.llm_provider):
            report.status = "completed_partial"
            report.error  = (
                f"Generation skipped: {req.llm_provider.upper()}_API_KEY is not set. "
                "Set it in your environment to enable generation metrics."
            )
        elif req.run_generation:
            runner = ModelRunner(
                provider=req.llm_provider,
                model=req.llm_model,
                failure_logger=logger,
            )
            # Build answers via simple RAG (retrieved context → LLM)
            answers: list[str] = []
            for i, (q, ret_res) in enumerate(zip(queries, ret_results)):
                ctx_texts = [
                    next((d["text"] for d in documents if d["id"] == rid), "")
                    for rid in ret_res.retrieved_ids[:3]
                ] if documents else []
                try:
                    answer = runner.complete(
                        f"Answer using only this context:\n\n{'  '.join(ctx_texts)}\n\nQuestion: {q}",
                        system="Be concise. Use only the provided context.",
                    )
                except Exception as e:
                    answer = f"[generation failed: {e}]"
                answers.append(answer)

            gen_service = GenerationEvaluatorService(
                runner=runner,
                failure_logger=logger,
            )
            gen_results = gen_service.evaluate_batch(
                run_id, queries, answers,
                [[d["text"] for d in documents if d["id"] in r.retrieved_ids] for r in ret_results],
                ground_truths,
            )
            report.aggregate_generation = gen_service.aggregate(gen_results)
            cost_report_dict = gen_service.get_cost_report(run_id, gen_results)
            cost_report = CostReport(**cost_report_dict)
            report.cost_report = cost_report

        # ── Hallucination detection ───────────────────────────────────────────
        if req.run_hallucination and gen_results:
            from rag_eval.metrics.hallucination import HallucinationDetector
            from rag_eval.llm.claude import ClaudeLLM as _ClaudeLLM

            detector = HallucinationDetector(llm=_ClaudeLLM())
            h_results = detector.check_batch(
                [r.answer for r in gen_results],
                [r.contexts for r in gen_results],
            )
            report.aggregate_hallucination = detector.aggregate(h_results)

        # ── Assemble per-sample results ───────────────────────────────────────
        from shared.schemas.models import EvalResult

        for i, ret_res in enumerate(ret_results):
            gen = gen_results[i] if gen_results and i < len(gen_results) else None
            er  = EvalResult(
                run_id=run_id,
                question=ret_res.query,
                retrieved_ids=ret_res.retrieved_ids,
                relevant_ids=ret_res.relevant_ids,
                answer=gen.answer if gen else None,
                retrieval_metrics=ret_res.metrics,
                generation_metrics=gen.metrics if gen else {},
                latency_ms=ret_res.latency_ms,
                input_tokens=gen.input_tokens if gen else 0,
                output_tokens=gen.output_tokens if gen else 0,
                estimated_cost_usd=gen.estimated_cost_usd if gen else 0.0,
                is_slow_query=ret_res.is_slow,
                is_faithfulness_failure=gen.is_faithfulness_failure if gen else False,
            )
            report.per_sample.append(er)

        report.failure_count = logger.count
        report.completed_at  = datetime.utcnow().isoformat()
        report.status        = report.status if report.status != "running" else "completed"

        # ── Persist report and results ────────────────────────────────────────
        report_path = _reports_dir / f"{run_id}.json"
        report_path.write_text(report.model_dump_json(indent=2))
        _dataset_mgr.store_result(dataset_id, run_id, json.loads(report.model_dump_json()))

        return {"run_id": run_id, "status": report.status, "dataset_id": dataset_id}

    # ── Report endpoints ──────────────────────────────────────────────────────

    @app.get("/reports/{run_id}")
    async def get_report(run_id: str):
        """Return the full RunReport for a completed run."""
        report_path = _reports_dir / f"{run_id}.json"
        if not report_path.exists():
            raise HTTPException(404, f"Report for run '{run_id}' not found.")
        return json.loads(report_path.read_text())

    @app.get("/failures/{run_id}")
    async def get_failures(run_id: str):
        """Return only the failure log entries for a run."""
        loader = FailureLogger.load(run_id=run_id, log_dir=str(_log_dir))
        failures = loader.get_all()
        return {
            "run_id": run_id,
            "count":    len(failures),
            "failures": [f.model_dump() for f in failures],
        }

    @app.get("/cost/{run_id}")
    async def get_cost(run_id: str):
        """Return the cost breakdown for a run."""
        report_path = _reports_dir / f"{run_id}.json"
        if not report_path.exists():
            raise HTTPException(404, f"Report for run '{run_id}' not found.")
        data = json.loads(report_path.read_text())
        return {
            "run_id":       run_id,
            "cost_report":  data.get("cost_report"),
            "total_samples": len(data.get("per_sample", [])),
        }

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_gateway_app(), host="0.0.0.0", port=8080)
