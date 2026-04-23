"""FastAPI application exposing the RAG evaluation harness as an HTTP service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_eval.backends import get_backend
from rag_eval.metrics.retrieval import RetrievalMetrics
from rag_eval.metrics.hallucination import HallucinationDetector
from rag_eval.llm.claude import ClaudeLLM


# ── Request / Response models ─────────────────────────────────────────────────

class DocumentIn(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    backend: str = "faiss"
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)
    documents: list[DocumentIn]


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class RetrievalMetricsRequest(BaseModel):
    retrieved_lists: list[list[str]]
    relevant_lists: list[list[str]]
    k_values: list[int] = Field(default=[1, 3, 5, 10])


class HallucinationRequest(BaseModel):
    answers: list[str]
    sources_list: list[list[str]]


class EvalRequest(BaseModel):
    backend: str = "faiss"
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)
    documents: list[DocumentIn]
    samples: list[dict[str, Any]]
    k_values: list[int] = Field(default=[1, 3, 5, 10])
    top_k: int = 10


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Evaluation Harness",
        description=(
            "Modular toolkit for evaluating RAG pipelines — retrieval metrics, "
            "RAGAS-style generation metrics via Claude, and hallucination detection."
        ),
        version="0.2.0",
    )

    # In-memory backend registry keyed by session id
    _backends: dict[str, Any] = {}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "claude_available": ClaudeLLM.is_available(),
        }

    @app.post("/index")
    async def index_documents(req: IndexRequest):
        """Embed and index documents into a fresh backend instance."""
        from rag_eval.backends.base import Document

        backend = get_backend(req.backend, **req.backend_kwargs)
        docs = [Document(id=d.id, text=d.text, metadata=d.metadata) for d in req.documents]
        backend.add_documents(docs)
        session_id = f"{req.backend}_{id(backend)}"
        _backends[session_id] = backend
        return {"session_id": session_id, "indexed": len(docs)}

    @app.post("/search/{session_id}")
    async def search(session_id: str, req: SearchRequest):
        """Search an indexed backend by query string."""
        backend = _backends.get(session_id)
        if not backend:
            raise HTTPException(404, f"Session '{session_id}' not found. Call /index first.")
        results = backend.search(req.query, k=req.k)
        return {
            "query": req.query,
            "results": [
                {"id": r.document.id, "text": r.document.text, "score": r.score, "rank": r.rank}
                for r in results
            ],
        }

    @app.post("/metrics/retrieval")
    async def retrieval_metrics(req: RetrievalMetricsRequest):
        """Compute Recall@K, MRR, and NDCG@K from pre-computed ID lists."""
        rm = RetrievalMetrics(k_values=req.k_values)
        return rm.compute(req.retrieved_lists, req.relevant_lists)

    @app.post("/metrics/hallucination")
    async def hallucination_metrics(req: HallucinationRequest):
        """Detect hallucinations in a batch of answers using Claude."""
        if not ClaudeLLM.is_available():
            raise HTTPException(
                503,
                "ANTHROPIC_API_KEY is not set. "
                "Set it in your environment to enable hallucination detection.",
            )
        detector = HallucinationDetector()
        results = detector.check_batch(req.answers, req.sources_list)
        return {
            "aggregate": detector.aggregate(results),
            "per_sample": [r.to_dict() for r in results],
        }

    @app.post("/evaluate")
    async def evaluate(req: EvalRequest):
        """
        Full retrieval-only evaluation: index docs, run queries, return retrieval metrics.

        For generation + hallucination metrics, use the pipeline directly or set
        ANTHROPIC_API_KEY and pass a rag_fn via the Python API.
        """
        from rag_eval.backends.base import Document
        from rag_eval.pipeline.dataset import EvalDataset
        from rag_eval.pipeline.evaluator import RAGEvaluator, EvalConfig

        backend = get_backend(req.backend, **req.backend_kwargs)
        docs = [Document(id=d.id, text=d.text, metadata=d.metadata) for d in req.documents]
        backend.add_documents(docs)

        dataset = EvalDataset.from_dicts(req.samples)
        config = EvalConfig(k_values=req.k_values)
        evaluator = RAGEvaluator(backend=backend, config=config)
        report = evaluator.evaluate(dataset, top_k=req.top_k)

        return report.summary()

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)
