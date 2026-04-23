"""RAG Evaluation Harness — modular toolkit for retrieval and generation metrics."""

from rag_eval.pipeline.evaluator import RAGEvaluator
from rag_eval.pipeline.dataset import EvalDataset, EvalSample

__all__ = ["RAGEvaluator", "EvalDataset", "EvalSample"]
__version__ = "0.1.0"
