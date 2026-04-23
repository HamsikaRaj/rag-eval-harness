"""CLI entrypoint for the RAG evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(name="rag-eval", help="RAG Evaluation Harness CLI")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev mode)"),
) -> None:
    """Start the FastAPI evaluation server."""
    import uvicorn

    uvicorn.run(
        "rag_eval.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def eval_retrieval(
    dataset: Path = typer.Argument(..., help="Path to JSON eval dataset"),
    docs: Path = typer.Argument(..., help="Path to JSON documents file"),
    backend: str = typer.Option("faiss", help="Vector store backend"),
    top_k: int = typer.Option(10, help="Number of docs to retrieve"),
    k_values: str = typer.Option("1,3,5,10", help="Comma-separated K values"),
    output: Optional[Path] = typer.Option(None, help="Save report to JSON file"),
) -> None:
    """Run retrieval-only evaluation from the command line."""
    from rag_eval.backends import get_backend
    from rag_eval.backends.base import Document
    from rag_eval.pipeline.dataset import EvalDataset
    from rag_eval.pipeline.evaluator import RAGEvaluator, EvalConfig

    ks = [int(k.strip()) for k in k_values.split(",")]

    raw_docs = json.loads(docs.read_text())
    documents = [
        Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
        for d in raw_docs
    ]

    store = get_backend(backend)
    console.print(f"[cyan]Indexing {len(documents)} documents into {backend}…[/cyan]")
    store.add_documents(documents)

    ev_dataset = EvalDataset.from_json(dataset)
    config = EvalConfig(k_values=ks)
    evaluator = RAGEvaluator(backend=store, config=config)

    console.print(f"[cyan]Evaluating {len(ev_dataset)} queries…[/cyan]")
    report = evaluator.evaluate(ev_dataset, top_k=top_k)
    report.print()

    if output:
        output.write_text(json.dumps(report.summary(), indent=2))
        console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def check_hallucination(
    answers_file: Path = typer.Argument(
        ..., help="JSON file: [{\"answer\": ..., \"sources\": [...]}]"
    ),
    output: Optional[Path] = typer.Option(None, help="Save results to JSON"),
) -> None:
    """Run citation-source hallucination detection on a batch of answers (requires ANTHROPIC_API_KEY)."""
    from rag_eval.llm.claude import ClaudeLLM
    from rag_eval.metrics.hallucination import HallucinationDetector

    if not ClaudeLLM.is_available():
        console.print(
            "[red]ANTHROPIC_API_KEY is not set. "
            "Set it in your environment to use hallucination detection.[/red]"
        )
        raise typer.Exit(1)

    data = json.loads(answers_file.read_text())
    answers = [d["answer"] for d in data]
    sources_list = [d["sources"] for d in data]

    detector = HallucinationDetector()
    console.print("[cyan]Running hallucination detection via Claude…[/cyan]")
    results = detector.check_batch(answers, sources_list)
    aggregate = detector.aggregate(results)

    console.print_json(json.dumps(aggregate))

    if output:
        full = {"aggregate": aggregate, "per_sample": [r.to_dict() for r in results]}
        output.write_text(json.dumps(full, indent=2))
        console.print(f"[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    app()
