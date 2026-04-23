"""
End-to-end quickstart demonstrating the full multi-service RAG evaluation platform.

Sections:
  1. Retrieval metrics only          — no API key needed
  2. Multi-service pipeline          — no API key needed
  3. Generation + hallucination      — requires ANTHROPIC_API_KEY
  4. Dataset manager + versioning    — no API key needed
  5. Cost estimation                 — no API key needed

Run:
    pip install -e .
    python examples/quickstart.py                   # retrieval + service demo
    ANTHROPIC_API_KEY=sk-ant-... python examples/quickstart.py  # full pipeline
"""

import os
import uuid

from rich.console import Console
from rich.rule import Rule

from rag_eval import RAGEvaluator, EvalDataset
from rag_eval.backends import FAISSBackend
from rag_eval.backends.base import Document
from rag_eval.llm.claude import ClaudeLLM
from rag_eval.pipeline.evaluator import EvalConfig

from shared.cost.estimator import estimate_cost, pricing_table
from shared.logging.logger import FailureLogger
from shared.schemas.models import RunReport, CostReport

from services.model_runner.runner import ModelRunner
from services.retrieval_evaluator.evaluator import RetrievalEvaluatorService
from services.generation_evaluator.evaluator import GenerationEvaluatorService
from services.dataset_manager.manager import DatasetManager

console = Console()

# ── Corpus + eval dataset ─────────────────────────────────────────────────────

DOCUMENTS = [
    Document(id="doc_1", text="Paris is the capital of France. It is known for the Eiffel Tower."),
    Document(id="doc_2", text="Berlin is the capital of Germany. The Berlin Wall divided the city."),
    Document(id="doc_3", text="Tokyo is the capital of Japan. It hosts the cherry blossom festival."),
    Document(id="doc_4", text="Madrid is the capital of Spain. The Prado Museum is there."),
    Document(id="doc_5", text="Rome is the capital of Italy. The Colosseum is a famous landmark."),
    Document(id="doc_6", text="The Eiffel Tower was built in 1889 for the World Fair in Paris."),
    Document(id="doc_7", text="The Colosseum in Rome was completed in 80 AD, seating 80,000."),
    Document(id="doc_8", text="Tokyo's population exceeds 13 million — one of the world's most populous."),
]

SAMPLES = [
    {
        "question":            "What is the capital of France and what is it famous for?",
        "relevant_doc_ids":    ["doc_1", "doc_6"],
        "ground_truth_answer": "Paris is the capital of France, famous for the Eiffel Tower (built 1889).",
    },
    {
        "question":            "Tell me about Rome's famous landmark.",
        "relevant_doc_ids":    ["doc_5", "doc_7"],
        "ground_truth_answer": "The Colosseum in Rome was completed in 80 AD and seats 80,000 spectators.",
    },
    {
        "question":            "What is Tokyo known for?",
        "relevant_doc_ids":    ["doc_3", "doc_8"],
        "ground_truth_answer": "Tokyo is Japan's capital, known for cherry blossoms and over 13M people.",
    },
]

RUN_ID = str(uuid.uuid4())[:8]

# ── 1. Classic retrieval-only evaluation ─────────────────────────────────────

console.print(Rule("[bold cyan]1. Classic RAG Evaluator (Retrieval Only)"))

backend = FAISSBackend()
backend.add_documents(DOCUMENTS)

dataset = EvalDataset.from_dicts(SAMPLES, name="capitals_demo")
config  = EvalConfig(k_values=[1, 3, 5])
evaluator = RAGEvaluator(backend=backend, config=config)
report = evaluator.evaluate(dataset, top_k=5)
report.print()

# ── 2. Multi-service pipeline ─────────────────────────────────────────────────

console.print(Rule("[bold cyan]2. Multi-Service Pipeline"))

logger = FailureLogger(run_id=RUN_ID, log_dir="logs")
ret_service = RetrievalEvaluatorService(
    backend=backend,
    k_values=[1, 3, 5],
    slow_threshold_ms=500,
    failure_logger=logger,
)

queries       = [s["question"]            for s in SAMPLES]
relevant_lists = [s["relevant_doc_ids"]   for s in SAMPLES]
ret_results   = ret_service.evaluate_dataset(RUN_ID, queries, relevant_lists, top_k=5)
agg           = ret_service.aggregate(ret_results)

console.print(f"[green]Retrieval aggregate:[/green] {agg}")
console.print(f"[yellow]Slow queries:[/yellow] {int(agg.get('slow_query_count', 0))}")
console.print(f"[yellow]Zero-relevant queries:[/yellow] {int(agg.get('zero_relevant_count', 0))}")
console.print(f"[yellow]Failures logged:[/yellow] {logger.count}")

# ── 3. Dataset manager — versioning & splitting ───────────────────────────────

console.print(Rule("[bold cyan]3. Dataset Manager — Versioning & Split"))

mgr = DatasetManager(data_dir="data")
dataset_id = mgr.load_inline(SAMPLES, name="capitals_v1", documents=[
    {"id": d.id, "text": d.text} for d in DOCUMENTS
])
console.print(f"Dataset versioned as: [bold]{dataset_id}[/bold]")

train, eval_ = mgr.train_eval_split(dataset_id, train_ratio=0.67)
console.print(f"Train samples: {len(train)}, Eval samples: {len(eval_)}")

mgr.store_result(dataset_id, RUN_ID, agg)
console.print(f"Result stored under dataset_id={dataset_id}, run_id={RUN_ID}")

# ── 4. Cost estimation ────────────────────────────────────────────────────────

console.print(Rule("[bold cyan]4. Cost Estimation (Model Pricing Table)"))

examples = [
    ("claude-sonnet-4-20250514", 5000, 1000),
    ("gpt-4o-mini",              5000, 1000),
    ("llama3-70b-8192",          5000, 1000),
    ("ollama/llama3",            5000, 1000),
]
for model, in_tok, out_tok in examples:
    cost = estimate_cost(model, in_tok, out_tok)
    console.print(f"  {model:40s} → ${cost:.6f}")

# ── 5. Generation + hallucination (requires ANTHROPIC_API_KEY) ───────────────

console.print(Rule("[bold cyan]5. Generation + Hallucination (Claude Judge)"))

if not ClaudeLLM.is_available():
    console.print(
        "[dim]ANTHROPIC_API_KEY not set — skipping generation and hallucination metrics.[/dim]\n"
        "[dim]Set ANTHROPIC_API_KEY in your environment to enable Claude-powered evaluation.[/dim]"
    )
else:
    import anthropic as _anthropic

    _claude_client = _anthropic.Anthropic()
    _model         = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Build answers via Claude
    answers = []
    for sample, ret_res in zip(SAMPLES, ret_results):
        ctx_texts = [d.text for d in DOCUMENTS if d.id in ret_res.retrieved_ids[:3]]
        ctx       = "\n\n".join(ctx_texts)
        response  = _claude_client.messages.create(
            model=_model, max_tokens=256,
            system="Answer using only the provided context. Be concise.",
            messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {sample['question']}"}],
        )
        answers.append(response.content[0].text)

    runner = ModelRunner(provider="anthropic", model=_model)
    gen_service = GenerationEvaluatorService(
        runner=runner,
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        failure_logger=logger,
    )
    contexts_list = [
        [d.text for d in DOCUMENTS if d.id in r.retrieved_ids[:3]]
        for r in ret_results
    ]
    gen_results = gen_service.evaluate_batch(RUN_ID, queries, answers, contexts_list)
    gen_agg     = gen_service.aggregate(gen_results)

    console.print(f"[green]Generation aggregate:[/green] {gen_agg}")

    # Hallucination detection
    from rag_eval.metrics.hallucination import HallucinationDetector
    detector  = HallucinationDetector()
    h_results = detector.check_batch(answers, contexts_list)
    h_agg     = detector.aggregate(h_results)
    console.print(f"[green]Hallucination aggregate:[/green] {h_agg}")

    # Cost report
    cost_report = gen_service.get_cost_report(RUN_ID, gen_results)
    console.print(f"[green]Cost report:[/green] {cost_report}")
