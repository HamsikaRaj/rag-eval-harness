# rag-eval-harness

Evaluating RAG pipelines properly is harder than it looks. Retrieval metrics tell you if the right documents came back, but not whether the generated answer was actually faithful to them, or whether the model hallucinated facts that weren't in the context at all. This project handles all of it in one place.

It covers the full evaluation loop — retrieval quality, generation quality using Claude as an automated judge, hallucination detection at the claim level, latency tracking, and cost estimation per run. Works as a Python library, a REST API, or from the CLI.

---

## What gets evaluated

**Retrieval**
- Recall@K, MRR, NDCG@K against ground-truth document annotations
- Per-query latency with slow query flagging

**Generation** (requires Anthropic API key)
- Faithfulness — are the answer's claims backed by retrieved context?
- Answer relevancy — does the answer actually address the question?
- Context precision and recall
- Scored by Claude via structured prompts, no fine-tuned model needed

**Hallucination**
- Decomposes answers into atomic claims
- Verifies each claim against source documents individually
- Returns a grounding score and a list of unsupported claims per answer

---

## Setup

```bash
git clone https://github.com/HamsikaRaj/rag-eval-harness.git
cd rag-eval-harness
pip install -e .

# optional backends
pip install -e ".[all]"   # Qdrant + ChromaDB + Weaviate + OpenAI + Groq
pip install -e ".[dev]"   # adds pytest + ruff
```

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env for generation metrics
```

Retrieval evaluation works without any API key.

---

## Quickstart

```bash
python examples/quickstart.py
```

Runs through retrieval eval, dataset versioning, cost estimation, and a full generation + hallucination eval if a key is set.

---

## Running as an API

```bash
uvicorn services.api_gateway.app:create_gateway_app --factory --port 8080 --reload
```

**Base routes**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + Claude availability |
| POST | `/index` | Embed and index documents |
| POST | `/search/{session_id}` | Semantic search over an indexed session |
| POST | `/metrics/retrieval` | Compute Recall@K, MRR, NDCG from ID lists |
| POST | `/metrics/hallucination` | Claim-level hallucination detection |
| POST | `/evaluate` | Retrieval-only evaluation |

**Gateway routes**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/datasets/load` | Ingest and version a dataset |
| GET | `/datasets` | List versioned datasets |
| POST | `/run/full-eval` | Run full pipeline, returns `run_id` |
| GET | `/reports/{run_id}` | Full results for a completed run |
| GET | `/failures/{run_id}` | Failure log entries for a run |
| GET | `/cost/{run_id}` | Token usage and USD cost breakdown |

**Example request**

```bash
curl -X POST http://localhost:8080/run/full-eval \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{"question": "What is Paris?", "relevant_doc_ids": ["d1"]}],
    "documents": [{"id": "d1", "text": "Paris is the capital of France."}],
    "backend": "faiss",
    "top_k": 5,
    "run_generation": false
  }'
```

---

## CLI

```bash
rag-eval serve                                   # start the base API server
rag-eval eval-retrieval dataset.json docs.json   # retrieval eval from files
rag-eval check-hallucination answers.json        # requires ANTHROPIC_API_KEY
```

---

## Vector store backends

FAISS is the default and runs in-memory with no server required. Alternatives:

| Backend | Install |
|---------|---------|
| FAISS | included |
| Qdrant | `pip install 'rag-eval-harness[qdrant]'` |
| ChromaDB | `pip install 'rag-eval-harness[chromadb]'` |
| Weaviate | `pip install 'rag-eval-harness[weaviate]'` |

Swapping backends is a one-line change — the evaluation logic is completely decoupled.

---

## LLM providers

```python
from services.model_runner.runner import ModelRunner

runner = ModelRunner(provider="anthropic")  # default
runner = ModelRunner(provider="openai")     # pip install 'rag-eval-harness[openai]'
runner = ModelRunner(provider="groq")       # pip install 'rag-eval-harness[groq]'
runner = ModelRunner(provider="ollama")     # local, no API key needed
```

All providers include 3x retry with exponential backoff, 30s timeout, and per-call token + cost tracking.

---

## Cost estimation

```python
from shared.cost.estimator import estimate_cost

estimate_cost("claude-sonnet-4-20250514", input_tokens=5000, output_tokens=1000)
# → 0.03 USD
```

Covers Claude (Sonnet/Haiku/Opus), GPT-4o/4o-mini, Groq Llama3/Mixtral, and Ollama (free).

---

## Failure logging

Every run writes structured JSON to `logs/failures_{run_id}.json`:

```json
{
  "run_id": "abc123",
  "timestamp": "2025-04-22T14:00:00",
  "service": "retrieval_evaluator",
  "query": "What is RBAC?",
  "issue": "zero_relevant_docs_retrieved",
  "retrieved": ["doc_5", "doc_12"],
  "expected": ["doc_3"],
  "latency_ms": 340
}
```

Captures slow queries, zero-relevant retrievals, and faithfulness failures — surfaced per run rather than buried in aggregate numbers.

---

## Tests

```bash
pytest tests/ -v
```

147 tests, all passing. LLM calls are mocked so no API key is needed to run the suite.

---

## Project structure

```
rag_eval/
  backends/      FAISS, Qdrant, ChromaDB, Weaviate
  metrics/       retrieval, generation, hallucination
  pipeline/      RAGEvaluator + EvalDataset
  llm/           Claude wrapper
  api/           base FastAPI app

services/
  model_runner/          LLM abstraction (Anthropic, OpenAI, Groq, Ollama)
  retrieval_evaluator/   per-query retrieval scoring and latency tracking
  generation_evaluator/  RAGAS-style metrics with cost tracking
  dataset_manager/       load, version, and split datasets
  api_gateway/           full orchestration API

shared/
  schemas/    Pydantic models shared across services
  cost/       token cost estimator
  logging/    structured failure logger
```
