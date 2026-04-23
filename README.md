# RAG Evaluation Harness

A production-grade, multi-service RAG evaluation platform. Evaluate retrieval quality, generation quality (via Claude as LLM judge), hallucination risk, system latency, and cost — all with a pluggable, modular architecture.

---

## Architecture

```
rag-eval-harness/
├── rag_eval/                      # Core library (stable, reusable)
│   ├── backends/                  # FAISS · Qdrant · ChromaDB · Weaviate
│   ├── metrics/                   # Retrieval · Generation · Hallucination
│   ├── pipeline/                  # RAGEvaluator + EvalDataset
│   ├── llm/                       # ClaudeLLM wrapper
│   └── api/                       # Base FastAPI app (/index /search /metrics/*)
│
├── shared/                        # Cross-service utilities
│   ├── schemas/                   # Pydantic models (EvalQuery, EvalResult, FailureLog,
│   │                              #   CostReport, RunReport)
│   ├── cost/                      # Token cost estimator (Claude · GPT · Groq · Ollama)
│   └── logging/                   # Structured failure logger → logs/failures_{run_id}.json
│
├── services/
│   ├── model_runner/              # Pluggable LLM: Anthropic · OpenAI · Groq · Ollama
│   │                              # Retry (3×, exp backoff) · timeout · token tracking
│   ├── retrieval_evaluator/       # Recall@K · MRR · NDCG · latency · slow-query flags
│   ├── generation_evaluator/      # RAGAS-style metrics + per-sample cost tracking
│   ├── dataset_manager/           # Load (JSON/CSV/HF) · SHA-256 versioning · train/eval split
│   └── api_gateway/               # Extended FastAPI (full-eval · reports · failures · cost)
│
├── tests/                         # Full test suite (all mocked — no API key required)
├── examples/quickstart.py
└── .env.example
```

---

## Quickstart

### Install

```bash
pip install -e .
# Full stack (all optional backends):
pip install -e ".[all,dev]"
```

### Environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY for generation metrics
```

### Run retrieval evaluation (no API key)

```bash
python examples/quickstart.py
```

### Run full pipeline (requires Claude API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/quickstart.py
```

### Start the API Gateway

```bash
# Uses the full gateway (existing routes + new endpoints)
uvicorn services.api_gateway.app:create_gateway_app --factory --host 0.0.0.0 --port 8080 --reload
```

### CLI

```bash
rag-eval serve                                              # start base API server
rag-eval eval-retrieval examples/sample_dataset.json examples/sample_docs.json
rag-eval check-hallucination examples/answers.json         # requires ANTHROPIC_API_KEY
```

---

## API Endpoints

### Base routes (`rag_eval/api`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + Claude availability |
| POST | `/index` | Embed + index documents into a backend |
| POST | `/search/{session_id}` | Semantic search |
| POST | `/metrics/retrieval` | Recall@K, MRR, NDCG from ID lists |
| POST | `/metrics/hallucination` | Claim-level hallucination detection (requires key) |
| POST | `/evaluate` | Retrieval-only evaluation |

### Gateway routes (`services/api_gateway`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/datasets/load` | Ingest + version a dataset (SHA-256 ID) |
| GET | `/datasets` | List all versioned datasets |
| POST | `/run/full-eval` | Run all evaluation layers → returns `run_id` |
| GET | `/reports/{run_id}` | Full `RunReport` for a completed run |
| GET | `/failures/{run_id}` | Failure log entries (slow queries, hallucinations, etc.) |
| GET | `/cost/{run_id}` | Cost breakdown per service |

### Full eval request example

```bash
curl -X POST http://localhost:8080/run/full-eval \
  -H "Content-Type: application/json" \
  -d '{
    "samples":   [{"question": "What is Paris?", "relevant_doc_ids": ["d1"]}],
    "documents": [{"id": "d1", "text": "Paris is the capital of France."}],
    "backend":   "faiss",
    "top_k":     5,
    "k_values":  [1, 3, 5],
    "run_generation": false
  }'
# → {"run_id": "a3f2c1b0", "status": "completed", "dataset_id": "..."}
```

---

## Metrics

### Retrieval

| Metric | Description |
|--------|-------------|
| `recall@K` | Fraction of relevant docs in top-K results |
| `mrr` | Mean Reciprocal Rank of first relevant doc |
| `ndcg@K` | Normalized Discounted Cumulative Gain at K |
| `latency_ms_mean` | Mean per-query retrieval time |
| `slow_query_count` | Queries exceeding `SLOW_QUERY_THRESHOLD_MS` (default 500ms) |

### Generation (Claude judge)

| Metric | Description |
|--------|-------------|
| `faithfulness` | Fraction of answer claims supported by retrieved context |
| `answer_relevancy` | How well the answer addresses the question |
| `context_precision` | Usefulness of retrieved chunks for answering |
| `context_recall` | Coverage of ground-truth information in context |

### Hallucination

| Metric | Description |
|--------|-------------|
| `grounding_score` | Fraction of claims supported by sources |
| `hallucination_score` | Fraction of claims **not** supported (lower is better) |

---

## Model Runner Providers

```python
from services.model_runner.runner import ModelRunner

runner = ModelRunner(provider="anthropic")   # default
runner = ModelRunner(provider="openai")      # pip install 'rag-eval-harness[openai]'
runner = ModelRunner(provider="groq")        # pip install 'rag-eval-harness[groq]'
runner = ModelRunner(provider="ollama")      # local, no API key needed
```

Features: **3× retry with exponential backoff**, **30s timeout**, **per-call token + cost tracking**.

---

## Cost Estimation

```python
from shared.cost.estimator import estimate_cost

cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=5000, output_tokens=1000)
# → 0.030000 USD
```

Supported models: Claude Sonnet/Haiku/Opus, GPT-4o/4o-mini, Groq Llama3/Mixtral, Ollama (free).

---

## Failure Logging

Every run writes a structured JSON log to `logs/failures_{run_id}.json`:

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

---

## Running Tests

```bash
pytest tests/ -v
```

All 59+ tests run without an API key (LLM calls are mocked).

---

## Backends

| Backend | Install | Notes |
|---------|---------|-------|
| FAISS | included | In-memory, no server |
| Qdrant | `pip install 'rag-eval-harness[qdrant]'` | Local or cloud |
| ChromaDB | `pip install 'rag-eval-harness[chromadb]'` | Ephemeral or persistent |
| Weaviate | `pip install 'rag-eval-harness[weaviate]'` | Requires Weaviate server |
