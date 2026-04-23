"""
RAGAS-style generation quality metrics powered by Claude as the LLM judge.

Metrics
-------
faithfulness        — What fraction of answer claims are supported by the retrieved context?
answer_relevancy    — How well does the answer address the original question?
context_precision   — Are the retrieved chunks actually useful for answering?
context_recall      — Do the retrieved chunks cover the ground-truth answer?

All metrics return floats in [0, 1].  Claude is called once per metric per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_eval.llm.claude import ClaudeLLM


# ── Prompt templates ──────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluation assistant. Given a question, its answer, and retrieved context passages, \
assess the faithfulness of the answer.

Question: {question}

Retrieved context:
{context}

Answer: {answer}

Task:
1. Identify every factual claim made in the answer.
2. For each claim, decide whether it is fully supported by the context (true/false).
3. Compute faithfulness_score = supported_claims / total_claims (1.0 if no claims).

Respond with ONLY valid JSON, exactly like this:
{{
  "claims": [
    {{"text": "<claim>", "supported": true}},
    {{"text": "<claim>", "supported": false}}
  ],
  "faithfulness_score": 0.85
}}"""

_ANSWER_RELEVANCY_PROMPT = """\
You are an evaluation assistant. Rate how relevant and complete the following answer is \
for the given question, on a scale from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Question: {question}
Answer: {answer}

Respond with ONLY valid JSON:
{{
  "score": 0.9,
  "reasoning": "<one sentence>"
}}"""

_CONTEXT_PRECISION_PROMPT = """\
You are an evaluation assistant. For the given question, rate how useful each retrieved context \
chunk is for answering it (0.0 = not useful, 1.0 = highly useful).

Question: {question}

Contexts:
{contexts_numbered}

Respond with ONLY valid JSON:
{{
  "scores": [0.9, 0.3, 0.7],
  "context_precision": 0.63
}}
The "context_precision" field must equal the mean of "scores"."""

_CONTEXT_RECALL_PROMPT = """\
You are an evaluation assistant. Given a ground-truth answer and a set of retrieved context \
passages, estimate what fraction of the information in the ground-truth answer is covered by \
the contexts (0.0 = none, 1.0 = fully covered).

Ground-truth answer: {ground_truth}

Retrieved contexts:
{context}

Respond with ONLY valid JSON:
{{
  "score": 0.8,
  "reasoning": "<one sentence>"
}}"""


# ── Per-metric functions ──────────────────────────────────────────────────────

def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))


def _format_contexts_numbered(contexts: list[str]) -> str:
    return "\n".join(f"{i + 1}. {c[:300]}..." if len(c) > 300 else f"{i + 1}. {c}"
                     for i, c in enumerate(contexts))


def faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    llm: ClaudeLLM,
) -> float:
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question,
        context=_format_contexts(contexts),
        answer=answer,
    )
    data = llm.complete_json(prompt)
    score = data.get("faithfulness_score")
    if score is None:
        claims = data.get("claims", [])
        if not claims:
            return 1.0
        supported = sum(1 for c in claims if c.get("supported"))
        score = supported / len(claims)
    return float(score)


def answer_relevancy(
    question: str,
    answer: str,
    llm: ClaudeLLM,
) -> float:
    prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
    data = llm.complete_json(prompt)
    return float(data.get("score", 0.0))


def context_precision(
    question: str,
    contexts: list[str],
    llm: ClaudeLLM,
) -> float:
    if not contexts:
        return 0.0
    prompt = _CONTEXT_PRECISION_PROMPT.format(
        question=question,
        contexts_numbered=_format_contexts_numbered(contexts),
    )
    data = llm.complete_json(prompt)
    score = data.get("context_precision")
    if score is None:
        scores = data.get("scores", [])
        score = sum(scores) / len(scores) if scores else 0.0
    return float(score)


def context_recall(
    ground_truth: str,
    contexts: list[str],
    llm: ClaudeLLM,
) -> float:
    if not ground_truth:
        return 0.0
    prompt = _CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth,
        context=_format_contexts(contexts),
    )
    data = llm.complete_json(prompt)
    return float(data.get("score", 0.0))


# ── Aggregating class ─────────────────────────────────────────────────────────

@dataclass
class GenerationMetrics:
    """
    Compute RAGAS-style generation quality metrics using Claude as the LLM judge.

    Example::

        gm = GenerationMetrics()
        scores = gm.compute(questions, answers, contexts, ground_truths)
    """

    metrics: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    llm: ClaudeLLM | None = None

    def _get_llm(self) -> ClaudeLLM:
        if self.llm is None:
            self.llm = ClaudeLLM()
        return self.llm

    def compute_sample(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """Score a single question-answer-context triple."""
        llm = self._get_llm()
        result: dict[str, float] = {}

        if "faithfulness" in self.metrics:
            result["faithfulness"] = faithfulness(question, answer, contexts, llm)

        if "answer_relevancy" in self.metrics:
            result["answer_relevancy"] = answer_relevancy(question, answer, llm)

        if "context_precision" in self.metrics:
            result["context_precision"] = context_precision(question, contexts, llm)

        if "context_recall" in self.metrics and ground_truth:
            result["context_recall"] = context_recall(ground_truth, contexts, llm)

        return result

    def compute(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate generation quality over a dataset.

        Returns mean scores across all samples.
        """
        assert len(questions) == len(answers) == len(contexts)
        n = len(questions)
        accumulator: dict[str, list[float]] = {}

        for i in range(n):
            gt = ground_truths[i] if ground_truths else None
            sample_scores = self.compute_sample(
                questions[i], answers[i], contexts[i], gt
            )
            for metric, score in sample_scores.items():
                accumulator.setdefault(metric, []).append(score)

        return {metric: sum(vals) / len(vals) for metric, vals in accumulator.items()}
