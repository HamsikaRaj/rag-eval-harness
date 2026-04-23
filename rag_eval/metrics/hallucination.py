"""
Citation-source hallucination detection powered by Claude.

Claude decomposes the generated answer into atomic factual claims, then verifies
each claim against the provided source documents.  The result is a grounding score
(fraction of claims that are supported) and a hallucination score (fraction that are not).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_eval.llm.claude import ClaudeLLM


_SYSTEM_PROMPT = (
    "You are a rigorous fact-checking assistant. "
    "Your job is to verify whether claims in an answer are supported by provided source documents. "
    "Always respond with valid JSON only — no extra text."
)

_VERIFY_PROMPT = """\
Given the answer and source documents below, perform the following steps:
1. Break the answer into every distinct atomic factual claim (ignore opinions and filler phrases).
2. For each claim, determine whether it is FULLY supported by at least one source document.
3. Identify which source index (0-based) supports the claim, or null if unsupported.

Answer:
{answer}

Source documents:
{sources}

Respond with ONLY this JSON structure:
{{
  "claims": [
    {{
      "text": "<atomic claim>",
      "supported": true,
      "source_idx": 0
    }},
    {{
      "text": "<another claim>",
      "supported": false,
      "source_idx": null
    }}
  ]
}}"""


@dataclass
class ClaimVerification:
    claim: str
    supported: bool
    source_idx: int | None = None


@dataclass
class HallucinationResult:
    answer: str
    sources: list[str]
    claims: list[ClaimVerification] = field(default_factory=list)

    @property
    def hallucination_score(self) -> float:
        """Fraction of claims NOT grounded in any source (0 = no hallucination)."""
        if not self.claims:
            return 0.0
        unsupported = sum(1 for c in self.claims if not c.supported)
        return unsupported / len(self.claims)

    @property
    def grounding_score(self) -> float:
        """Fraction of claims grounded in at least one source (1 = fully grounded)."""
        return 1.0 - self.hallucination_score

    def to_dict(self) -> dict:
        return {
            "hallucination_score": round(self.hallucination_score, 4),
            "grounding_score": round(self.grounding_score, 4),
            "total_claims": len(self.claims),
            "supported_claims": [c.claim for c in self.claims if c.supported],
            "unsupported_claims": [c.claim for c in self.claims if not c.supported],
        }


class HallucinationDetector:
    """
    Detects hallucinations by verifying each claim in a generated answer against
    its source documents using Claude as the reasoning engine.

    Usage::

        detector = HallucinationDetector()
        result = detector.check(answer, sources)
        print(result.hallucination_score)   # 0.0 → fully grounded
    """

    def __init__(self, llm: ClaudeLLM | None = None) -> None:
        self._llm = llm or ClaudeLLM()

    def check(self, answer: str, sources: list[str]) -> HallucinationResult:
        """Verify a single answer against its source documents."""
        sources_text = "\n\n".join(
            f"[Source {i}]:\n{src}" for i, src in enumerate(sources)
        )
        prompt = _VERIFY_PROMPT.format(answer=answer, sources=sources_text)
        data = self._llm.complete_json(prompt, system=_SYSTEM_PROMPT)

        claims: list[ClaimVerification] = []
        for item in data.get("claims", []):
            if not isinstance(item, dict):
                continue
            claims.append(
                ClaimVerification(
                    claim=item.get("text", ""),
                    supported=bool(item.get("supported", False)),
                    source_idx=item.get("source_idx"),
                )
            )

        return HallucinationResult(answer=answer, sources=sources, claims=claims)

    def check_batch(
        self, answers: list[str], sources_list: list[list[str]]
    ) -> list[HallucinationResult]:
        """Verify multiple answer-source pairs."""
        return [self.check(ans, srcs) for ans, srcs in zip(answers, sources_list)]

    def aggregate(self, results: list[HallucinationResult]) -> dict:
        """Return mean hallucination/grounding scores across a batch."""
        if not results:
            return {
                "hallucination_score": 0.0,
                "grounding_score": 1.0,
                "total_samples": 0,
                "total_claims": 0,
            }
        return {
            "hallucination_score": round(
                sum(r.hallucination_score for r in results) / len(results), 4
            ),
            "grounding_score": round(
                sum(r.grounding_score for r in results) / len(results), 4
            ),
            "total_samples": len(results),
            "total_claims": sum(len(r.claims) for r in results),
        }
