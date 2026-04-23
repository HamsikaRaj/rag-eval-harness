"""
Pluggable LLM runner with per-call token tracking, cost estimation, retry logic,
and timeout handling.

Supported providers: anthropic (default), openai, groq, ollama.
All providers expose the same interface so they're interchangeable.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from shared.cost.estimator import estimate_cost


@dataclass
class ModelResponse:
    """Structured response from any LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    model: str
    provider: str


@dataclass
class _UsageAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0


class ModelRunner:
    """
    Unified LLM client supporting Anthropic, OpenAI, Groq, and Ollama.

    Compatible with ClaudeLLM's `complete()` / `complete_json()` interface so it
    can be used as a drop-in replacement in generation-metric functions.

    Per-call metrics (latency, tokens, cost) are tracked internally and
    accessible via `get_usage()`.

    Retry behaviour:
      - Up to `max_retries` attempts on API errors or timeouts
      - Exponential backoff: 1 s, 2 s, 4 s, …
      - Raises the final exception after exhausting retries
    """

    _PROVIDER_DEFAULTS: dict[str, str] = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai":    "gpt-4o-mini",
        "groq":      "llama3-70b-8192",
        "ollama":    "llama3",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        failure_logger=None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model or os.getenv(
            f"{self.provider.upper()}_MODEL",
            self._PROVIDER_DEFAULTS.get(self.provider, ""),
        )
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._failure_logger = failure_logger
        self._usage = _UsageAccumulator()
        self._client: Any = None

    # ── Public interface ──────────────────────────────────────────────────────

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the text response."""
        return self._run_with_retry(prompt, system).text

    def complete_json(self, prompt: str, system: str | None = None) -> dict | list:
        """Send a prompt expecting JSON; strips markdown fences and parses."""
        raw = self.complete(prompt, system)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}

    def run(self, prompt: str, system: str | None = None) -> ModelResponse:
        """Full ModelResponse with all per-call metadata."""
        return self._run_with_retry(prompt, system)

    def get_usage(self) -> dict[str, Any]:
        """Cumulative token and cost summary since last reset."""
        return {
            "input_tokens":  self._usage.input_tokens,
            "output_tokens": self._usage.output_tokens,
            "call_count":    self._usage.call_count,
            "estimated_cost_usd": round(
                estimate_cost(self.model, self._usage.input_tokens, self._usage.output_tokens), 6
            ),
        }

    def reset_usage(self) -> None:
        """Reset cumulative token counters."""
        self._usage = _UsageAccumulator()

    @staticmethod
    def is_available(provider: str = "anthropic") -> bool:
        """Check whether the required API key for a provider is set."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "groq":      "GROQ_API_KEY",
            "ollama":    None,  # always available (local)
        }
        env_key = key_map.get(provider.lower())
        return env_key is None or bool(os.getenv(env_key))

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _run_with_retry(self, prompt: str, system: str | None) -> ModelResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._dispatch(prompt, system)
                # Track usage here so mocking _run_* still accumulates tokens
                self._update_usage(resp.input_tokens, resp.output_tokens)
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
        # All retries exhausted
        if self._failure_logger:
            self._failure_logger.log(
                service="model_runner",
                query=prompt[:200],
                issue=f"api_error_after_{self.max_retries}_retries: {last_exc}",
            )
        raise RuntimeError(
            f"ModelRunner: all {self.max_retries} attempts failed. Last error: {last_exc}"
        ) from last_exc

    def _dispatch(self, prompt: str, system: str | None) -> ModelResponse:
        if self.provider == "anthropic":
            return self._run_anthropic(prompt, system)
        if self.provider == "openai":
            return self._run_openai(prompt, system)
        if self.provider == "groq":
            return self._run_groq(prompt, system)
        if self.provider == "ollama":
            return self._run_ollama(prompt, system)
        raise ValueError(f"Unknown provider '{self.provider}'. Choose: anthropic, openai, groq, ollama")

    # ── Provider implementations ──────────────────────────────────────────────

    def _run_anthropic(self, prompt: str, system: str | None) -> ModelResponse:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                timeout=self.timeout_s,
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens

        return ModelResponse(
            text=resp.content[0].text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=estimate_cost(self.model, in_tok, out_tok),
            model=self.model,
            provider="anthropic",
        )

    def _run_openai(self, prompt: str, system: str | None) -> ModelResponse:
        import openai

        if self._client is None:
            self._client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=self.timeout_s,
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        in_tok  = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens

        return ModelResponse(
            text=resp.choices[0].message.content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=estimate_cost(self.model, in_tok, out_tok),
            model=self.model,
            provider="openai",
        )

    def _run_groq(self, prompt: str, system: str | None) -> ModelResponse:
        import groq as groq_sdk

        if self._client is None:
            self._client = groq_sdk.Groq(
                api_key=os.getenv("GROQ_API_KEY"),
                timeout=self.timeout_s,
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        in_tok  = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens

        return ModelResponse(
            text=resp.choices[0].message.content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=estimate_cost(self.model, in_tok, out_tok),
            model=self.model,
            provider="groq",
        )

    def _run_ollama(self, prompt: str, system: str | None) -> ModelResponse:
        import httpx

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        resp = httpx.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        data    = resp.json()
        in_tok  = data.get("prompt_eval_count", 0)
        out_tok = data.get("eval_count", 0)

        return ModelResponse(
            text=data.get("response", ""),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=0.0,  # local model — no cost
            model=self.model,
            provider="ollama",
        )

    def _update_usage(self, in_tok: int, out_tok: int) -> None:
        self._usage.input_tokens  += in_tok
        self._usage.output_tokens += out_tok
        self._usage.call_count    += 1
