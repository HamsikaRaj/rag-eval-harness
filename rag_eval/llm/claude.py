"""Thin Claude API wrapper used by generation metrics and hallucination detection."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=False), override=True)


_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 1024


class ClaudeLLM:
    """
    Lightweight wrapper around the Anthropic Messages API.

    Keeps the rest of the codebase free from SDK details and makes it easy
    to mock in tests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a single-turn prompt, return the text response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._get_client().messages.create(**kwargs)
        return response.content[0].text

    def complete_json(self, prompt: str, system: str | None = None) -> dict | list:
        """
        Send a prompt that expects a JSON response.

        Extracts the first JSON object or array from the response text,
        tolerating markdown code fences.
        """
        raw = self.complete(prompt, system)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        # Find first {...} or [...] block
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group())

    @staticmethod
    def is_available() -> bool:
        """Return True if ANTHROPIC_API_KEY is present in the environment."""
        return bool(os.getenv("ANTHROPIC_API_KEY"))
