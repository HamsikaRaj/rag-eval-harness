"""
Token-based cost estimator with a per-model pricing table.

All prices are USD per 1 000 000 tokens (as of 2025 — update as pricing changes).
"""

from __future__ import annotations

# Pricing: {model_key: (input_price_per_1M, output_price_per_1M)}
_PRICING: dict[str, tuple[float, float]] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-5":        (3.00, 15.00),
    "claude-sonnet-4-6":        (3.00, 15.00),
    "claude-opus-4-7":          (15.00, 75.00),
    "claude-haiku-4-5":         (0.25, 1.25),
    "claude-haiku-4-5-20251001":(0.25, 1.25),
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o":                   (2.50, 10.00),
    "gpt-4o-mini":              (0.15, 0.60),
    "gpt-4-turbo":              (10.00, 30.00),
    # ── Groq ─────────────────────────────────────────────────────────────────
    "llama3-70b-8192":          (0.59, 0.79),
    "llama3-8b-8192":           (0.05, 0.08),
    "mixtral-8x7b-32768":       (0.27, 0.27),
    "gemma2-9b-it":             (0.20, 0.20),
    # ── Ollama (local — free) ─────────────────────────────────────────────────
    "ollama":                   (0.00, 0.00),
}

_UNKNOWN_PRICE = (0.0, 0.0)


def get_pricing(model: str) -> tuple[float, float]:
    """
    Return (input_price_per_1M, output_price_per_1M) for a model.

    Tries exact match first, then prefix match (e.g. "claude-haiku" matches
    any key starting with that string), then falls back to (0, 0).
    """
    model_lower = model.lower()
    if model_lower in _PRICING:
        return _PRICING[model_lower]
    # prefix match — handles versioned names like "claude-sonnet-4-20250514-preview"
    for key, prices in _PRICING.items():
        if model_lower.startswith(key) or key.startswith(model_lower):
            return prices
    # provider-level fallback for ollama (any model name)
    if "ollama" in model_lower:
        return (0.00, 0.00)
    return _UNKNOWN_PRICE


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD for a single LLM call.

    Args:
        model:         Model name / identifier string.
        input_tokens:  Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Estimated cost in USD (float, rounded to 6 decimal places).
    """
    in_price, out_price = get_pricing(model)
    cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return round(cost, 6)


def pricing_table() -> dict[str, dict[str, float]]:
    """Return the full pricing table as a human-readable dict."""
    return {
        model: {"input_per_1m": inp, "output_per_1m": out}
        for model, (inp, out) in _PRICING.items()
    }
