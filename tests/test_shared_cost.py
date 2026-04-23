"""Tests for the cost estimator."""

import pytest
from shared.cost.estimator import estimate_cost, get_pricing, pricing_table


def test_anthropic_sonnet_pricing():
    in_p, out_p = get_pricing("claude-sonnet-4-20250514")
    assert in_p == 3.00
    assert out_p == 15.00


def test_gpt4o_mini_pricing():
    in_p, out_p = get_pricing("gpt-4o-mini")
    assert in_p == 0.15
    assert out_p == 0.60


def test_groq_llama_pricing():
    in_p, out_p = get_pricing("llama3-70b-8192")
    assert in_p == 0.59


def test_ollama_free():
    cost = estimate_cost("ollama/llama3", 10000, 5000)
    assert cost == 0.0


def test_estimate_cost_anthropic():
    # 1M input tokens at $3/1M = $3.00, 0 output = $3.00
    cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 0)
    assert cost == pytest.approx(3.0, abs=1e-4)


def test_estimate_cost_zero_tokens():
    assert estimate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_estimate_cost_small_call():
    # 100 input, 50 output with gpt-4o-mini
    # = (100 * 0.15 + 50 * 0.60) / 1_000_000 = (15 + 30) / 1_000_000
    cost = estimate_cost("gpt-4o-mini", 100, 50)
    assert cost == pytest.approx(45 / 1_000_000, rel=1e-3)


def test_unknown_model_returns_zero():
    cost = estimate_cost("unknown-model-xyz", 1000, 500)
    assert cost == 0.0


def test_prefix_match():
    # "claude-haiku-4-5-extra-suffix" should match "claude-haiku-4-5"
    in_p, out_p = get_pricing("claude-haiku-4-5-20251001")
    assert in_p == 0.25


def test_pricing_table_keys():
    table = pricing_table()
    assert "claude-sonnet-4-20250514" in table
    assert "gpt-4o-mini" in table
    assert "llama3-70b-8192" in table
