"""Tests for the ModelRunner (mocked at the _run_anthropic level — no real API calls)."""

import json
import time
from unittest.mock import MagicMock, patch
import pytest

from services.model_runner.runner import ModelRunner, ModelResponse


def _make_response(text: str = "Answer", in_tok: int = 100, out_tok: int = 50) -> ModelResponse:
    return ModelResponse(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=10.0,
        estimated_cost_usd=0.001,
        model="claude-test",
        provider="anthropic",
    )


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return ModelRunner(provider="anthropic", model="claude-test")


# ── complete / run ────────────────────────────────────────────────────────────

def test_complete_returns_text(runner):
    with patch.object(ModelRunner, "_run_anthropic", return_value=_make_response("Hello!")):
        result = runner.complete("Say hello")
    assert result == "Hello!"


def test_run_returns_model_response(runner):
    with patch.object(ModelRunner, "_run_anthropic",
                      return_value=_make_response("Answer", in_tok=200, out_tok=80)):
        resp = runner.run("Prompt")
    assert isinstance(resp, ModelResponse)
    assert resp.input_tokens  == 200
    assert resp.output_tokens == 80
    assert resp.provider      == "anthropic"


# ── usage accumulation ────────────────────────────────────────────────────────

def test_usage_accumulates(runner):
    with patch.object(ModelRunner, "_run_anthropic",
                      return_value=_make_response("A", in_tok=100, out_tok=50)):
        runner.run("P1")
        runner.run("P2")

    usage = runner.get_usage()
    assert usage["input_tokens"]  == 200
    assert usage["output_tokens"] == 100
    assert usage["call_count"]    == 2


def test_reset_usage(runner):
    with patch.object(ModelRunner, "_run_anthropic", return_value=_make_response("A")):
        runner.run("P")
    runner.reset_usage()
    usage = runner.get_usage()
    assert usage["input_tokens"] == 0
    assert usage["call_count"]   == 0


# ── complete_json ─────────────────────────────────────────────────────────────

def test_complete_json_parses_response(runner):
    payload = {"score": 0.9}
    with patch.object(ModelRunner, "_run_anthropic",
                      return_value=_make_response(json.dumps(payload))):
        result = runner.complete_json("Rate this")
    assert result["score"] == 0.9


def test_complete_json_strips_markdown_fence(runner):
    payload = {"score": 0.75}
    raw = f"```json\n{json.dumps(payload)}\n```"
    with patch.object(ModelRunner, "_run_anthropic", return_value=_make_response(raw)):
        result = runner.complete_json("Rate this")
    assert result["score"] == 0.75


def test_complete_json_returns_empty_on_no_json(runner):
    with patch.object(ModelRunner, "_run_anthropic",
                      return_value=_make_response("No JSON here.")):
        result = runner.complete_json("Bad prompt")
    assert result == {}


# ── retry logic ───────────────────────────────────────────────────────────────

def test_retry_on_api_error(runner, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    call_count = 0

    def flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Transient API error")
        return _make_response("Success")

    with patch.object(ModelRunner, "_run_anthropic", side_effect=flaky):
        result = runner.complete("Prompt")

    assert result == "Success"
    assert call_count == 3


def test_raises_after_max_retries(runner, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with patch.object(ModelRunner, "_run_anthropic",
                      side_effect=RuntimeError("Always fails")):
        with pytest.raises(RuntimeError, match="all 3 attempts failed"):
            runner.complete("Prompt")


def test_failure_logger_called_on_exhausted_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    mock_logger = MagicMock()
    r = ModelRunner(provider="anthropic", model="claude-test", failure_logger=mock_logger)

    with patch.object(ModelRunner, "_run_anthropic",
                      side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError):
            r.complete("P")

    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args[1]
    assert "api_error" in call_kwargs["issue"]


# ── is_available ──────────────────────────────────────────────────────────────

def test_is_available_anthropic_true(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    assert ModelRunner.is_available("anthropic") is True


def test_is_available_anthropic_false(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ModelRunner.is_available("anthropic") is False


def test_is_available_ollama_always_true():
    assert ModelRunner.is_available("ollama") is True


# ── unknown provider ──────────────────────────────────────────────────────────

def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    r = ModelRunner(provider="badprovider", model="x", max_retries=1)
    with pytest.raises(RuntimeError, match="all 1 attempts failed"):
        r.complete("Hello")


# ── cost in get_usage ─────────────────────────────────────────────────────────

def test_cost_included_in_get_usage(runner):
    with patch.object(ModelRunner, "_run_anthropic",
                      return_value=_make_response("A", in_tok=1000, out_tok=500)):
        runner.run("P")
    usage = runner.get_usage()
    assert isinstance(usage["estimated_cost_usd"], float)
    assert usage["estimated_cost_usd"] >= 0.0
