"""Tests for ClaudeLLM wrapper (no real API calls — uses mocking)."""

import json
import pytest
from unittest.mock import MagicMock, patch
from rag_eval.llm.claude import ClaudeLLM


def _make_response(text: str):
    content = MagicMock()
    content.text = text
    response = MagicMock()
    response.content = [content]
    return response


@pytest.fixture
def llm():
    return ClaudeLLM(api_key="sk-ant-test", model="claude-test")


def test_complete(llm):
    with patch.object(llm, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _make_response("Hello!")
        result = llm.complete("Say hello")
    assert result == "Hello!"


def test_complete_json_plain(llm):
    payload = {"score": 0.9, "reasoning": "good answer"}
    with patch.object(llm, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _make_response(
            json.dumps(payload)
        )
        result = llm.complete_json("Rate this answer")
    assert result["score"] == 0.9


def test_complete_json_with_markdown_fence(llm):
    payload = {"faithfulness_score": 0.75}
    raw = f"```json\n{json.dumps(payload)}\n```"
    with patch.object(llm, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _make_response(raw)
        result = llm.complete_json("Check faithfulness")
    assert result["faithfulness_score"] == 0.75


def test_complete_json_returns_empty_dict_on_no_json(llm):
    with patch.object(llm, "_get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = _make_response("No JSON here.")
        result = llm.complete_json("Bad prompt")
    assert result == {}


def test_is_available_true(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    assert ClaudeLLM.is_available() is True


def test_is_available_false(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ClaudeLLM.is_available() is False
