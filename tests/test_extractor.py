"""Tests for LLM client with fallback chain."""
from brbrain.extractor.llm_client import LLMClient, call_with_fallback
from unittest.mock import patch, MagicMock

def test_single_model_call():
    """LLMClient calls litellm with correct kwargs."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
    ]
    client = LLMClient(models)
    with patch("brbrain.extractor.llm_client.litellm") as mock_litellm:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"ok": true}'
        mock_litellm.completion.return_value = mock_resp
        result = client.call("test prompt")
        assert result == {"ok": True}
        mock_litellm.completion.assert_called_once()

def test_fallback_on_failure():
    """call_with_fallback tries next model on exception."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
        {"provider": "ollama", "model": "qwen2.5:7b", "api_key": None, "base_url": "http://localhost:11434"},
    ]
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("API error")
        resp = MagicMock()
        resp.choices[0].message.content = '{"ok": true}'
        return resp

    with patch("brbrain.extractor.llm_client.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = side_effect
        result = call_with_fallback("test", models)
        assert result == {"ok": True}
        assert call_count == 2

def test_fallback_all_fail():
    """call_with_fallback returns None when all models fail."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
        {"provider": "ollama", "model": "qwen2.5:7b", "api_key": None, "base_url": None},
    ]
    with patch("brbrain.extractor.llm_client.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = Exception("fail")
        result = call_with_fallback("test", models)
        assert result is None

def test_model_name_formatting():
    """Provider/model are joined as 'provider/model' for litellm."""
    models = [
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "sk-1", "base_url": None},
    ]
    client = LLMClient(models)
    with patch("brbrain.extractor.llm_client.litellm") as mock_litellm:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"x": 1}'
        mock_litellm.completion.return_value = mock_resp
        client.call("test")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
