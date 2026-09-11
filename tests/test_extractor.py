"""Tests for LLM client with fallback chain."""

from unittest.mock import MagicMock, patch

from drbrain.extractor.llm_client import LLMClient, call_with_fallback


def test_single_model_call():
    """LLMClient calls the OpenAI SDK chat completions with correct kwargs."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
    ]
    llm = LLMClient(models)
    client = MagicMock()
    with patch("drbrain.extractor.llm_client._openai_client", return_value=client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"ok": true}'
        client.chat.completions.create.return_value = mock_resp
        result = llm.call("test prompt")
        assert result == {"ok": True}
        client.chat.completions.create.assert_called_once()


def test_fallback_on_failure():
    """call_with_fallback tries next model on exception."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
        {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "api_key": None,
            "base_url": "http://localhost:11434",
        },
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

    client = MagicMock()
    with patch("drbrain.extractor.llm_client._openai_client", return_value=client):
        client.chat.completions.create.side_effect = side_effect
        result = call_with_fallback("test", models)
        assert result == {"ok": True}
        assert call_count == 2


def test_fallback_all_fail():
    """call_with_fallback returns None when all models fail."""
    models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "sk-1", "base_url": None},
        {"provider": "ollama", "model": "qwen2.5:7b", "api_key": None, "base_url": None},
    ]
    client = MagicMock()
    with patch("drbrain.extractor.llm_client._openai_client", return_value=client):
        client.chat.completions.create.side_effect = Exception("fail")
        result = call_with_fallback("test", models)
        assert result is None


def test_provider_routes_via_base_url():
    """Provider selects the OpenAI-compatible endpoint; wire model is bare."""
    models = [
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-1",
            "base_url": None,
        },
    ]
    client = LLMClient(models)
    with patch("drbrain.extractor.llm_client._openai_client") as factory:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"x": 1}'
        factory.return_value.chat.completions.create.return_value = mock_resp
        client.call("test")
        factory.assert_called_once_with("sk-1", "https://api.deepseek.com/v1")
        call_kwargs = factory.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
