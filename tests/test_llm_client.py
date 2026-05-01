"""Tests for LLM client with logging and metrics tracking."""

from unittest import mock


def test_call_with_fallback_records_metrics():
    """Successful LLM call records metrics."""
    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock()]
    mock_response.choices[0].message.content = '{"result": "ok"}'
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    with (
        mock.patch("drbrain.extractor.llm_client.litellm.completion", return_value=mock_response),
        mock.patch("drbrain.extractor.llm_client._record_llm") as mock_record,
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        )
        assert result == {"result": "ok"}
        mock_record.assert_called_once()


def test_call_with_fallback_tries_next_model_on_failure():
    """When first model fails, second model is tried."""
    mock_fail = mock.Mock(side_effect=Exception("API error"))
    mock_success = mock.Mock()
    mock_success.choices = [mock.Mock()]
    mock_success.choices[0].message.content = '{"ok": true}'

    with (
        mock.patch(
            "drbrain.extractor.llm_client.litellm.completion",
            side_effect=[mock_fail, mock_success],
        ),
        mock.patch("drbrain.extractor.llm_client._record_llm"),
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [
                {"provider": "openai", "model": "broken", "api_key": "x"},
                {"provider": "openai", "model": "working", "api_key": "x"},
            ],
        )
        assert result == {"ok": True}


def test_call_with_fallback_all_fail():
    """When all models fail, returns None."""
    with mock.patch(
        "drbrain.extractor.llm_client.litellm.completion",
        side_effect=Exception("All dead"),
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [{"provider": "openai", "model": "broken", "api_key": "x"}],
        )
        assert result is None
