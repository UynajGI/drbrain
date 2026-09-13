"""Regression tests for secret-free durable projections."""

from __future__ import annotations

import json

from drbrain.security import (
    REDACTED,
    configured_secret_values,
    public_model_configs,
    redact_sensitive,
    redact_sensitive_text,
    safe_error,
)


def test_public_model_configs_omits_credentials_without_mutating_input():
    models = [
        {
            "provider": "openai",
            "model": "gpt-test",
            "api_key": "live-key",
            "api_keys": ["key-a", "key-b"],
            "base_url": "https://example.test/v1",
            "extra_headers": {"Authorization": "Bearer nested-key"},
        }
    ]

    projected = public_model_configs(models)

    assert projected == [
        {
            "provider": "openai",
            "model": "gpt-test",
            "base_url": "https://example.test/v1",
            "extra_headers": {"Authorization": REDACTED},
        }
    ]
    assert models[0]["api_key"] == "live-key"
    assert "live-key" not in json.dumps(projected)
    assert "nested-key" not in json.dumps(projected)


def test_redaction_handles_json_and_provider_error_forms():
    value = {
        "api_key": "one",
        "nested": [{"private_key": "two"}],
        "message": "Authorization: Bearer three",
        "url": "https://example.test/?access_token=four",
    }
    redacted = redact_sensitive(value)

    assert redacted["api_key"] == REDACTED
    assert redacted["nested"][0]["private_key"] == REDACTED
    assert "three" not in redacted["message"]
    assert "four" not in redacted["url"]

    encoded = '{"api_key":"five","token":"six"}'
    safe_encoded = redact_sensitive(encoded)
    assert "five" not in safe_encoded
    assert "six" not in safe_encoded

    error = redact_sensitive_text("request failed: api_key=seven")
    assert error is not None
    assert "seven" not in error


def test_text_redaction_covers_quoted_values_with_spaces_and_punctuation():
    value = 'request failed: api_key="secret with spaces, & punctuation"'

    safe = redact_sensitive_text(value)

    assert safe is not None
    assert "secret with spaces" not in safe
    assert "punctuation" not in safe


def test_text_redaction_covers_url_userinfo_and_cli_flags():
    safe = redact_sensitive_text(
        "https://user:pw@example.test/v1?token=query-secret --api-key flag-secret"
    )

    assert safe is not None
    assert "user:pw" not in safe
    assert "query-secret" not in safe
    assert "flag-secret" not in safe
    assert safe.count(REDACTED) == 3


def test_numeric_token_counters_are_not_redacted():
    value = {"prompt_tokens": 12, "completion_tokens": 8, "max_tokens": 32}

    assert redact_sensitive(value) == value


def test_camel_case_credential_keys_are_redacted():
    value = {
        "apiToken": "camel-api-secret",
        "clientToken": "camel-client-secret",
        "secretKey": "camel-secret-key",
        "authorizationToken": "camel-auth-secret",
        "tokenizer": "safe-model-component",
    }

    redacted = redact_sensitive(value)

    assert redacted["apiToken"] == REDACTED
    assert redacted["clientToken"] == REDACTED
    assert redacted["secretKey"] == REDACTED
    assert redacted["authorizationToken"] == REDACTED
    assert redacted["tokenizer"] == value["tokenizer"]
    encoded = redact_sensitive_text('{"apiToken":"camel-json-secret"}')
    assert "camel-json-secret" not in encoded


def test_camel_case_credential_key_variants_are_detected():
    from drbrain.security import is_sensitive_key

    assert all(
        is_sensitive_key(name)
        for name in (
            "apiToken",
            "accessToken",
            "clientToken",
            "bearerToken",
            "secretKey",
            "authorizationToken",
            "privateKey",
            "AWS_ACCESS_KEY_ID",
            "openai_key",
        )
    )
    assert not is_sensitive_key("tokenizer")


def test_text_redaction_covers_combined_credential_names():
    samples = (
        "api_token=opaque-one",
        "clientToken=opaque-two",
        "secretKey=opaque-three",
        "authorizationToken=opaque-four",
        "AWS_ACCESS_KEY_ID=opaque-five",
        "openai_key=opaque-six",
        "?api_token=opaque-seven",
        "--apiToken opaque-eight",
    )

    for sample in samples:
        safe = redact_sensitive_text(sample)
        assert safe is not None
        assert "opaque-" not in safe
        assert REDACTED in safe


def test_redact_sensitive_projects_typed_dataclasses():
    from dataclasses import dataclass

    @dataclass
    class ProviderConfig:
        api_token: str
        model: str

    value = redact_sensitive(ProviderConfig("typed-secret", "model-a"))

    assert value == {"api_token": REDACTED, "model": "model-a"}


def test_safe_error_scrubs_explicit_unlabelled_secrets_and_bounds_output():
    secret = "sk-live-provider-key"

    rendered = safe_error(
        RuntimeError(f"provider rejected {secret} after " + "x" * 500),
        limit=80,
        secrets=(secret,),
    )

    assert secret not in rendered
    assert REDACTED in rendered
    assert len(rendered) == 80


def test_configured_secret_values_collects_nested_credentials_without_placeholders():
    cfg = {
        "llm": {
            "models": [
                {"api_key": "sk-one", "api_keys": ["sk-two", "${SECOND_KEY}"]},
            ]
        },
        "mineru": {"token": "mineru-secret"},
        "api": {"deepxiv_token": "${DEEPXIV_TOKEN}", "s2_api_key": "s2-secret"},
    }

    values = configured_secret_values(cfg)

    assert values == ("sk-one", "sk-two", "mineru-secret", "s2-secret")


def test_configured_secret_values_supports_typed_config_objects():
    from drbrain.config import Config, EmbedConfig, LLMConfig

    cfg = Config(
        llm=LLMConfig(models=[{"api_key": "typed-llm-secret"}]),
        embed=EmbedConfig(api_key="typed-embed-secret"),
    )

    assert configured_secret_values(cfg) == (
        "typed-llm-secret",
        "typed-embed-secret",
    )
