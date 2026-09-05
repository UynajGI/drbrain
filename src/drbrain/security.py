"""Small, dependency-free helpers for keeping secrets out of durable data.

Runtime model/API credentials are intentionally short-lived inputs.  This
module provides the one-way projection used by session and research-ledger
writers when they need to retain enough routing metadata for an audit trail
without retaining a credential itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

REDACTED = "[REDACTED]"

# Match names such as ``api_key``, ``access-token`` and ``client_secret`` while
# avoiding ordinary words such as ``tokenizer``.  The compact-name fallback
# below also covers camelCase spellings used by provider SDKs.
_SENSITIVE_NAME = re.compile(
    r"(?:^|[_-])(?:api[_-]?keys?|auth(?:orization)?|access[_-]?tokens?|"
    r"refresh[_-]?tokens?|tokens?|password|passwd|secrets?|credentials?|"
    r"private[_-]?keys?|client[_-]?secrets?|keys?(?:[_-]?id)?|cookies?|bearer)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_COMPACT_NAMES = frozenset(
    {
        "apikey",
        "apikeys",
        "authorization",
        "authtoken",
        "accesstoken",
        "accesstokens",
        "refreshtoken",
        "refreshtokens",
        "token",
        "tokens",
        "password",
        "passwd",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "privatekey",
        "privatekeys",
        "clientsecret",
        "clientsecrets",
        "cookie",
        "cookies",
        "bearer",
    }
)

# ``token`` is a credential by default, but workflow budgets and provider
# usage reports conventionally use names such as ``max_tokens`` and
# ``prompt_tokens`` for numeric counters.  Preserve those counters so
# redaction cannot silently disable budget enforcement; string/list values
# under the same names are still treated as untrusted secret material.
_NUMERIC_TOKEN_FIELDS = frozenset(
    {
        "tokens",
        "tokencount",
        "tokensin",
        "tokensout",
        "prompttokens",
        "completiontokens",
        "inputtokens",
        "outputtokens",
        "cachedtokens",
        "totaltokens",
        "totaltokencount",
        "maxtokens",
        "maxoutputtokens",
        "maxinputtokens",
    }
)

# These expressions only redact values when a secret-looking name or scheme is
# present.  We deliberately do not attempt to guess arbitrary opaque strings.
# Keep a quoted form separate: a credential can contain spaces, commas, or
# shell punctuation, all of which are valid inside a quoted assignment value.
# The optional separators are intentional: provider SDKs commonly spell the
# same field as ``api_key``, ``api-key`` or camelCase ``apiKey``.  Bare
# ``key``/``token`` terms are bounded below, so words such as ``tokenizer`` do
# not match.
_SENSITIVE_TEXT_LABEL = (
    r"(?:api[-_ ]?(?:keys?|tokens?|secrets?)|"
    r"authorization[-_ ]?(?:tokens?|keys?)|"
    r"auth[-_ ]?(?:tokens?|keys?)|"
    r"access[-_ ]?(?:tokens?|keys?(?:[-_ ]?id)?)|"
    r"refresh[-_ ]?tokens?|"
    r"client[-_ ]?(?:secrets?|tokens?|keys?)|"
    r"bearer[-_ ]?tokens?|"
    r"secret[-_ ]?(?:keys?|tokens?)|"
    r"private[-_ ]?keys?|"
    r"(?:[A-Za-z0-9]+[-_ ])*(?:keys?(?:[-_ ]?id)?|tokens?|secrets?|credentials?)|"
    r"password|passwd|cookies?)"
)

_SENSITIVE_QUOTED_ASSIGNMENT = re.compile(
    rf"(?i)((?<![A-Za-z0-9]){_SENSITIVE_TEXT_LABEL}(?![A-Za-z0-9])\s*[:=]\s*)"
    r"(?P<quote>['\"])(?:\\.|(?!(?P=quote)).)*(?P=quote)"
)
_SENSITIVE_ASSIGNMENT = re.compile(
    rf"(?i)((?<![A-Za-z0-9]){_SENSITIVE_TEXT_LABEL}(?![A-Za-z0-9])\s*[:=]\s*)"
    r"(?:(?:bearer|basic|token)\s+)?(['\"]?)[^'\"\s,;&]+\2"
)
_SENSITIVE_QUERY = re.compile(rf"(?i)([?&]{_SENSITIVE_TEXT_LABEL}=)[^&#\s]+")
_BEARER_VALUE = re.compile(r"(?i)(\b(?:bearer|basic|token)\s+)[A-Za-z0-9._~+/=-]+")
_URL_USERINFO = re.compile(r"(?i)(https?://)(?:[^/@\s]+(?::[^/@\s]*)?@)")
_SENSITIVE_FLAG = re.compile(
    rf"(?i)(--{_SENSITIVE_TEXT_LABEL}(?:=|\s+))"
    r"(?P<quote>['\"]?)(?P<value>[^\s,';&]+)(?P=quote)"
)
# SDKs frequently append a harmless-looking suffix (``apiKeyValue``) or
# vendor prefix (``openaiApiKey``).  Keep this separate from the delimiter
# expression above so ordinary words such as ``tokenizer`` remain untouched.
_CAMEL_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)((?<![A-Za-z0-9])(?:[A-Za-z0-9]+[-_])?"
    r"(?:api|auth|access|refresh|client|openai|secret|private|authorization|bearer)"
    r"(?:[-_]?(?:key|token|secret|password|credential|value))(?:[A-Za-z0-9_-]*)"
    r"\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>[^'\"\s,;&]+)(?P=quote)"
)
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<key>[A-Za-z][A-Za-z0-9_-]{0,80})"
    r"\s*[:=]\s*(?P<quote>['\"]?)(?P<value>[^'\"\s,;&]+)(?P=quote)"
)
_GENERIC_SENSITIVE_FLAG = re.compile(
    r"(?i)(?P<prefix>--)(?P<key>[A-Za-z][A-Za-z0-9_-]{0,80})"
    r"(?P<sep>=|\s+)(?P<quote>['\"]?)(?P<value>[^'\"\s,;&]+)(?P=quote)"
)


def is_sensitive_key(key: object) -> bool:
    """Return whether *key* conventionally names a credential or secret."""
    name = str(key).strip()
    if not name:
        return False
    if _SENSITIVE_NAME.search(name):
        return True
    # The delimiter-oriented expression above intentionally avoids matching
    # ordinary words such as ``tokenizer``.  Expand camelCase boundaries before
    # applying it so common SDK spellings (``apiToken``, ``secretKey``,
    # ``authorizationToken``) receive the same key-aware treatment.
    camel_delimited = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    if camel_delimited != name and _SENSITIVE_NAME.search(camel_delimited):
        return True
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    if compact in _SENSITIVE_COMPACT_NAMES:
        return True
    components = [part for part in re.split(r"[-_]", camel_delimited.lower()) if part]
    return any(part in _SENSITIVE_COMPACT_NAMES for part in components)


def _is_numeric_token_field(key: object, value: object) -> bool:
    """Return whether a token-named mapping value is a safe numeric counter."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    compact = re.sub(r"[^a-z0-9]", "", str(key).strip().lower())
    return compact in _NUMERIC_TOKEN_FIELDS


def redact_sensitive_text(value: str | None) -> str | None:
    """Redact credential values embedded in assignment/URL/scheme text."""
    if value is None:
        return None
    # JSON is a common transport for tool errors and model responses.  Handle
    # quoted keys here as well as in ``redact_sensitive`` so callers that only
    # have a text API cannot accidentally bypass key-aware redaction.
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, (Mapping, list)):
        return json.dumps(redact_sensitive(parsed), ensure_ascii=False, separators=(",", ":"))
    # Scheme-first prevents ``authorization: Bearer secret`` from leaving the
    # value after the key/value expression has consumed only ``Bearer``.
    redacted = _SENSITIVE_QUOTED_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group('quote')}{REDACTED}{match.group('quote')}",
        value,
    )
    redacted = _URL_USERINFO.sub(lambda match: f"{match.group(1)}{REDACTED}@", redacted)
    redacted = _BEARER_VALUE.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    redacted = _SENSITIVE_FLAG.sub(
        lambda match: f"{match.group(1)}{REDACTED}",
        redacted,
    )
    redacted = _CAMEL_SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group('quote')}{REDACTED}{match.group('quote')}",
        redacted,
    )
    redacted = _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)

    def _assignment(match: re.Match[str]) -> str:
        key = match.group("key")
        if not is_sensitive_key(key):
            return match.group(0)
        return match.group(0).replace(match.group("value"), REDACTED, 1)

    redacted = _ASSIGNMENT.sub(_assignment, redacted)
    def _flag(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group("key")):
            return match.group(0)
        return match.group(0).replace(match.group("value"), REDACTED, 1)

    redacted = _GENERIC_SENSITIVE_FLAG.sub(_flag, redacted)
    return _SENSITIVE_QUERY.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)


def safe_error(
    value: Any,
    limit: int = 200,
    *,
    secrets: Sequence[str | None] = (),
) -> str:
    """Return a bounded, redacted representation suitable for CLI/log output.

    Provider exceptions sometimes contain an API key without a conventional
    ``api_key=`` label.  ``secrets`` lets the caller provide the active
    credential for exact replacement before the pattern-based scrubber runs.
    Longer values are replaced first so overlapping credentials cannot expose a
    suffix of the longer value.
    """
    rendered = str(value)
    exact_secrets = sorted(
        {str(secret) for secret in secrets if secret},
        key=len,
        reverse=True,
    )
    for secret in exact_secrets:
        rendered = rendered.replace(secret, REDACTED)
    redacted = redact_sensitive_text(rendered)
    return (redacted or "")[: max(0, int(limit))]


def configured_secret_values(config: Any) -> tuple[str, ...]:
    """Collect concrete credential values from a config mapping.

    This is intentionally a narrow bridge for error boundaries: unresolved
    ``${ENV_VAR}`` placeholders and non-string counters are ignored, while
    nested ``api_key``/``token`` fields and token lists are included.  The
    input object is never mutated.
    """
    values: list[str] = []

    def visit(value: Any, key: object | None = None) -> None:
        if key is not None and is_sensitive_key(key):
            if isinstance(value, str):
                if value and not value.startswith("${"):
                    values.append(value)
                return
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    visit(child_value, child_key)
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for child in value:
                    if isinstance(child, str):
                        if child and not child.startswith("${"):
                            values.append(child)
                    else:
                        visit(child)
                return
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), field.name)
            return
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(config)
    return tuple(dict.fromkeys(values))


def redact_sensitive(value: Any) -> Any:
    """Return a recursively copied value with credential fields redacted.

    Mapping keys are retained for schema/debugging purposes, while their
    values become :data:`REDACTED`.  Sequences are copied into JSON-friendly
    lists.  The input object is never mutated.
    """
    if is_dataclass(value) and not isinstance(value, type):
        # Typed ``Config`` instances are common at CLI boundaries.  Project
        # them through the same key-aware mapping path instead of falling back
        # to ``str(value)``, which would include credential fields verbatim.
        return redact_sensitive({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if is_sensitive_key(key) and not _is_numeric_token_field(key, item)
                else redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        # Tool-call arguments and provider errors are often JSON serialized
        # before they reach a writer.  Parse object/array strings so quoted
        # keys (``{"api_key": "..."}``) receive the same key-aware treatment.
        return redact_sensitive_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_sensitive_text(str(value))


def public_model_configs(models: Any) -> list[dict[str, Any]]:
    """Project model configs without persisting credentials.

    A session can still be resumed with the caller's current model list.  The
    stored projection intentionally omits secret-named fields altogether;
    this means a pre-existing session row cannot accidentally rehydrate an old
    key when a caller does not provide a fresh runtime config.
    """
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes, bytearray)):
        return []
    projected: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        item: dict[str, Any] = {}
        for key, value in model.items():
            if is_sensitive_key(key) and not _is_numeric_token_field(key, value):
                continue
            item[str(key)] = redact_sensitive(value)
        projected.append(item)
    return projected


__all__ = [
    "REDACTED",
    "configured_secret_values",
    "is_sensitive_key",
    "public_model_configs",
    "redact_sensitive",
    "redact_sensitive_text",
    "safe_error",
]
