"""JSON Schema 2020-12 validation used by all capability adapters."""

from __future__ import annotations

from typing import Any


def validate_schema(schema: Any) -> tuple[str, ...]:
    """Return schema-document errors without importing an adapter or handler."""
    if not isinstance(schema, dict):
        return ("schema must be a JSON object",)
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except ImportError:
        return ("jsonschema is required for capability validation",)
    except Exception as exc:  # jsonschema.SchemaError is intentionally normalized
        return (f"invalid JSON Schema 2020-12 document: {exc}",)
    return ()


def validate_instance(schema: Any, instance: Any) -> tuple[str, ...]:
    """Validate an invocation payload and return concise path-aware errors."""
    schema_errors = validate_schema(schema)
    if schema_errors:
        return schema_errors
    if not schema:
        return ()
    try:
        from jsonschema import Draft202012Validator

        errors = sorted(
            Draft202012Validator(schema).iter_errors(instance),
            key=lambda error: [
                (0, part) if isinstance(part, int) else (1, part) for part in error.absolute_path
            ],
        )
    except ImportError:
        return ("jsonschema is required for capability validation",)
    except Exception as exc:
        return (f"schema validation failed: {exc}",)
    messages: list[str] = []
    for error in errors[:8]:
        path = "".join(
            f"[{part!r}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        messages.append(f"{path.lstrip('.') or '<root>'}: {error.message}")
    return tuple(messages)
