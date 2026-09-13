"""Tests for logging setup."""

import logging
import re

import pytest
from loguru import logger

from drbrain.log import LOG_FORMAT, STDERR_FORMAT, get_logger, get_session_id, setup_logging, ui

# ── Existing tests (updated for configurable log_path) ──


def test_setup_logging_creates_log_dir(tmp_path):
    """setup_logging creates log directory and file."""
    log_path = tmp_path / "logs" / "drbrain.log"

    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()

    setup_logging(log_path=str(log_path))
    logger.complete()  # flush pending writes

    assert log_path.parent.exists()
    assert log_path.exists()

    logger.remove()
    log_mod._initialized = False


def test_setup_logging_idempotent():
    """setup_logging is idempotent — second call does nothing."""
    import drbrain.log as log_mod

    log_mod._initialized = True
    setup_logging()  # Should be a no-op
    assert log_mod._initialized  # Still True


def test_setup_logging_writes_to_file(tmp_path):
    """Log messages go to the log file."""
    log_path = tmp_path / "logs" / "test.log"

    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()

    setup_logging(level="INFO", log_path=str(log_path))
    logger.info("Test log message")
    logger.complete()  # flush

    content = log_path.read_text()
    assert "Test log message" in content

    logger.remove()
    log_mod._initialized = False


def test_setup_logging_redacts_sink_messages(tmp_path):
    log_path = tmp_path / "logs" / "safe.log"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path)
    logger.info("Authorization: Bearer durable-log-secret")
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert "durable-log-secret" not in content
    assert "[REDACTED]" in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_redacts_unlabelled_configured_secret(tmp_path):
    """Opaque provider keys are scrubbed even when an error omits a field name."""
    log_path = tmp_path / "logs" / "opaque-secret.log"
    secret = "sk-opaque-provider-secret"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path, secrets=(secret,))
    logger.info("provider rejected credential {}", secret)
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_redacts_short_unlabelled_configured_secret(tmp_path):
    """Short local/test credentials must not bypass the exact scrubber."""
    log_path = tmp_path / "logs" / "short-secret.log"
    secret = "sk-one"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path, secrets=(secret,))
    logger.info("provider rejected credential {}", secret)
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_same_path_keeps_configured_secrets(tmp_path):
    """A repeated setup on one sink must not disable an earlier redaction rule."""
    log_path = tmp_path / "logs" / "reused.log"
    secret = "sk-reused-provider-secret"
    import drbrain.log as log_mod

    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()
    logger.remove()
    setup_logging(level="INFO", log_path=log_path, secrets=(secret,))
    # This mirrors a second embedded invocation whose config did not expose
    # the provider key; the existing sink must remain protected.
    setup_logging(level="INFO", log_path=log_path, secrets=())
    logger.info("reused credential {}", secret)
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content
    logger.remove()
    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()


def test_setup_logging_redacts_literal_sentinel_secret(tmp_path):
    log_path = tmp_path / "logs" / "sentinel-secret.log"
    import drbrain.log as log_mod

    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()
    logger.remove()
    setup_logging(level="INFO", log_path=log_path, secrets=("none",))
    logger.info("configured value none")
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert "value [REDACTED]" in content
    assert "value none" not in content
    logger.remove()
    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()


def test_setup_logging_redacts_exception_trace(tmp_path):
    log_path = tmp_path / "logs" / "safe-exception.log"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path)
    try:
        raise RuntimeError("api_key=durable-exception-secret")
    except RuntimeError:
        logger.exception("request failed")
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert "durable-exception-secret" not in content
    assert "RuntimeError: api_key=[REDACTED]" in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_drops_unclassified_exception_values(tmp_path):
    log_path = tmp_path / "logs" / "opaque-exception.log"
    opaque = "opaque-http-secret"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path)
    try:
        raise RuntimeError(opaque)
    except RuntimeError:
        logger.exception("request failed")
    logger.complete()

    content = log_path.read_text(encoding="utf-8")
    assert opaque not in content
    assert "RuntimeError: [REDACTED]" in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_handles_exception_without_active_error(capsys, tmp_path):
    """A logger.exception call outside ``except`` must stay a safe boundary."""
    log_path = tmp_path / "logs" / "inactive-exception.log"
    marker = "opaque-inactive-secret"
    import drbrain.log as log_mod

    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()
    logger.remove()
    setup_logging(level="INFO", log_path=log_path)
    logger.exception("request failed with %s", marker)
    logger.complete()

    captured = capsys.readouterr()
    content = log_path.read_text(encoding="utf-8")
    assert marker not in captured.err
    assert marker not in content
    assert "Exception: [REDACTED]" in content
    logger.remove()
    log_mod._initialized = False
    log_mod._configured_log_path = None
    log_mod._configured_secrets = ()


def test_setup_logging_failure_does_not_poison_retry(tmp_path):
    import drbrain.log as log_mod

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    log_mod._initialized = False
    log_mod._configured_log_path = None
    logger.remove()

    with pytest.raises(OSError):
        setup_logging(log_path=blocked_parent / "drbrain.log")

    assert log_mod._initialized is False
    assert log_mod._configured_log_path is None

    valid_path = tmp_path / "logs" / "retry.log"
    setup_logging(log_path=valid_path)
    logger.info("retry succeeded")
    logger.complete()
    assert "retry succeeded" in valid_path.read_text(encoding="utf-8")
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_redacts_stdlib_logging(capsys, tmp_path):
    """Third-party stdlib loggers must use the same redaction boundary."""
    log_path = tmp_path / "logs" / "stdlib.log"
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    setup_logging(level="INFO", log_path=log_path)
    logging.getLogger("third_party").warning("provider failed api_key=stdlib-secret")

    captured = capsys.readouterr()
    logger.complete()
    content = log_path.read_text(encoding="utf-8")
    assert "stdlib-secret" not in captured.err
    assert "stdlib-secret" not in content
    logger.remove()
    log_mod._initialized = False


def test_setup_logging_rejects_external_path_under_runtime_root(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    with pytest.raises(ValueError, match="escapes runtime root"):
        setup_logging(log_path=tmp_path / "outside.log")
    logger.remove()


@pytest.mark.parametrize("root_value", ["", "/definitely/not/a/runtime"])
def test_setup_logging_rejects_invalid_runtime_selector(tmp_path, monkeypatch, root_value):
    """Logging must not silently fall back to CWD or a legacy root."""
    import drbrain.log as log_mod

    log_mod._initialized = False
    logger.remove()
    monkeypatch.setenv("DRBRAIN_ROOT", root_value)
    monkeypatch.setenv("DRBRAIN_RUNTIME_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="Runtime root|DRBRAIN_ROOT"):
        setup_logging(log_path=tmp_path / "logs" / "drbrain.log")

    logger.remove()
    log_mod._initialized = False


def test_get_logger_returns_bound_logger():
    """get_logger with name returns a bound logger."""
    named = get_logger("test.module")
    assert named is not None


def test_log_format_has_expected_fields():
    """LOG_FORMAT includes time, level, name, message."""
    assert "time" in LOG_FORMAT
    assert "level" in LOG_FORMAT
    assert "name" in LOG_FORMAT
    assert "message" in LOG_FORMAT


def test_stderr_format_uses_color():
    """STDERR_FORMAT includes level coloring tags."""
    assert "<level>" in STDERR_FORMAT
    assert "</level>" in STDERR_FORMAT


# ── New tests for session_id and ui() ──


def test_setup_logging_logs_session_id(tmp_path):
    """setup_logging writes session start message with session_id to log."""
    log_path = tmp_path / "logs" / "drbrain.log"

    import drbrain.log as log_mod

    log_mod._initialized = False
    log_mod._session_id = None
    logger.remove()

    setup_logging(log_path=str(log_path))
    logger.complete()

    content = log_path.read_text()
    assert "Session started:" in content
    assert log_mod.get_session_id() in content

    logger.remove()
    log_mod._initialized = False


def test_session_id_is_stable():
    """get_session_id returns the same value on repeated calls."""
    import drbrain.log as log_mod

    log_mod._session_id = None  # reset for test isolation

    sid1 = get_session_id()
    sid2 = get_session_id()
    assert sid1 == sid2
    assert sid1 is not None


def test_session_id_is_uuid4_format():
    """get_session_id returns a valid UUID4 string."""
    import drbrain.log as log_mod

    log_mod._session_id = None  # reset for test isolation

    sid = get_session_id()
    # UUID4: version=4 (position 13), variant=8/9/a/b (position 17)
    uuid4_pattern = (
        r"^[0-9a-f]{8}-"
        r"[0-9a-f]{4}-"
        r"4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-"
        r"[0-9a-f]{12}$"
    )
    assert re.match(uuid4_pattern, sid), f"Not a UUID4: {sid}"


def test_ui_writes_to_stdout(capsys):
    """ui() writes message to stdout."""
    ui("Hello, world!")
    captured = capsys.readouterr()
    assert "Hello, world!" in captured.out


def test_ui_redacts_sensitive_text(capsys):
    ui("request failed api_key=ui-secret")
    captured = capsys.readouterr()
    assert "ui-secret" not in captured.out


def test_redact_cli_args_hides_sensitive_values():
    """CLI audit logs must not contain API keys or passwords."""
    from drbrain.cli._helpers.security import redact_cli_args

    rendered = redact_cli_args(
        [
            "fetch",
            "--api-key",
            "SECRET-123",
            "--password=letmein",
            "--token",
            "token-value",
            "--limit",
            "3",
        ]
    )

    assert "SECRET-123" not in rendered
    assert "letmein" not in rendered
    assert "token-value" not in rendered
    assert "<redacted>" in rendered
    assert "--limit 3" in rendered


def test_redact_cli_args_handles_sensitive_value_starting_with_dash():
    """A secret beginning with a dash is still treated as a value."""
    from drbrain.cli._helpers.security import redact_cli_args

    rendered = redact_cli_args(["repair", "--api-key", "-secret"])

    assert "-secret" not in rendered
