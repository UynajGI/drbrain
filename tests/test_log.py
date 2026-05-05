"""Tests for logging setup."""

import re

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
