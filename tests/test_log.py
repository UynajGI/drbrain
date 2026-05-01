"""Tests for logging setup."""

from unittest import mock

from loguru import logger

from drbrain.log import LOG_FORMAT, STDERR_FORMAT, get_logger, setup_logging


def test_setup_logging_creates_log_dir(tmp_path):
    """setup_logging creates log directory and file."""
    log_dir = tmp_path / "logs"
    with (
        mock.patch("drbrain.log.LOG_DIR", log_dir),
        mock.patch("drbrain.log.LOG_FILE", log_dir / "drbrain.log"),
    ):
        # Reset initialization flag
        import drbrain.log as log_mod

        log_mod._initialized = False
        logger.remove()

        setup_logging()
        assert log_dir.exists()
        assert (log_dir / "drbrain.log").exists()

        # Cleanup: remove handler so test output is clean
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
    log_dir = tmp_path / "logs"
    log_file = log_dir / "test.log"
    with mock.patch("drbrain.log.LOG_DIR", log_dir), mock.patch("drbrain.log.LOG_FILE", log_file):
        import drbrain.log as log_mod

        log_mod._initialized = False
        logger.remove()

        setup_logging(level="INFO")
        logger.info("Test log message")
        logger.complete()  # Flush

        content = log_file.read_text()
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
