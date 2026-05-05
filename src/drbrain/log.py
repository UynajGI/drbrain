"""Logging setup via loguru — zero-config, rotating files, stderr for warnings+."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from loguru import logger as _logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
STDERR_FORMAT = "<level>{level: <8}</level> | {name}:{line} | {message}"

_initialized = False
_session_id: str | None = None


def get_session_id() -> str:
    """Return a stable UUID4 for this process lifetime. Lazily initialized."""
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())
    return _session_id


def ui(message: str) -> None:
    """Write to both console and log — canonical output for CLI commands."""
    _logger.opt(depth=1).info(message)
    print(message, file=sys.stdout)


def setup_logging(level: str = "DEBUG", log_path: str | Path = "data/logs/drbrain.log") -> None:
    """Configure loguru with rotating file + stderr output. Idempotent."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    _logger.remove()  # clear default handler

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.add(
        str(log_path),
        rotation="10 MB",
        retention=5,
        level=level,
        format=LOG_FORMAT,
        encoding="utf-8",
    )

    _logger.add(
        sys.stderr,
        level="WARNING",
        format=STDERR_FORMAT,
        colorize=True,
    )

    _logger.info(f"Session started: {get_session_id()}")


def get_logger(name: str = ""):
    """Get a logger instance. If name is empty, returns root loguru logger."""
    if name:
        return _logger.bind(name=name)
    return _logger
