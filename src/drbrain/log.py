"""Logging setup via loguru — zero-config, rotating files, stderr for warnings+."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
STDERR_FORMAT = "<level>{level: <8}</level> | {name}:{line} | {message}"
LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "drbrain.log"

_initialized = False


def setup_logging(level: str = "DEBUG") -> None:
    """Configure loguru with rotating file + stderr output. Idempotent."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    _logger.remove()  # clear default handler

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _logger.add(
        str(LOG_FILE),
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


def get_logger(name: str = ""):
    """Get a logger instance. If name is empty, returns root loguru logger."""
    if name:
        return _logger.bind(name=name)
    return _logger
