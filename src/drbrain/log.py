"""Logging setup via loguru — zero-config, rotating files, stderr for warnings+."""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from loguru import logger as _logger

from drbrain.runtime import RuntimeContext
from drbrain.security import REDACTED, configured_secret_values, redact_sensitive_text

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
STDERR_FORMAT = "<level>{level: <8}</level> | {name}:{line} | {message}"

_initialized = False
_session_id: str | None = None
_configured_log_path: Path | None = None
_configured_secrets: tuple[str, ...] = ()


def _replace_configured_secrets(value: str) -> str:
    """Replace configured credentials, including short opaque test tokens.

    Exact replacement is required for provider exceptions that omit a field
    name.  Very short values are matched as standalone tokens so a toy key
    such as ``one`` cannot erase ordinary words that merely contain it; longer
    credentials are replaced anywhere because they commonly occur in URLs or
    concatenated provider diagnostics.
    """
    rendered = value
    for secret in _configured_secrets:
        if not secret:
            continue
        if len(secret) < 4:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])"
            rendered = re.sub(pattern, REDACTED, rendered)
        else:
            rendered = rendered.replace(secret, REDACTED)
    return rendered


def _redact_log_text(value: object) -> str:
    """Apply exact configured-secret replacement before pattern redaction."""
    rendered = _replace_configured_secrets(str(value))
    return redact_sensitive_text(rendered) or ""


class _RedactingLoggingFilter(logging.Filter):
    """Apply the same secret policy to records emitted by stdlib logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        safe = _redact_log_text(rendered)
        record.msg = safe
        record.args = ()
        if record.exc_info:
            try:
                traceback_text = "".join(traceback.format_exception(*record.exc_info))
            except Exception:
                traceback_text = str(record.exc_info[1])
            safe_traceback = _redact_log_text(traceback_text)
            if safe_traceback == traceback_text:
                # An exception value is an untrusted provider-controlled
                # string.  If no configured or labelled credential was found,
                # retain only its type rather than publishing an opaque value
                # that cannot be proven non-secret.
                safe_traceback = f"{type(record.exc_info[1]).__name__}: {REDACTED}"
            if safe_traceback:
                record.msg = f"{record.msg}\n{safe_traceback.rstrip()}"
            record.exc_info = None
            record.exc_text = None
        return True


_STD_LOG_FILTER = _RedactingLoggingFilter()


def _install_std_logging_redaction() -> None:
    """Attach a redacting filter to existing stdlib sinks and lastResort."""
    root_logger = logging.getLogger()
    handlers = list(root_logger.handlers)
    for name in list(root_logger.manager.loggerDict):
        logger = logging.getLogger(name)
        handlers.extend(logger.handlers)
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    for handler in handlers:
        if not any(isinstance(item, _RedactingLoggingFilter) for item in handler.filters):
            handler.addFilter(_STD_LOG_FILTER)


def _redact_log_record(record: dict) -> bool:
    """Sanitize messages and exception traces at the final sink boundary."""
    record["message"] = _redact_log_text(record.get("message", ""))
    exception = record.get("exception")
    if exception is not None:
        exception_type = getattr(exception, "type", None)
        exception_value = getattr(exception, "value", None)
        exception_traceback = getattr(exception, "traceback", None)
        try:
            if exception_type is None and exception_value is None and exception_traceback is None:
                # ``logger.exception`` is occasionally called outside an
                # active ``except`` block.  Loguru still supplies an
                # exception record, but rendering its ``None`` fields can
                # raise inside this filter and leak the original message via
                # the handler's fallback diagnostic.
                rendered = ""
            else:
                rendered = "".join(
                    traceback.format_exception(
                        exception_type,
                        exception_value,
                        exception_traceback,
                    )
                )
        except Exception:
            type_name = getattr(exception_type, "__name__", "Exception")
            rendered = f"{type_name}: {exception_value}"
        safe_rendered = _redact_log_text(rendered)
        if not rendered or safe_rendered == rendered:
            type_name = getattr(exception_type, "__name__", "Exception")
            safe_rendered = f"{type_name}: {REDACTED}"
        if safe_rendered:
            record["message"] = f"{record['message']}\n{safe_rendered[:4000].rstrip()}"
        # Loguru renders ``record["exception"]`` after the message, bypassing
        # message filters.  Always replace it, including when the value is an
        # opaque provider string that no key-oriented regex can classify.
        record["exception"] = None
    return True


def get_session_id() -> str:
    """Return a stable UUID4 for this process lifetime. Lazily initialized."""
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())
    return _session_id


def ui(message: str) -> None:
    """Write to both console and log — canonical output for CLI commands."""
    safe_message = _redact_log_text(message)
    _logger.opt(depth=1).info(safe_message)
    print(safe_message, file=sys.stdout)


def setup_logging(
    level: str = "DEBUG",
    log_path: str | Path | None = None,
    *,
    secrets: Sequence[str | None] = (),
) -> None:
    """Configure loguru with rotating file + stderr output.

    A runtime root may change between embedded CLI invocations, so an already
    initialized logger is reused only when it points at the same file.
    """
    global _initialized, _configured_log_path, _configured_secrets
    requested_secrets = tuple(
        sorted(
            {str(secret) for secret in secrets if secret and not str(secret).startswith("${")},
            key=len,
            reverse=True,
        )
    ) + tuple(configured_secret_values(os.environ))
    _install_std_logging_redaction()
    # Presence is significant: an explicitly empty selector is a malformed
    # isolation request, not permission to fall back to the process CWD or a
    # legacy alias.  RuntimeContext performs the detailed validation.
    has_runtime_selector = "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ
    runtime = RuntimeContext.create() if has_runtime_selector else None
    if log_path is None:
        log_path = Path("data/logs/drbrain.log")
    lexical_path = Path(log_path).expanduser()
    if runtime is not None:
        log_path = runtime.assert_within_root(lexical_path, label="log file")
    else:
        if lexical_path.is_symlink():
            raise ValueError(f"log file must not be a symlink: {lexical_path}")
        current = lexical_path.parent
        while current != current.parent:
            if current.is_symlink():
                raise ValueError("log file contains a symlink component")
            current = current.parent
        log_path = lexical_path.resolve()
    if _initialized and _configured_log_path == log_path:
        # A process can invoke the CLI repeatedly (or embed it in a worker)
        # while reusing one sink.  Keep every secret seen by that sink: an
        # empty/partial second config must never turn off redaction for the
        # already-installed handler.  Retaining old values is harmless and
        # errs toward over-redaction when a deployment rotates credentials.
        _configured_secrets = tuple(
            sorted(
                set(_configured_secrets).union(requested_secrets),
                key=len,
                reverse=True,
            )
        )
        return
    if _initialized:
        _logger.remove()
        _initialized = False
    _configured_secrets = requested_secrets
    _configured_log_path = None

    _logger.remove()  # clear default handler
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if runtime is not None:
            runtime.assert_within_root(log_path, label="log file")
        elif log_path.is_symlink():
            raise ValueError(f"log file must not be a symlink: {log_path}")

        # Pre-create the sink with restrictive permissions.  This keeps the
        # invariant even when the process umask is permissive and before the
        # first log record causes loguru to open the file.
        if not log_path.exists():
            log_path.touch(mode=0o600)
        os.chmod(log_path, 0o600)
        os.umask(0o077)

        _logger.add(
            str(log_path),
            rotation="10 MB",
            retention=5,
            level=level,
            format=LOG_FORMAT,
            encoding="utf-8",
            filter=cast(Any, _redact_log_record),
            diagnose=False,
            backtrace=False,
        )
        # Log files can contain research data and provider diagnostics.  Make
        # the permission invariant explicit instead of relying on umask.
        if log_path.exists():
            os.chmod(log_path, 0o600)

        _logger.add(
            sys.stderr,
            level="WARNING",
            format=STDERR_FORMAT,
            colorize=True,
            filter=cast(Any, _redact_log_record),
            diagnose=False,
            backtrace=False,
        )
    except Exception:
        # Do not leave a failed first setup looking initialized: a subsequent
        # call must get a real chance to install its sinks.
        _logger.remove()
        _initialized = False
        _configured_log_path = None
        _configured_secrets = tuple(
            sorted(set(_configured_secrets).union(requested_secrets), key=len, reverse=True)
        )
        raise

    _configured_log_path = log_path
    _initialized = True
    _logger.info(f"Session started: {get_session_id()}")


def get_logger(name: str = ""):
    """Get a logger instance. If name is empty, returns root loguru logger."""
    if name:
        return _logger.bind(name=name)
    return _logger
