"""Small security helpers shared by the CLI entry points."""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence

from drbrain.security import is_sensitive_key

# Keep this deliberately name-based: command arguments are the only values we
# can reliably identify before Typer has parsed them. The patterns cover the
# options used by DrBrain and common aliases used by plugins.
_SENSITIVE_OPTION = re.compile(
    r"(?:^|[-_])(api[-_]?key|auth[-_]?token|access[-_]?token|refresh[-_]?token|"
    r"password|passwd|secret|credential|private[-_]?key|token)(?:$|[-_=])",
    re.IGNORECASE,
)


def _is_sensitive_option(arg: str) -> bool:
    """Return whether an argv option name carries a secret value."""
    if not arg.startswith("-"):
        return False
    name = arg.split("=", 1)[0]
    return bool(_SENSITIVE_OPTION.search(name) or is_sensitive_key(name.lstrip("-")))


def redact_cli_args(args: Sequence[str]) -> str:
    """Render command arguments with secret values replaced.

    Both ``--api-key VALUE`` and ``--api-key=VALUE`` forms are supported. The
    return value is shell-quoted for unambiguous, safe logging.
    """
    redacted: list[str] = []
    redact_next = False
    for raw in args:
        arg = str(raw)
        if redact_next:
            # A sensitive option consumes exactly one following token, even if
            # that token starts with ``-`` (secrets may legitimately do so).
            redacted.append("<redacted>")
            redact_next = _is_sensitive_option(arg) and "=" not in arg
            continue

        if _is_sensitive_option(arg):
            if "=" in arg:
                option = arg.split("=", 1)[0]
                redacted.append(f"{option}=<redacted>")
            else:
                redacted.append(arg)
                redact_next = True
            continue

        redacted.append(arg)

    return shlex.join(redacted)


__all__ = ["redact_cli_args"]
