"""Compatibility aliases for the CLI main-line convergence.

``docs/cli-pipeline-redesign.md`` renames several entry points onto the
``ingest → index build → search / ask`` main line.  The old names stay
callable for a compatibility period — identical flags, defaults, exit codes
and JSON contracts — but they are registered ``hidden=True`` (so the main help
page shows only the new flow) and print one migration line on stderr.

``functools.wraps`` is load-bearing: Typer reads the command signature with
``inspect.signature``, which follows ``__wrapped__``, so the registered alias
keeps exactly the original parameters.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import typer


def migration_alias(
    command: Callable[..., Any],
    *,
    name: str,
    hint: str,
) -> Callable[..., Any]:
    """Wrap ``command`` so the old entry point announces the new one.

    The notice goes to stderr only; stdout stays byte-compatible for callers
    that parse ``--json``.  Direct (non-CLI) callers of the original function
    are unaffected because the wrapper exists only on the Typer registration.
    """

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        typer.echo(f"[drbrain] '{name}' has moved: use '{hint}'", err=True)
        return command(*args, **kwargs)

    return wrapper


__all__ = ["migration_alias"]
