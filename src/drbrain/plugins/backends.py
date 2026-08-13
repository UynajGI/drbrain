"""Backend helpers for plugin handlers.

These are *convenience* functions a registrant's handler can reuse; they are
not part of the protocol contract. The protocol itself is backend-agnostic:
a handler is any ``Callable[[dict], Any]`` that returns the raw result data or
raises. ``plugin.backend`` (``subprocess`` / ``inprocess`` / ``static``) is
declarative metadata — it documents *how* a plugin executes, but the handler
decides.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


def run_subprocess(
    cmd: list[str],
    *,
    timeout: float = 60.0,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run an external command/software and return the completed process.

    Unlike :func:`run_subprocess_json`, this does not parse output — a
    software handler is expected to write its own input files, run the binary,
    and parse its own output. ``stdout`` / ``stderr`` are captured as text on
    the returned :class:`subprocess.CompletedProcess`.

    Security: ``cmd`` must be a caller-controlled *static* command (e.g.
    ``["lmp", "-in", "input.in"]``); never interpolate untrusted input into
    ``cmd`` — untrusted data belongs in input files/stdin, not on the command
    line. ``shell`` is left ``False`` (no shell interpolation).
    """
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, shell=False
    )


def run_subprocess_json(cmd: list[str], *, timeout: float = 60.0) -> Any:
    """Run ``cmd`` and parse its stdout as JSON; raise on non-zero exit.

    ``stderr`` is captured into the exception message so the agent/audit trail
    can see *why* a CLI-backed plugin failed (e.g. missing weights, ImportError).
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"命令退出码 {proc.returncode}: {proc.stderr.strip()[:500]}")
    if not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def load_joblib(path: str) -> Any:
    """Lazily load a joblib artifact (imports joblib only when called)."""
    import joblib  # local import: joblib is optional until a joblib plugin is used

    return joblib.load(path)
