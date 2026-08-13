"""Backend helpers for model-tool handlers.

These are *convenience* functions a registrant's handler can reuse; they are
not part of the protocol contract. The protocol itself is backend-agnostic:
a handler is any ``Callable[[dict], Any]`` that returns the raw result data or
raises. ``tool.backend`` (``cli`` / ``inprocess`` / ``static``) is declarative
metadata — it documents *how* a tool executes, but the handler decides.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


def run_subprocess_json(cmd: list[str], *, timeout: float = 60.0) -> Any:
    """Run ``cmd`` and parse its stdout as JSON; raise on non-zero exit.

    ``stderr`` is captured into the exception message so the agent/audit trail
    can see *why* a CLI model failed (e.g. missing weights, ImportError).
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
    """Lazily load a joblib model (imports joblib only when called)."""
    import joblib  # local import: joblib is optional until a joblib tool is used

    return joblib.load(path)
