"""Runtime-bound path helpers shared by CLI commands."""

from __future__ import annotations

import os
from pathlib import Path

from drbrain.runtime import RuntimeContext


def runtime_data_path(ctx, value: str | Path, *, label: str) -> Path:
    """Resolve a command-owned path within the active runtime root.

    Direct function callers without a selected runtime retain support for
    explicit temporary paths.  A real isolated CLI invocation always carries
    a ``RuntimeContext`` (or publishes one of the selector environment vars),
    in which case ``assert_within_root`` rejects traversal, URI values, and
    symlink aliases before any writer receives the path.
    """

    obj = getattr(ctx, "obj", None)
    runtime = obj.get("runtime") if isinstance(obj, dict) else None
    # The callback always has a context object for path defaults, but legacy
    # invocations without a selector must retain their caller environment.
    isolated = obj.get("runtime_isolated") if isinstance(obj, dict) else None
    if isolated is False:
        runtime = None
    # The CLI callback always writes ``runtime_isolated``.  Direct command
    # callers and embedded workers often provide only a RuntimeContext; the
    # absence of the marker must retain that context's containment policy.
    # Only an explicit ``False`` opts into the legacy, intentionally shared
    # path behavior (used by restore without --root).
    isolation_marker = obj.get("runtime_isolated") if isinstance(obj, dict) else None
    explicitly_isolated = isolation_marker is True
    explicitly_legacy = (
        isolation_marker is False and isinstance(obj, dict) and "runtime_isolated" in obj
    )
    has_runtime_selector = "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ
    configured_root = (
        os.environ["DRBRAIN_ROOT"]
        if "DRBRAIN_ROOT" in os.environ
        else os.environ.get("DRBRAIN_RUNTIME_ROOT")
    )

    if runtime is None and not has_runtime_selector and not explicitly_isolated:
        return Path(value).expanduser().resolve()
    # The callback always supplies a context, including legacy invocations.
    # Do not turn that compatibility anchor into an implicit containment policy.
    if runtime is not None and explicitly_legacy and not has_runtime_selector:
        return Path(value).expanduser().resolve()
    if runtime is None:
        runtime = RuntimeContext.create()

    assert_path = getattr(runtime, "assert_within_root", None)
    if callable(assert_path):
        return assert_path(value, label=label)

    # Keep compatibility with lightweight test doubles that expose only a
    # ``root`` attribute, while preserving the explicit-empty selector rule.
    root = getattr(runtime, "root", None) or configured_root
    if not root:
        raise ValueError("runtime root selector must not be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


__all__ = ["runtime_data_path"]
