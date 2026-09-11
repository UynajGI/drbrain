"""Data-first plugin declaration: ``PLUGIN_MANIFEST`` + ``HANDLER`` (+ ``JOB_METHODS``).

The v2 declaration style separates metadata from code: instead of (or in
addition to) the inline ``register(registry)`` function, a plugin module MAY
declare module-level

    PLUGIN_MANIFEST = {...}   # Plugin dataclass fields as plain data
    HANDLER = <callable>      # Callable[[dict], Any]
    JOB_METHODS = <object>    # optional; callable submit/poll/cancel

and :meth:`PluginRegistry.discover` builds the descriptor from the manifest.
:func:`build_from_manifest` is the single translation point, shared by
discovery and the conformance suite (:mod:`drbrain.plugins.conformance`).

A module that declares both a manifest and ``register()`` is registered via
the manifest; the inline function is not consulted (declared precedence, one
style per module keeps the audit trail unambiguous).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from drbrain.plugins.protocol import JobMethods, Plugin

# Module-level attribute names of the declaration style.
MANIFEST_KEY = "PLUGIN_MANIFEST"
HANDLER_KEY = "HANDLER"
JOB_METHODS_KEY = "JOB_METHODS"

# Manifest keys with no sensible default — a manifest missing any of them is
# skipped with a warning, mirroring the inline style's skip-on-failure rule.
REQUIRED_FIELDS: tuple[str, ...] = ("name", "description", "input_schema")

_JOB_ATTRS: tuple[str, ...] = ("submit", "poll", "cancel")


def has_manifest(module: Any) -> bool:
    """Whether the module carries a usable (dict) ``PLUGIN_MANIFEST`` declaration."""
    return isinstance(getattr(module, MANIFEST_KEY, None), dict)


def job_methods_from(module: Any) -> JobMethods | None:
    """Normalize ``JOB_METHODS`` into :class:`JobMethods`, or ``None`` when absent.

    Accepts a :class:`JobMethods` instance as-is, or any object exposing
    callable ``submit``/``poll``/``cancel`` (e.g. a ``SimpleNamespace``).
    Raises :class:`TypeError` when declared but not a valid job-method carrier.
    """
    raw = getattr(module, JOB_METHODS_KEY, None)
    if raw is None:
        return None
    if isinstance(raw, JobMethods):
        return raw
    methods = [getattr(raw, attr, None) for attr in _JOB_ATTRS]
    if not all(callable(method) for method in methods):
        raise TypeError(f"{JOB_METHODS_KEY} must expose callable submit/poll/cancel")
    return JobMethods(**dict(zip(_JOB_ATTRS, methods, strict=True)))


def build_from_manifest(
    module: Any,
) -> tuple[Plugin, Callable[[dict[str, Any]], Any], JobMethods | None]:
    """Translate a manifest-style module into ``(plugin, handler, jobs)``.

    Raises :class:`ValueError` (missing required keys) or :class:`TypeError`
    (no callable handler / malformed job methods) so the caller can log and
    skip — the same failure semantics as an inline ``register()`` that raises.
    Unknown manifest keys are dropped by the ``Plugin`` constructor, matching
    the inline style's forward-compatibility contract (the conformance suite
    reports them, the loader tolerates them).
    """
    manifest: dict[str, Any] = getattr(module, MANIFEST_KEY)
    missing = [key for key in REQUIRED_FIELDS if key not in manifest]
    if missing:
        raise ValueError(f"PLUGIN_MANIFEST missing required fields: {', '.join(missing)}")
    handler = getattr(module, HANDLER_KEY, None)
    if not callable(handler):
        raise TypeError(f"manifest-style plugin must define a callable module-level {HANDLER_KEY}")
    return Plugin(**manifest), handler, job_methods_from(module)
