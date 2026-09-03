"""Plugin registry: register plugins, invoke them, expose them to the agent.

The registry is the single entry point an agent (or a workflow step) uses to
call an external capability. It keeps the plugin-specific handler *out* of the
reasoning path: the agent only ever sees :class:`Plugin` descriptors and
:class:`PluginResult` envelopes.

Concrete plugins (ML models, DFT binaries, data sources) are loaded at runtime
via :meth:`PluginRegistry.discover`, which imports each ``*.py`` module in a
directory and calls its ``register(registry)`` function — drbrain itself ships
only this interface abstraction, never a concrete plugin.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from drbrain.plugins.protocol import (
    Plugin,
    PluginResult,
    ResultStatus,
    make_evidence,
)

_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _input_schema_to_model(plugin: Plugin) -> type | None:
    """Build a pydantic model from ``plugin.input_schema`` for ``fn_schema``.

    Handles the common flat ``object`` schema (typed top-level properties);
    returns ``None`` for anything more complex, in which case the bridge falls
    back to a bare tool (no schema validation).
    """
    try:
        from pydantic import create_model
    except ImportError:
        return None
    schema = plugin.input_schema or {}
    properties = schema.get("properties") or {}
    if schema.get("type") != "object" or not properties:
        return None
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for key, prop in properties.items():
        ptype_raw = prop.get("type")
        if isinstance(ptype_raw, list):
            # JSON Schema union type (e.g. ["array", "string"]): take the
            # first branch — a list is unhashable and would raise TypeError
            # here, killing the whole plugin discovery for the directory.
            ptype_raw = ptype_raw[0] if ptype_raw else None
        ptype = _JSON_TO_PY.get(ptype_raw, Any)
        if key in required:
            fields[key] = (ptype, ...)
        else:
            fields[key] = (ptype | None, None)
    return create_model(f"{plugin.name}_Input", **fields)


class PluginRegistry:
    """Holds registered plugins and invokes them with degradation semantics.

    ``handler`` is ``Callable[[dict], Any]``: it takes the JSON arguments and
    returns the raw result data (or raises). The registry wraps it with a
    timeout and classifies exceptions into :class:`ResultStatus`.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the shared executor, created lazily and reused across calls."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="drbrain-plugin")
        return self._executor

    def register(self, plugin: Plugin, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a plugin and its handler (idempotent: re-register replaces)."""
        if not plugin.name:
            raise ValueError("plugin name must be non-empty")
        self._plugins[plugin.name] = plugin
        self._handlers[plugin.name] = handler

    def discover(self, plugin_dir: str | Path) -> int:
        """Load plugins from an external directory and return the number registered.

        Every ``*.py`` module in ``plugin_dir`` that defines a
        ``register(registry)`` function is imported and its ``register`` called
        with this registry; each module decides how many plugins it registers.
        Modules whose name starts with ``_`` are skipped. A module that fails to
        import (or whose ``register`` raises) is logged and skipped, so one bad
        plugin never aborts discovery of the rest.
        """
        import importlib.util
        import logging

        logger = logging.getLogger(__name__)
        directory = Path(plugin_dir)
        before = len(self._plugins)

        if not directory.is_dir():
            logger.warning("plugin directory %s does not exist", directory)
            return 0

        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"drbrain_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    logger.warning("cannot build module spec for %s", path)
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 — a bad plugin must not stop discovery
                logger.warning("failed to load plugin module %s: %s", path, exc)
                continue
            register = getattr(module, "register", None)
            if not callable(register):
                continue
            try:
                register(self)
            except Exception as exc:  # noqa: BLE001
                logger.warning("plugin module %s register() failed: %s", path, exc)
                continue

        return len(self._plugins) - before

    def get(self, name: str) -> Plugin:
        """Return a registered plugin descriptor (KeyError if unknown)."""
        return self._plugins[name]

    def list_plugins(self) -> list[Plugin]:
        """Return all registered plugin descriptors (registration order)."""
        return list(self._plugins.values())

    def call(self, name: str, arguments: dict[str, Any]) -> PluginResult:
        """Invoke ``name`` with ``arguments`` and wrap the outcome.

        Never raises for plugin-side failures: a broken/missing plugin yields a
        ``MODEL_UNAVAILABLE`` result (the caller abstains) instead of an
        exception propagating into the reasoning loop.
        """
        plugin = self._plugins[name]
        handler = self._handlers[name]
        evidence = make_evidence(plugin, arguments)

        future = self._get_executor().submit(handler, arguments)
        try:
            data = future.result(timeout=plugin.timeout_s)
        except (FutureTimeout, TimeoutError):
            return PluginResult(
                ResultStatus.TIMEOUT,
                evidence=evidence,
                error=f"调用超时(>{plugin.timeout_s}s)",
            )
        except (ValueError, TypeError, KeyError) as exc:
            return PluginResult(
                ResultStatus.INVALID_INPUT,
                evidence=evidence,
                error=f"输入不符合 schema: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — plugin failure must not propagate
            return PluginResult(
                ResultStatus.MODEL_UNAVAILABLE,
                evidence=evidence,
                error=str(exc),
            )

        if data is None:
            return PluginResult(ResultStatus.NO_RESULT, evidence=evidence, error="插件无输出")
        if isinstance(data, PluginResult):
            # Handlers may opt into the public envelope to report precise
            # resource use. Preserve its result semantics while ensuring the
            # registry-owned provenance cannot be replaced by plugin output.
            result_evidence = dict(data.evidence)
            result_evidence.update(evidence)
            if data.data is not None:
                result_evidence["output"] = data.data
            return PluginResult(
                data.status,
                data=data.data,
                evidence=result_evidence,
                error=data.error,
                resource_usage=data.resource_usage,
            )
        evidence["output"] = data
        return PluginResult(
            ResultStatus.OK,
            data=data,
            evidence=evidence,
        )

    def to_llamaindex_tools(
        self,
        *,
        call_override: Callable[[Plugin, dict[str, Any]], Any] | None = None,
        include: Callable[[Plugin], bool] | None = None,
    ) -> list:
        """Bridge every registered plugin to a LlamaIndex ``FunctionTool``.

        Returns ``[]`` when llama-index is unavailable. Each plugin's handler
        calls back into :meth:`call` so degradation semantics and evidence
        provenance are preserved end-to-end.  ``call_override`` is an additive
        durable-loop hook: it receives the descriptor plus JSON arguments and
        may route the call through a ToolBroker without changing legacy callers.
        """
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError:
            return []

        tools = []
        for plugin in self._plugins.values():
            if include is not None and not include(plugin):
                continue

            def _make_fn(p: Plugin) -> Any:
                if call_override is not None:

                    async def _brokered_fn(**kwargs: Any) -> str:
                        result = call_override(p, dict(kwargs))
                        if inspect.isawaitable(result):
                            result = await result
                        return str(result)

                    return _brokered_fn

                def _fn(**kwargs: Any) -> str:
                    result = self.call(p.name, dict(kwargs))
                    return result.to_llm_message(p)

                return _fn

            tools.append(
                FunctionTool.from_defaults(
                    fn=_make_fn(plugin),
                    name=plugin.name,
                    description=plugin.description,
                    fn_schema=_input_schema_to_model(plugin),
                )
            )
        return tools
