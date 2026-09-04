"""Plugin registry: register plugins, invoke them, expose them to the agent.

The registry is the single entry point an agent (or a workflow step) uses to
call an external capability. It keeps the plugin-specific handler *out* of the
reasoning path: the agent only ever sees :class:`Plugin` descriptors and
:class:`PluginResult` envelopes.

Concrete plugins (ML models, DFT binaries, data sources) are loaded at runtime
via :meth:`PluginRegistry.discover`, which imports each ``*.py`` module in a
directory and calls its ``register(registry)`` function — drbrain itself ships
only this interface abstraction, never a concrete plugin.

Execution model (review 2026-09-03 P-E2): by default each call runs in a
disposable worker process, so a timed-out call is SIGKILLed and genuinely
reclaimed instead of leaving a stuck thread parked on the shared executor.
Handlers/arguments that cannot cross a process boundary automatically fall
back to the legacy shared-thread path with identical semantics.

Output hygiene (P-I3): every successful output is clipped to a hard byte cap
(:data:`DEFAULT_MAX_OUTPUT_BYTES`, overridable per plugin or per call) so one
greedy data plugin cannot blow up the model context.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import pickle
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Literal

from drbrain.plugins.protocol import (
    JobMethods,
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

# P-I3: default hard cap on plugin output size (JSON-encoded, UTF-8 bytes).
DEFAULT_MAX_OUTPUT_BYTES = 256_000

# How often (seconds) the parent re-checks worker liveness/timeout.
_PROCESS_POLL_INTERVAL_S = 0.05

# Recursion guard for pathological nested schemas; deeper objects degrade to
# plain ``dict`` annotations instead of exploding the generated model tree.
_MAX_SCHEMA_DEPTH = 8

_SAFE_NAME_RE = re.compile(r"\W")

_TOOL_INPUT_BASE: type | None = None


def _safe_identifier(name: str, used: set[str] | None = None) -> str:
    """Sanitize an arbitrary JSON-schema key into a python identifier."""
    cleaned = _SAFE_NAME_RE.sub("_", name) or "field"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if used is not None:
        base, n = cleaned, 1
        while cleaned in used:
            n += 1
            cleaned = f"{base}_{n}"
        used.add(cleaned)
    return cleaned


def _tool_input_base() -> type:
    """Base model for generated tool inputs: extra keys are forbidden.

    Mirrors ``additionalProperties: false`` so the emitted tool schema pins
    the parameter set down for small models instead of letting them freestyle
    (the disease behind the "只传 code 字段" prompt band-aids).
    """
    global _TOOL_INPUT_BASE
    if _TOOL_INPUT_BASE is None:
        from pydantic import BaseModel, ConfigDict

        class _ToolInputBase(BaseModel):
            model_config = ConfigDict(extra="forbid")

        _TOOL_INPUT_BASE = _ToolInputBase
    return _TOOL_INPUT_BASE


def _enum_annotation(prop: dict[str, Any]) -> Any:
    """``Literal[...]`` for a scalar-only ``enum`` spec; ``None`` otherwise.

    Non-scalar enum values (float/dict/list) cannot be expressed as
    ``Literal`` and fall back to the plain type annotation.
    """
    values = prop.get("enum")
    if not isinstance(values, list) or not values:
        return None
    if not all(v is None or isinstance(v, (str, int, bool)) for v in values):
        return None
    deduped = tuple(dict.fromkeys(values))
    return Literal[deduped[0]] if len(deduped) == 1 else Literal[deduped]


def _annotation_for(key: str, prop: dict[str, Any], model_name: str, base: type, depth: int) -> Any:
    """Map one property spec to a python annotation, recursing into objects."""
    ptype_raw = prop.get("type")
    if isinstance(ptype_raw, list):
        # JSON Schema union type (e.g. ["array", "string"]): take the
        # first branch — same fallback as the legacy bridge.
        ptype_raw = ptype_raw[0] if ptype_raw else None
    if ptype_raw == "object" and isinstance(prop.get("properties"), dict) and prop["properties"]:
        if depth < _MAX_SCHEMA_DEPTH:
            sub = _build_model(_safe_identifier(f"{model_name}_{key}"), prop, base, depth + 1)
            if sub is not None:
                return sub
        return dict
    if ptype_raw == "array":
        items = prop.get("items")
        if (
            depth < _MAX_SCHEMA_DEPTH
            and isinstance(items, dict)
            and items.get("type") == "object"
            and isinstance(items.get("properties"), dict)
            and items["properties"]
        ):
            sub = _build_model(_safe_identifier(f"{model_name}_{key}_item"), items, base, depth + 1)
            if sub is not None:
                return list[sub]  # type: ignore[valid-type]  # runtime-built alias
        return list
    annotation: Any = _JSON_TO_PY.get(str(ptype_raw), Any)
    enum = _enum_annotation(prop)
    if enum is not None:
        annotation = enum
    return annotation


def _build_model(
    model_name: str, obj_schema: dict[str, Any], base: type, depth: int = 0
) -> type | None:
    """Recursively build one (nested) tool-input model from an object schema."""
    from pydantic import Field, create_model

    if not isinstance(obj_schema, dict) or obj_schema.get("type") != "object":
        return None
    properties = obj_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    required = set(obj_schema.get("required") or [])
    used_names: set[str] = set()
    fields: dict[str, Any] = {}
    for raw_key, raw_prop in properties.items():
        key = str(raw_key)
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        annotation = _annotation_for(key, prop, model_name, base, depth)
        if key in required:
            # required 参数仍尊重 schema 里的 default（有默认值即可不传）
            default: Any = prop.get("default", ...)
        else:
            default = prop.get("default", None)
            if default is None:
                annotation = annotation | None
        field_name = _safe_identifier(key, used_names)
        info_kwargs: dict[str, Any] = {}
        if field_name != key:
            info_kwargs["alias"] = key  # 保留原始 JSON 参数名（如 "k-points"）
        description = prop.get("description")
        if description:
            info_kwargs["description"] = str(description)
        field_info = Field(default, **info_kwargs) if info_kwargs else default
        fields[field_name] = (annotation, field_info)
    return create_model(_safe_identifier(model_name), __base__=base, **fields)


def json_schema_to_model(name: str, schema: dict[str, Any]) -> type | None:
    """Build a pydantic model from a JSON Schema for ``FunctionTool.fn_schema``.

    Unlike the legacy flat bridge, this preserves per-parameter
    ``description`` / ``enum`` / ``default``, recurses nested ``object``
    properties and object ``array`` items into sub-models, and sets
    ``additionalProperties: false`` on every generated model — so the tool
    definition a (4B-class) model sees carries real parameter constraints.

    Returns ``None`` for non-object / property-less schemas or when pydantic
    is unavailable; callers then fall back to a schema-less tool.
    """
    try:
        import pydantic  # noqa: F401 — availability probe
    except ImportError:
        return None
    return _build_model(_safe_identifier(name), schema, _tool_input_base())


def _input_schema_to_model(plugin: Plugin) -> type | None:
    """Build a pydantic model from ``plugin.input_schema`` for ``fn_schema``."""
    return json_schema_to_model(plugin.name, plugin.input_schema or {})


def _load_module_from_file(path: str) -> Any:
    """Import a plugin module from its source file (child-side file reference)."""
    import importlib.util

    module_name = f"drbrain_plugin_{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_envelope(
    envelope: dict[str, Any],
) -> tuple[Callable[[dict[str, Any]], Any], dict[str, Any]]:
    """Materialize the ``(handler, arguments)`` pair described by an envelope."""
    kind = envelope.get("kind")
    arguments = envelope["arguments"]
    if kind == "pickle":
        return envelope["handler"], arguments
    if kind == "file":
        # discover() 以 spec 方式加载插件模块，模块名不可 import；
        # 用 (源文件, 限定名) 在子进程里重建同一个函数。
        module = _load_module_from_file(envelope["path"])
        obj: Any = module
        for part in str(envelope["qualname"]).split("."):
            obj = getattr(obj, part)
        return obj, arguments
    raise TypeError(f"unknown handler envelope kind: {kind!r}")


def _process_entry(conn: Any, payload: bytes) -> None:
    """Child-side entry: unpack the payload, run the handler, send the outcome.

    A failure BEFORE the handler runs is reported as ``"unpicklable"`` so the
    parent can safely re-run the call on the legacy thread path (no side
    effects have happened yet). This function never raises past its body.
    """
    try:
        handler, arguments = _resolve_envelope(pickle.loads(payload))
    except BaseException as exc:  # noqa: BLE001 — must always answer the parent
        conn.send(("unpicklable", str(exc)))
        return
    try:
        data = handler(arguments)
    except (ValueError, TypeError, KeyError) as exc:
        conn.send(("invalid_input", str(exc)))
    except TimeoutError:
        # 与线程路径一致：handler 自身抛超时按 TIMEOUT 归类
        conn.send(("timeout", None))
    except BaseException as exc:  # noqa: BLE001
        conn.send(("error", str(exc)))
    else:
        try:
            conn.send(("ok", data))
        except BaseException as exc:  # noqa: BLE001 — 结果必须能跨进程
            conn.send(("error", f"插件结果无法跨进程传输: {exc}"))
    finally:
        conn.close()


def _file_reference(handler: Callable[..., Any]) -> tuple[str, str] | None:
    """``(source file, qualname)`` for a plain module-level function, else ``None``."""
    if not inspect.isfunction(handler):
        return None
    qualname = getattr(handler, "__qualname__", "")
    if not qualname or "<" in qualname:
        return None  # lambda / 局部函数无法按文件重建
    path = handler.__globals__.get("__file__")
    if not path:
        return None
    return str(path), qualname


def _pack_payload(
    handler: Callable[[dict[str, Any]], Any], arguments: dict[str, Any]
) -> bytes | None:
    """Serialize ``(handler, arguments)`` for a child; ``None`` when impossible.

    Preference order: direct pickle (importable callables), then a
    file-reference envelope for ``discover()``-loaded plugin modules (their
    module name is not importable, but their source file is on disk). The
    pickle candidate is round-trip verified in the parent so an unimportable
    module never reaches the child.
    """
    try:
        payload = pickle.dumps({"kind": "pickle", "handler": handler, "arguments": arguments})
        pickle.loads(payload)  # 验证子进程侧能还原（模块可导入）
        return payload
    except Exception:  # noqa: BLE001 — falls through to the file reference
        pass
    ref = _file_reference(handler)
    if ref is None:
        return None
    try:
        return pickle.dumps(
            {"kind": "file", "path": ref[0], "qualname": ref[1], "arguments": arguments}
        )
    except Exception:  # noqa: BLE001
        return None


def _run_handler_in_process(
    handler: Callable[[dict[str, Any]], Any], arguments: dict[str, Any], timeout: float
) -> tuple[str, Any]:
    """Run ``handler(arguments)`` in a disposable worker process.

    Returns the same ``(kind, payload)`` outcome tuples as the thread path;
    ``"fallback"`` means the call must be re-run on the legacy thread path
    (the handler has provably not started). On timeout the worker is
    SIGKILLed, so the OS reclaims it — stuck compute can never wedge the
    registry (P-E2).
    """
    payload = _pack_payload(handler, arguments)
    if payload is None:
        return ("fallback", "handler/arguments cannot be pickled")
    ctx = multiprocessing.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_process_entry, args=(send_conn, payload), daemon=True)
    try:
        proc.start()
    except Exception as exc:  # noqa: BLE001 — 进程机制不可用时退回线程
        recv_conn.close()
        return ("fallback", f"cannot start worker process: {exc}")
    finally:
        send_conn.close()  # 父进程必须关闭写端，EOF 语义才成立

    deadline = time.monotonic() + max(timeout, 0.0)
    outcome: tuple[str, Any] = ("error", "worker produced no outcome")
    while True:
        if time.monotonic() >= deadline:
            proc.kill()  # SIGKILL：超时后工作单元真正可回收
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
            outcome = ("timeout", None)
            break
        if recv_conn.poll(_PROCESS_POLL_INTERVAL_S):
            try:
                outcome = recv_conn.recv()
            except EOFError:
                outcome = ("error", "worker closed the pipe without a result")
            break
        if not proc.is_alive():
            outcome = ("error", f"worker process died (exitcode={proc.exitcode})")
            break
    recv_conn.close()
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
    # 子进程报告载荷无法还原 → 安全回退线程路径（handler 尚未执行）
    if outcome[0] == "unpicklable":
        return ("fallback", outcome[1])
    return outcome


def _resolve_output_cap(plugin: Plugin, override: int | None) -> int | None:
    """Cap precedence: explicit call argument > plugin declaration > default."""
    if override is not None:
        return override
    if plugin.max_output_bytes is not None:
        return plugin.max_output_bytes
    return DEFAULT_MAX_OUTPUT_BYTES


def _truncate_output(data: Any, max_bytes: int | None) -> tuple[Any, bool]:
    """Bound an output to ``max_bytes``; ``None``/``<=0`` means unlimited.

    Mirrors the loop-side ``tool_broker._bounded`` digest shape so evidence
    records look the same whichever boundary clipped them.
    """
    if max_bytes is None or max_bytes <= 0:
        return data, False
    encoded = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return data, False
    return (
        {
            "truncated": True,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "preview": encoded[:max_bytes].decode("utf-8", errors="ignore"),
        },
        True,
    )


class PluginRegistry:
    """Holds registered plugins and invokes them with degradation semantics.

    ``handler`` is ``Callable[[dict], Any]``: it takes the JSON arguments and
    returns the raw result data (or raises). The registry wraps it with a
    timeout, classifies exceptions into :class:`ResultStatus`, clips oversized
    outputs, and — by default — runs each call in a disposable worker process
    so timeouts are actually enforceable.
    """

    def __init__(self, *, process_isolation: bool = True) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._jobs: dict[str, JobMethods] = {}
        self._executor: ThreadPoolExecutor | None = None
        # P-E2: 默认每调用一个独立子进程，超时可 SIGKILL 真正回收；
        # 不可 pickle 的 handler 自动回退共享线程路径（旧行为）。
        self._process_isolation = process_isolation

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the shared executor, created lazily and reused across calls."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="drbrain-plugin")
        return self._executor

    def register(
        self,
        plugin: Plugin,
        handler: Callable[[dict[str, Any]], Any],
        jobs: JobMethods | None = None,
    ) -> None:
        """Register a plugin, its handler and optional job methods (idempotent: re-register replaces)."""
        if not plugin.name:
            raise ValueError("plugin name must be non-empty")
        self._plugins[plugin.name] = plugin
        self._handlers[plugin.name] = handler
        if jobs is not None:
            self._jobs[plugin.name] = jobs
        else:
            self._jobs.pop(plugin.name, None)

    def supports_jobs(self, name: str) -> bool:
        """Whether ``name`` registered asynchronous-job methods."""
        return name in self._jobs

    def submit_job(self, name: str, arguments: dict[str, Any]) -> str:
        """Submit an asynchronous job and return its ``job_id``.

        Raises :class:`NotImplementedError` for plugins that did not register
        :class:`JobMethods` (the protocol default). ``submit`` must return
        quickly — enqueue only; the heavy work runs outside this process, and
        the durable result lands in the ``jobs/`` directory (see the package
        docstring of :mod:`drbrain.plugins`).
        """
        jobs = self._jobs.get(name)
        if jobs is None:
            raise NotImplementedError(f"plugin {name!r} 未注册异步作业方法(submit/poll/cancel)")
        return str(jobs.submit(arguments))

    def poll_job(self, name: str, job_id: str) -> dict[str, Any]:
        """Poll one job: ``{"status": JobStatus | str, "result"?: Any, "error"?: str}``.

        Raises :class:`NotImplementedError` for plugins without job methods.
        """
        jobs = self._jobs.get(name)
        if jobs is None:
            raise NotImplementedError(f"plugin {name!r} 未注册异步作业方法(submit/poll/cancel)")
        return jobs.poll(job_id)

    def cancel_job(self, name: str, job_id: str) -> bool:
        """Best-effort cancel one job; returns whether the request was accepted.

        Raises :class:`NotImplementedError` for plugins without job methods.
        """
        jobs = self._jobs.get(name)
        if jobs is None:
            raise NotImplementedError(f"plugin {name!r} 未注册异步作业方法(submit/poll/cancel)")
        return bool(jobs.cancel(job_id))

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

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        max_output_bytes: int | None = None,
    ) -> PluginResult:
        """Invoke ``name`` with ``arguments`` and wrap the outcome.

        Never raises for plugin-side failures: a broken/missing plugin yields a
        ``MODEL_UNAVAILABLE`` result (the caller abstains) instead of an
        exception propagating into the reasoning loop.

        ``max_output_bytes`` overrides the output cap for this call
        (``0``/negative disables it); otherwise the plugin's declared
        :attr:`Plugin.max_output_bytes` wins, and finally
        :data:`DEFAULT_MAX_OUTPUT_BYTES` applies. An oversized output is
        clipped to a ``{"truncated", "bytes", "sha256", "preview"}`` digest and
        the result flagged ``truncated`` (P-I3).
        """
        # P-I5: an unknown plugin name must honor the never-raise contract —
        # a dict KeyError here used to escape into the reasoning loop.
        plugin = self._plugins.get(name)
        handler = self._handlers.get(name)
        if plugin is None or handler is None:
            return PluginResult(
                ResultStatus.INVALID_INPUT,
                error=f"unknown plugin: {name!r}",
            )
        evidence = make_evidence(plugin, arguments)

        kind, payload = self._invoke(handler, arguments, plugin.timeout_s)
        if kind == "timeout":
            return PluginResult(
                ResultStatus.TIMEOUT,
                evidence=evidence,
                error=f"调用超时(>{plugin.timeout_s}s)",
            )
        if kind == "invalid_input":
            return PluginResult(
                ResultStatus.INVALID_INPUT,
                evidence=evidence,
                error=f"输入不符合 schema: {payload}",
            )
        if kind == "error":
            # P-I5: a crashing plugin implementation is PLUGIN_ERROR, not
            # MODEL_UNAVAILABLE — an unavailable model and a broken plugin
            # need different operator responses.
            return PluginResult(
                ResultStatus.PLUGIN_ERROR,
                evidence=evidence,
                error=payload,
            )

        data = payload
        cap = _resolve_output_cap(plugin, max_output_bytes)
        if data is None:
            return PluginResult(ResultStatus.NO_RESULT, evidence=evidence, error="插件无输出")
        if isinstance(data, PluginResult):
            # Handlers may opt into the public envelope to report precise
            # resource use. Preserve its result semantics while ensuring the
            # registry-owned provenance cannot be replaced by plugin output.
            result_evidence = dict(data.evidence)
            result_evidence.update(evidence)
            bounded, truncated = _truncate_output(data.data, cap)
            if data.data is not None:
                result_evidence["output"] = bounded
            return PluginResult(
                data.status,
                data=bounded,
                evidence=result_evidence,
                error=data.error,
                resource_usage=data.resource_usage,
                job_id=data.job_id,
                artifacts=list(data.artifacts),
                truncated=truncated,
            )
        bounded, truncated = _truncate_output(data, cap)
        evidence["output"] = bounded
        return PluginResult(
            ResultStatus.OK,
            data=bounded,
            evidence=evidence,
            truncated=truncated,
        )

    def _invoke(
        self,
        handler: Callable[[dict[str, Any]], Any],
        arguments: dict[str, Any],
        timeout: float,
    ) -> tuple[str, Any]:
        """Run the handler, returning a ``(kind, payload)`` outcome tuple.

        Kinds ``ok`` / ``timeout`` / ``invalid_input`` / ``error`` carry the
        legacy call semantics; ``fallback`` (process path only) means the
        handler provably never started and the thread path must run instead.
        """
        if self._process_isolation:
            outcome = _run_handler_in_process(handler, arguments, timeout)
            if outcome[0] != "fallback":
                return outcome
        # 旧行为：共享线程池。超时后线程仍占位——仅当 handler/参数不可
        # pickle 或进程机制不可用时才走到这里，保留兼容语义。
        future = self._get_executor().submit(handler, arguments)
        try:
            return ("ok", future.result(timeout=timeout))
        except (FutureTimeout, TimeoutError):
            return ("timeout", None)
        except (ValueError, TypeError, KeyError) as exc:
            return ("invalid_input", str(exc))
        except Exception as exc:  # noqa: BLE001 — plugin failure must not propagate
            return ("error", str(exc))

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
