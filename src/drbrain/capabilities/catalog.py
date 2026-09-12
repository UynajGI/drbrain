"""One registry for every Agent tool, regardless of its native protocol."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from drbrain.capabilities.protocol import (
    CapabilityDescriptor,
    CapabilityJobMethods,
    InvocationResult,
    InvocationStatus,
    function_tool_name,
    input_digest,
    runtime_fingerprint,
    valid_job_id,
)
from drbrain.capabilities.validation import validate_instance, validate_schema

DescriptorList = list[CapabilityDescriptor]
AnyList = list[Any]

MAX_JOB_RECORDS = 4096
JOB_RECORD_TTL_SECONDS = 7 * 24 * 60 * 60
_TERMINAL_JOB_STATUSES = frozenset({"ok", "error", "cancelled", "no_result"})


@dataclass(frozen=True)
class CapabilityEntry:
    """Descriptor plus its host-owned invocation function."""

    descriptor: CapabilityDescriptor
    invoke: Callable[[dict[str, Any]], Any]
    jobs: CapabilityJobMethods | None = None


class CapabilityCatalog:
    """Discover, recommend, and invoke capabilities through one stable API."""

    def __init__(self, *, state_dir: str | Path | None = None) -> None:
        self._entries: dict[str, CapabilityEntry] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._state_dir = Path(state_dir).resolve() if state_dir is not None else None
        self._job_records: dict[tuple[str, str], dict[str, Any]] = {}
        self._load_state()

    @property
    def _state_path(self) -> Path | None:
        return self._state_dir / "capability-jobs.json" if self._state_dir is not None else None

    def _load_state(self) -> None:
        path = self._state_path
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            for item in payload.get("idempotency", []):
                if isinstance(item, list) and len(item) == 3:
                    self._idempotency[(str(item[0]), str(item[1]))] = str(item[2])
            for item in payload.get("jobs", []):
                if isinstance(item, dict) and item.get("capability_id") and item.get("job_id"):
                    self._job_records[(str(item["capability_id"]), str(item["job_id"]))] = item
        except (OSError, ValueError, TypeError):
            return

    def _persist_state(self) -> None:
        path = self._state_path
        if path is None:
            return
        try:
            self._prune_job_records()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "idempotency": [
                    [capability_id, key, job_id]
                    for (capability_id, key), job_id in sorted(self._idempotency.items())
                ],
                "jobs": list(self._job_records.values()),
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError:
            return

    def _prune_job_records(self) -> None:
        """Bound durable job state while retaining active and recent jobs."""
        now = time.time()
        removable: list[tuple[tuple[str, str], float]] = []

        def updated_at(record: dict[str, Any]) -> float:
            try:
                value = record.get("updated_at", now)
                if isinstance(value, bool):
                    raise TypeError
                return float(value)
            except (TypeError, ValueError):
                return now

        for key, record in self._job_records.items():
            status = str(record.get("status") or "").lower()
            if status not in _TERMINAL_JOB_STATUSES:
                continue
            timestamp = updated_at(record)
            if now - timestamp >= JOB_RECORD_TTL_SECONDS:
                removable.append((key, timestamp))
        for key, _timestamp in removable:
            self._job_records.pop(key, None)

        terminal = sorted(
            (
                (key, updated_at(record))
                for key, record in self._job_records.items()
                if str(record.get("status") or "").lower() in _TERMINAL_JOB_STATUSES
            ),
            key=lambda item: item[1],
        )
        overflow = max(0, len(self._job_records) - MAX_JOB_RECORDS)
        for key, _updated_at in terminal[:overflow]:
            self._job_records.pop(key, None)

        retained_jobs = set(self._job_records)
        for key, job_id in list(self._idempotency.items()):
            if (key[0], job_id) not in retained_jobs:
                self._idempotency.pop(key, None)

    def register(
        self,
        descriptor: CapabilityDescriptor,
        invoke: Callable[[dict[str, Any]], Any],
        *,
        jobs: CapabilityJobMethods | None = None,
        replace: bool = False,
    ) -> None:
        if not callable(invoke):
            raise TypeError("capability invoker must be callable")
        if jobs is not None and not isinstance(jobs, CapabilityJobMethods):
            jobs = CapabilityJobMethods(
                submit=getattr(jobs, "submit", None),
                poll=getattr(jobs, "poll", None),
                cancel=getattr(jobs, "cancel", None),
            )
        schema_errors = validate_schema(descriptor.input_schema)
        if schema_errors:
            raise ValueError(f"{descriptor.id} input schema: {'; '.join(schema_errors)}")
        if descriptor.output_schema is not None:
            output_errors = validate_schema(descriptor.output_schema)
            if output_errors:
                raise ValueError(f"{descriptor.id} output schema: {'; '.join(output_errors)}")
        if descriptor.id in self._entries and not replace:
            raise ValueError(f"capability {descriptor.id!r} is already registered")
        self._entries[descriptor.id] = CapabilityEntry(descriptor, invoke, jobs)

    def register_adapter(self, adapter: Any, *, replace: bool = False) -> CapabilityDescriptor:
        """Register any future adapter implementing ``descriptor()`` and ``invoke()``."""
        descriptor = adapter.descriptor()
        jobs = getattr(adapter, "jobs", None) or getattr(adapter, "job_methods", None)
        self.register(descriptor, adapter.invoke, jobs=jobs, replace=replace)
        return descriptor

    def register_plugin_registry(self, registry: Any, *, replace: bool = False) -> int:
        """Expose an existing :class:`PluginRegistry` without changing its API."""
        count = 0
        for plugin in registry.list_plugins():
            descriptor = plugin.to_capability_descriptor()

            def invoke(arguments: dict[str, Any], *, name: str = plugin.name) -> Any:
                return registry.call(name, arguments).to_invocation_result()

            jobs = None
            if registry.supports_jobs(plugin.name):
                jobs = CapabilityJobMethods(
                    submit=lambda arguments, name=plugin.name: registry.submit_job(name, arguments),
                    poll=lambda job_id, name=plugin.name: registry.poll_job(name, job_id),
                    cancel=lambda job_id, name=plugin.name: registry.cancel_job(name, job_id),
                )
            self.register(descriptor, invoke, jobs=jobs, replace=replace)
            count += 1
        return count

    def register_mcp_servers(
        self,
        servers: list[dict[str, Any]],
        *,
        require_trusted: bool = True,
        replace: bool = False,
    ) -> int:
        """Discover MCP tools and expose their rich result envelopes."""
        from drbrain.rag.mcp_tools import (
            call_mcp_tool_result,
            discover_mcp_tools,
            mcp_descriptor_to_capability,
        )

        count = 0
        for server in servers:
            try:
                descriptors = discover_mcp_tools(
                    server, require_trusted=require_trusted, namespace=True
                )
            except Exception:
                continue
            for raw in descriptors:
                descriptor = mcp_descriptor_to_capability(server, raw)

                def invoke(
                    arguments: dict[str, Any],
                    *,
                    server: dict[str, Any] = server,
                    tool_name: str = str(raw["name"]),
                    trusted: bool = require_trusted,
                ) -> InvocationResult:
                    return call_mcp_tool_result(
                        server, tool_name, arguments, require_trusted=trusted
                    )

                self.register(descriptor, invoke, replace=replace)
                count += 1
        return count

    def register_skills(self, root: str, *, replace: bool = False) -> int:
        """Register Skill instructions as discoverable, non-executable entries."""
        from drbrain.capabilities.skills import discover_skills

        count = 0
        for descriptor in discover_skills(root):

            def invoke(_arguments: dict[str, Any]) -> InvocationResult:
                return InvocationResult(
                    InvocationStatus.DENIED,
                    error="Skill packages are instructions/resources and have no executable handler",
                )

            self.register(descriptor, invoke, replace=replace)
            count += 1
        return count

    def get(self, capability_id: str) -> CapabilityEntry:
        return self._entries[capability_id]

    def submit_job(
        self,
        capability_id: str,
        arguments: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Submit a job with a path-safe ID and an optional caller idempotency key."""
        entry = self._entries.get(capability_id)
        if entry is None or entry.jobs is None:
            raise NotImplementedError(f"capability {capability_id!r} has no job methods")
        payload = arguments if arguments is not None else {}
        errors = validate_instance(entry.descriptor.input_schema, payload)
        if errors:
            raise ValueError("; ".join(errors))
        if idempotency_key is not None and not valid_job_id(idempotency_key):
            raise ValueError("idempotency_key must be a path-safe ASCII token")
        if idempotency_key is not None:
            previous = self._idempotency.get((capability_id, idempotency_key))
            if previous is not None:
                return previous
        job_id = str(entry.jobs.submit(payload)).strip()
        if not valid_job_id(job_id):
            raise ValueError("job submit must return a path-safe job_id")
        if idempotency_key is not None:
            self._idempotency[(capability_id, idempotency_key)] = job_id
        self._job_records[(capability_id, job_id)] = {
            "capability_id": capability_id,
            "job_id": job_id,
            "status": "pending",
            "idempotency_key": idempotency_key,
            "updated_at": time.time(),
        }
        self._persist_state()
        return job_id

    def poll_job(self, capability_id: str, job_id: str) -> InvocationResult:
        """Normalize native poll payloads and MCP task states."""
        entry = self._entries.get(capability_id)
        if entry is None or entry.jobs is None:
            return InvocationResult(
                InvocationStatus.UNAVAILABLE, error="capability has no job methods"
            )
        if not valid_job_id(job_id):
            return InvocationResult(InvocationStatus.INVALID_INPUT, error="invalid job_id")
        try:
            payload = entry.jobs.poll(job_id)
        except Exception as exc:  # noqa: BLE001 - job errors are data
            return InvocationResult(InvocationStatus.ERROR, error=str(exc), job_id=job_id)
        if isinstance(payload, InvocationResult):
            self._job_records[(capability_id, job_id)] = {
                "capability_id": capability_id,
                "job_id": job_id,
                "status": payload.status.value
                if isinstance(payload.status, InvocationStatus)
                else str(payload.status),
                "updated_at": time.time(),
            }
            self._persist_state()
            return replace(payload, job_id=job_id)
        if not isinstance(payload, dict):
            return InvocationResult(
                InvocationStatus.ERROR, error="poll must return a mapping", job_id=job_id
            )
        state = str(payload.get("status") or "pending").lower()
        status = {
            "pending": InvocationStatus.PENDING,
            "queued": InvocationStatus.PENDING,
            "running": InvocationStatus.RUNNING,
            "done": InvocationStatus.OK,
            "completed": InvocationStatus.OK,
            "failed": InvocationStatus.ERROR,
            "cancelled": InvocationStatus.CANCELLED,
            "canceled": InvocationStatus.CANCELLED,
        }.get(state, InvocationStatus.ERROR)
        self._job_records[(capability_id, job_id)] = {
            "capability_id": capability_id,
            "job_id": job_id,
            "status": status.value,
            "updated_at": time.time(),
        }
        self._persist_state()
        return InvocationResult(
            status,
            data=payload.get("result"),
            structured_content=payload.get("result"),
            error=str(payload.get("error")) if payload.get("error") else None,
            job_id=job_id,
        )

    def cancel_job(self, capability_id: str, job_id: str) -> bool:
        entry = self._entries.get(capability_id)
        if entry is None or entry.jobs is None or not valid_job_id(job_id):
            return False
        accepted = bool(entry.jobs.cancel(job_id))
        if accepted:
            self._job_records[(capability_id, job_id)] = {
                "capability_id": capability_id,
                "job_id": job_id,
                "status": InvocationStatus.CANCELLED.value,
                "updated_at": time.time(),
            }
            self._persist_state()
        return accepted

    def list(self, *, kind: str | None = None) -> list[CapabilityDescriptor]:
        values = [entry.descriptor for entry in self._entries.values()]
        return [item for item in values if kind is None or item.kind == kind]

    def recommend(
        self, query: str, *, kinds: IterableABC[str] | None = None, limit: int = 10
    ) -> DescriptorList:
        """Return deterministic lexical recommendations for an Agent planner."""
        terms = {term.lower() for term in query.split() if term.strip()}
        allowed = set(kinds) if kinds is not None else None
        ranked: list[tuple[int, str, CapabilityDescriptor]] = []
        for descriptor in self.list():
            if allowed is not None and descriptor.kind not in allowed:
                continue
            haystack = " ".join(
                [descriptor.id, descriptor.name, descriptor.description, *descriptor.permissions]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                ranked.append((score, descriptor.id, descriptor))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[: max(0, limit)]]

    def invoke(
        self, capability_id: str, arguments: dict[str, Any] | None = None
    ) -> InvocationResult:
        """Synchronously invoke a capability.

        This method is a blocking compatibility bridge.  Async callers should
        use :meth:`ainvoke` so the current event loop can await the adapter
        directly instead of waiting on a worker future.
        """
        return _run_awaitable(self.ainvoke(capability_id, arguments))

    async def ainvoke(
        self, capability_id: str, arguments: dict[str, Any] | None = None
    ) -> InvocationResult:
        """Asynchronously invoke a capability without blocking the event loop."""
        entry = self._entries.get(capability_id)
        if entry is None:
            return InvocationResult(
                InvocationStatus.UNAVAILABLE, error=f"unknown capability: {capability_id}"
            )
        payload = arguments if arguments is not None else {}
        if not isinstance(payload, dict):
            return InvocationResult(
                InvocationStatus.INVALID_INPUT, error="arguments must be a JSON object"
            )
        errors = validate_instance(entry.descriptor.input_schema, payload)
        host_evidence = {
            "capability_id": capability_id,
            "input_digest": input_digest(payload),
            "runtime": runtime_fingerprint(),
        }
        if errors:
            return InvocationResult(
                InvocationStatus.INVALID_INPUT,
                error="; ".join(errors),
                evidence=host_evidence,
            )
        try:
            result = entry.invoke(payload)
            if inspect.isawaitable(result):
                result = await result
        except TimeoutError as exc:
            return InvocationResult(
                InvocationStatus.TIMEOUT, error=str(exc), evidence=host_evidence
            )
        except PermissionError as exc:
            return InvocationResult(InvocationStatus.DENIED, error=str(exc), evidence=host_evidence)
        except Exception as exc:  # noqa: BLE001 - capability errors are data
            return InvocationResult(
                InvocationStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                evidence=host_evidence,
            )
        if isinstance(result, InvocationResult):
            return replace(result, evidence={**result.evidence, **host_evidence})
        if hasattr(result, "to_invocation_result"):
            normalized = result.to_invocation_result()
            return replace(normalized, evidence={**normalized.evidence, **host_evidence})
        status = InvocationStatus.NO_RESULT if result is None else InvocationStatus.OK
        return InvocationResult(
            status,
            data=result,
            structured_content=result,
            evidence=host_evidence,
        )

    def to_llamaindex_tools(self, *, kinds: IterableABC[str] | None = None) -> AnyList:
        """Optional bridge for function-calling agents; core stays LlamaIndex-free."""
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError:
            return []
        allowed = set(kinds) if kinds is not None else None
        tools: list[Any] = []
        used_names: set[str] = set()
        for entry in self._entries.values():
            descriptor = entry.descriptor
            if allowed is not None and descriptor.kind not in allowed:
                continue
            model = None
            try:
                from drbrain.plugins.registry import json_schema_to_model

                model = json_schema_to_model(descriptor.name, descriptor.input_schema)
            except ImportError:
                pass

            def _make_fn(capability_id: str) -> Callable[..., str]:
                def _fn(**kwargs: Any) -> str:
                    return self.invoke(capability_id, dict(kwargs)).to_llm_message()

                return _fn

            tools.append(
                FunctionTool.from_defaults(
                    fn=_make_fn(descriptor.id),
                    name=function_tool_name(descriptor.id, used_names),
                    description=descriptor.description,
                    fn_schema=model,
                )
            )
        return tools


def _run_awaitable(awaitable: Any) -> Any:
    """Run an adapter coroutine for synchronous callers.

    Calls made from an active event loop are bridged through a worker thread
    and therefore block that loop; use :meth:`CapabilityCatalog.ainvoke` from
    async code instead.
    """
    try:
        import asyncio

        asyncio.get_running_loop()
    except RuntimeError:
        import asyncio

        return asyncio.run(awaitable)
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()
