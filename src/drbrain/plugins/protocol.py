"""Plugin protocol — generic abstraction for exposing external capabilities as agent tools.

Similar to MCP's *tool* abstraction (``name + description + JSON Schema + call``)
but deliberately **not** MCP's transport/resource layer. This protocol adds what
a plain tool schema cannot express:

* **plugin metadata** — ``plugin_type`` / ``version`` / ``resource``, so the
  agent and the audit trail know *which* artifact produced a result;
* **degradation semantics** — a :class:`PluginResult` status wired to the
  Epistemic Layer (:mod:`drbrain.rag.status`), so a broken plugin leads to
  ``abstain`` instead of a hallucinated answer;
* **evidence provenance** — every result carries plugin name / version /
  input / timestamp, ready to be persisted into ``answer_records`` / ``claims``
  / ``evidence``.

The protocol is **plugin-agnostic**: this module never imports or names a
concrete plugin. A flat-band ML model, a DFT binary, and a material-property
database are all just :class:`Plugin` registrations — the protocol does not
change when the plugins change. Concrete plugins are discovered at runtime from
external directories (see :meth:`PluginRegistry.discover`).

**Asynchronous jobs** — hour-scale compute (e.g. DFT) must not block a loop
turn, so a plugin may additionally register :class:`JobMethods`
(``submit`` / ``poll`` / ``cancel``) alongside its synchronous handler. The
on-disk ``jobs/`` directory contract these methods feed (result JSON at
``jobs/<job_id>.json``, log at ``jobs/<job_id>.log`` — the only evidence the
T4 gate trusts) is documented in the package docstring of
:mod:`drbrain.plugins`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from drbrain.capabilities import (
    CapabilityAnnotations,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityProvenance,
    InvocationResult,
    InvocationStatus,
    dependency_lock_digest,
    file_digest,
    input_digest,
    runtime_fingerprint,
)

# Coarse capability category. Concrete implementation details (e.g. ``gbdt`` vs
# ``gnn``) belong in :attr:`Plugin.metadata`, not in this strict type.
PluginType = Literal["model", "software", "data", "formula", "other"]
Backend = Literal["subprocess", "inprocess", "static"]
# Deprecated aliases kept ONLY so external plugins importing them keep working;
# no code path reads them anymore (P-I2).
OnFailure = Literal["abstain", "stale", "none"]
PluginSideEffect = Literal["pure", "read", "write", "irreversible", "unspecified"]

# Plugin ABI contract.  ``HOST_ABI_VERSION`` is the newest descriptor revision
# this build speaks; a plugin that declares a version outside
# ``SUPPORTED_ABI_VERSIONS`` is rejected at registration time (fail closed)
# instead of being loaded with unknown fields silently dropped.  Plugins that
# do not declare ``abi_version`` at all default to 1 and stay loadable.
HOST_ABI_VERSION = 1
SUPPORTED_ABI_VERSIONS: tuple[int, ...] = (1,)


class ResultStatus(StrEnum):
    """Machine-readable outcome of one plugin call (degradation-aware).

    Mirrors :class:`drbrain.rag.status.RetrievalStatus` semantics, narrowed to
    plugin calls: ``OK`` / ``NO_RESULT`` are usable, the rest are failures the
    caller must not treat as an answer.
    """

    OK = "ok"
    NO_RESULT = "no_result"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    # P-I5: a plugin that CRASHED is not "the model is gone" — callers (and
    # reports) should be able to tell an infrastructure outage from a broken
    # plugin implementation.
    PLUGIN_ERROR = "plugin_error"


class JobStatus(StrEnum):
    """Machine-readable lifecycle state of one asynchronous plugin job.

    Returned by ``poll(job_id)`` inside the ``"status"`` key. ``DONE`` means
    the on-disk ``jobs/<job_id>.json`` result file is ready for consumption;
    the other states mean the caller must not treat anything as a result.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _drop_unknown_kwargs(cls):
    """Constructor wrapper for :class:`Plugin`: drop unknown keyword arguments.

    A plain frozen dataclass raises TypeError on unknown kwargs, which would
    kill plugin discovery for any external plugin written against an older
    protocol revision. Wrapping ``__init__`` keeps the frozen dataclass
    semantics while honoring the documented "unknown kwargs are ignored"
    forward-compatibility contract.
    """
    import inspect

    original_init = cls.__init__
    known = {name for name in inspect.signature(original_init).parameters if name != "self"}

    def init_wrapper(self, *args: Any, **kwargs: Any) -> None:
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        original_init(self, *args, **kwargs)

    cls.__init__ = init_wrapper
    return cls


@_drop_unknown_kwargs
@dataclass(frozen=True)
class Plugin:
    """Generic descriptor of an external capability exposed as an agent tool.

    Frozen: a registered plugin is immutable. Unknown keyword arguments are
    silently dropped (see :func:`_drop_unknown_kwargs`) so external plugins
    written against older protocol revisions (e.g. passing
    ``on_failure=``/``entry=``) keep loading instead of raising TypeError
    during discovery. ``summary_fields`` names the keys whose values are worth
    surfacing to the LLM alongside the raw JSON (the "raw JSON + key-field
    summary" compromise).

    ``plugin_type`` is a coarse capability category (``model`` / ``software`` /
    ``data`` / ``formula`` / ``other``). ``resource`` is the single path/pointer
    to the plugin's artifact — a weights file, a software binary, or a data
    file, depending on ``plugin_type``. Free-form specifics go in ``metadata``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    # Compat-only fields (P-I2): declared unread since the schema-bridge
    # rework, but external plugins still pass them to Plugin(...); removing
    # them outright would TypeError during third-party discovery.
    output_schema: dict[str, Any] | None = None
    plugin_type: PluginType = "other"
    version: str = ""
    resource: str | None = None
    backend: Backend = "inprocess"
    entry: str = ""  # deprecated, unread (compat: older plugins pass it)
    on_failure: OnFailure = "abstain"  # deprecated, unread (compat)
    timeout_s: float = 60.0
    summary_fields: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # Durable-loop metadata is additive.  ``unspecified`` keeps legacy direct
    # registry calls working, but a ToolBroker rejects it until the host has
    # explicitly classified the external capability.
    side_effect: PluginSideEffect = "unspecified"
    required_capabilities: tuple[str, ...] = ()
    code_digest: str = ""
    resource_scope: dict[str, Any] = field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    max_output_bytes: int | None = None
    cost_hint: float | None = None
    supports_idempotency: bool = False
    supports_reconcile: bool = False
    supports_cancel: bool = False
    sandbox_profile: str = ""
    approval_policy: str = "default"
    # Protocol revision this plugin was written against.  Appended LAST so the
    # pre-ABI positional argument layout is preserved for external plugins
    # (``1`` covers every plugin written before the field existed; a value
    # outside ``SUPPORTED_ABI_VERSIONS`` fails registration).
    abi_version: int = 1

    @property
    def capability_id(self) -> str:
        """Stable ID used when this descriptor crosses an adapter boundary."""
        return f"plugin:{self.name}"

    def to_capability_descriptor(self) -> CapabilityDescriptor:
        """Translate the legacy plugin descriptor into the neutral contract."""
        read_only: bool | None = None
        destructive: bool | None = None
        if self.side_effect == "pure":
            read_only, destructive = True, False
        elif self.side_effect == "read":
            read_only, destructive = True, False
        elif self.side_effect == "write":
            read_only, destructive = False, False
        elif self.side_effect == "irreversible":
            read_only, destructive = False, True
        metadata = dict(self.metadata)
        metadata.setdefault("plugin_type", self.plugin_type)
        metadata.setdefault("side_effect", self.side_effect)
        if self.resource:
            metadata.setdefault("resource", self.resource)
        resource_scope = dict(self.resource_scope)
        if self.resource:
            resource_scope.setdefault("resource", self.resource)
        return CapabilityDescriptor(
            id=self.capability_id,
            name=self.name,
            description=self.description,
            kind="plugin",
            version=self.version,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema) if self.output_schema else None,
            annotations=CapabilityAnnotations(
                read_only=read_only,
                destructive=destructive,
                idempotent=self.supports_idempotency,
            ),
            execution=CapabilityExecution(
                timeout_seconds=self.timeout_s,
                supports_cancel=self.supports_cancel,
                supports_idempotency=self.supports_idempotency,
                supports_reconcile=self.supports_reconcile,
            ),
            permissions=tuple(self.required_capabilities),
            metadata=metadata,
            provenance=CapabilityProvenance(
                source="plugin",
                version=self.version,
                code_digest=self.code_digest,
                resource_digests=(digest,) if (digest := file_digest(self.resource or "")) else (),
                runtime=runtime_fingerprint(),
                dependency_digest=dependency_lock_digest(),
            ),
            resource_scope=resource_scope,
        )


def _job_method_not_implemented(*_args: Any) -> Any:
    """Protocol default for any job method the plugin did not register."""
    raise NotImplementedError("plugin has no async-job methods registered (submit/poll/cancel)")


@dataclass(frozen=True)
class JobMethods:
    """Optional asynchronous-job callables registered beside a sync handler.

    This is the ``submit(args) -> job_id`` / ``poll(job_id) -> {status,
    result?}`` / ``cancel(job_id)`` surface for hour-scale compute. Capability
    is declared by construction: a plugin that never registers
    :class:`JobMethods` keeps working exactly as before, and calling a job
    method it left at the default raises :class:`NotImplementedError` instead
    of silently misbehaving.

    Contract (see the package docstring of :mod:`drbrain.plugins` for the
    on-disk ``jobs/`` half): ``submit`` must return quickly with a ``job_id``;
    ``poll`` returns ``{"status": JobStatus | str, "result"?: Any,
    "error"?: str}``; ``cancel`` best-effort cancels and returns whether the
    cancellation was accepted.
    """

    submit: Callable[[dict[str, Any]], str] = _job_method_not_implemented
    poll: Callable[[str], dict[str, Any]] = _job_method_not_implemented
    cancel: Callable[[str], bool] = _job_method_not_implemented


@dataclass(frozen=True)
class Artifact:
    """One file a plugin call/job produced, with its content hash for audit.

    ``path`` is filesystem-relative or absolute as the plugin reports it;
    ``sha256`` lets an evidence consumer re-verify the file byte-for-byte
    instead of trusting the conversation transcript.
    """

    path: str
    sha256: str


@dataclass
class PluginResult:
    """Result envelope returned by every plugin call.

    ``resource_usage`` is additive, provider-reported accounting for a completed
    invocation.  Plugins may report ``tokens``, ``cpu_seconds``, and/or
    ``gpu_seconds``; consumers ignore unrecognised or invalid values.

    ``job_id`` / ``artifacts`` / ``truncated`` are additive job-lifecycle and
    output-hygiene fields (review 2026-09-03 P-E1 / P-I3): ``job_id`` ties a
    result to an asynchronous job, ``artifacts`` lists produced files with
    content hashes, and ``truncated`` marks that the registry clipped an
    oversized output to the ``max_output_bytes`` cap.
    """

    status: ResultStatus
    data: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    resource_usage: Mapping[str, int | float] = field(default_factory=dict)
    job_id: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.OK

    @property
    def completed(self) -> bool:
        """Whether the call completed without an infrastructure/plugin failure."""
        return self.status in {ResultStatus.OK, ResultStatus.NO_RESULT}

    def to_invocation_result(self) -> InvocationResult:
        """Expose a plugin result through the shared capability envelope."""
        status_map = {
            ResultStatus.OK: InvocationStatus.OK,
            ResultStatus.NO_RESULT: InvocationStatus.NO_RESULT,
            ResultStatus.INVALID_INPUT: InvocationStatus.INVALID_INPUT,
            ResultStatus.TIMEOUT: InvocationStatus.TIMEOUT,
            ResultStatus.MODEL_UNAVAILABLE: InvocationStatus.UNAVAILABLE,
            ResultStatus.PLUGIN_ERROR: InvocationStatus.ERROR,
        }
        return InvocationResult(
            status=status_map[self.status],
            data=self.data,
            structured_content=self.data,
            error=self.error,
            evidence=self.evidence,
            job_id=self.job_id,
            artifacts=tuple({"path": item.path, "sha256": item.sha256} for item in self.artifacts),
            truncated=self.truncated,
        )

    def to_llm_message(self, plugin: Plugin | None = None) -> str:
        """Render the result for the LLM: raw JSON + key-field summary.

        Failures are rendered as an explicit "调用失败 [status]" message so the
        agent abstains rather than hallucinating from an absent result.
        """
        if not self.ok:
            return f"插件调用失败 [{self.status.value}]: {self.error or '无输出'}"
        if self.data is None:
            return "插件调用成功但无输出。"
        summary = ""
        if plugin and plugin.summary_fields and isinstance(self.data, dict):
            parts = [f"{k}={self.data[k]}" for k in plugin.summary_fields if k in self.data]
            if parts:
                summary = " 摘要: " + ", ".join(str(p) for p in parts)
        body = json.dumps(self.data, ensure_ascii=False, default=str)
        notes: list[str] = []
        if self.truncated:
            notes.append("输出已截断(超出 max_output_bytes)")
        if self.job_id:
            notes.append(f"job_id={self.job_id}")
        if self.artifacts:
            notes.append("产物文件: " + ", ".join(a.path for a in self.artifacts))
        suffix = (" 备注: " + "；".join(notes)) if notes else ""
        return f"结果(JSON): {body}{summary}{suffix}"


def make_evidence(plugin: Plugin, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence record for a call (plugin / version / type / input / time)."""
    return {
        "plugin": plugin.name,
        "plugin_type": plugin.plugin_type,
        "version": plugin.version,
        "input": arguments,
        "input_digest": input_digest(arguments),
        "capability_id": plugin.capability_id,
        "runtime": runtime_fingerprint(),
        "code_digest": plugin.code_digest,
        "timestamp": time.time(),
    }
