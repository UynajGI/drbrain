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
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

# Coarse capability category. Concrete implementation details (e.g. ``gbdt`` vs
# ``gnn``) belong in :attr:`Plugin.metadata`, not in this strict type.
PluginType = Literal["model", "software", "data", "formula", "other"]
Backend = Literal["subprocess", "inprocess", "static"]
OnFailure = Literal["abstain", "stale", "none"]
PluginSideEffect = Literal["pure", "read", "write", "irreversible", "unspecified"]


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


@dataclass(frozen=True)
class Plugin:
    """Generic descriptor of an external capability exposed as an agent tool.

    Frozen: a registered plugin is immutable. ``summary_fields`` names the keys
    whose values are worth surfacing to the LLM alongside the raw JSON (the
    "raw JSON + key-field summary" compromise).

    ``plugin_type`` is a coarse capability category (``model`` / ``software`` /
    ``data`` / ``formula`` / ``other``). ``resource`` is the single path/pointer
    to the plugin's artifact — a weights file, a software binary, or a data
    file, depending on ``plugin_type``. Free-form specifics go in ``metadata``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    plugin_type: PluginType = "other"
    version: str = ""
    resource: str | None = None
    backend: Backend = "inprocess"
    entry: str = ""
    on_failure: OnFailure = "abstain"
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


@dataclass
class PluginResult:
    """Result envelope returned by every plugin call."""

    status: ResultStatus
    data: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.OK

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
        return f"结果(JSON): {body}{summary}"


def make_evidence(plugin: Plugin, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence record for a call (plugin / version / type / input / time)."""
    return {
        "plugin": plugin.name,
        "plugin_type": plugin.plugin_type,
        "version": plugin.version,
        "input": arguments,
        "timestamp": time.time(),
    }
