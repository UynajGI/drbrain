"""Model-as-Tool protocol — generic abstraction for exposing trained models as agent tools.

Similar to MCP's *tool* abstraction (``name + description + JSON Schema + call``)
but deliberately **not** MCP's transport/resource layer. This protocol adds what
a plain tool schema cannot express:

* **model metadata** — ``model_type`` / ``model_version`` / ``weights``, so the
  agent and the audit trail know *which* artifact produced a result;
* **degradation semantics** — a :class:`ModelResult` status wired to the
  Epistemic Layer (:mod:`drbrain.rag.status`), so a broken model leads to
  ``abstain`` instead of a hallucinated answer;
* **evidence provenance** — every result carries tool name / model version /
  input / timestamp, ready to be persisted into ``answer_records`` / ``claims``
  / ``evidence``.

The protocol is **model-agnostic**: this module never imports or names a
concrete model. Today's flat-band predictor and tomorrow's superconductivity
predictor are both just :class:`ModelTool` registrations — the protocol does
not change when the models change.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

ModelType = Literal["gbdt", "gnn", "embedding", "validator", "formula", "llm_route", "other"]
Backend = Literal["cli", "inprocess", "static"]
OnFailure = Literal["abstain", "stale", "none"]


class ResultStatus(StrEnum):
    """Machine-readable outcome of one model-tool call (degradation-aware).

    Mirrors :class:`drbrain.rag.status.RetrievalStatus` semantics, narrowed to
    model calls: ``OK`` / ``NO_RESULT`` are usable, the rest are failures the
    caller must not treat as an answer.
    """

    OK = "ok"
    NO_RESULT = "no_result"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class ModelTool:
    """Generic descriptor of a model exposed as an agent tool.

    Frozen: a registered tool is immutable. ``summary_fields`` names the keys
    whose values are worth surfacing to the LLM alongside the raw JSON (the
    "raw JSON + key-field summary" compromise).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    model_type: ModelType = "other"
    model_version: str = ""
    weights: str | None = None
    backend: Backend = "inprocess"
    entry: str = ""
    on_failure: OnFailure = "abstain"
    timeout_s: float = 60.0
    summary_fields: tuple[str, ...] = ()


@dataclass
class ModelResult:
    """Result envelope returned by every model-tool call."""

    status: ResultStatus
    data: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.OK

    def to_llm_message(self, tool: ModelTool | None = None) -> str:
        """Render the result for the LLM: raw JSON + key-field summary.

        Failures are rendered as an explicit "调用失败 [status]" message so the
        agent abstains rather than hallucinating from an absent result.
        """
        if not self.ok:
            return f"模型调用失败 [{self.status.value}]: {self.error or '无输出'}"
        if self.data is None:
            return "模型调用成功但无输出。"
        summary = ""
        if tool and tool.summary_fields and isinstance(self.data, dict):
            parts = [f"{k}={self.data[k]}" for k in tool.summary_fields if k in self.data]
            if parts:
                summary = " 摘要: " + ", ".join(str(p) for p in parts)
        body = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"结果(JSON): {body}{summary}"


def make_evidence(tool: ModelTool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence record for a call (tool / version / type / input / time)."""
    return {
        "tool": tool.name,
        "model_type": tool.model_type,
        "model_version": tool.model_version,
        "input": arguments,
        "timestamp": time.time(),
    }
