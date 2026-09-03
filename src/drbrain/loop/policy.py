"""Host-owned policy for durable autoresearch tool execution.

The model may propose a tool invocation, but only this module decides whether a
workflow node may expose or execute it.  The policy is deliberately independent
of LlamaIndex, plugin discovery, and MCP transport so all tool sources share
the same fail-closed decision boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatchcase
from typing import Any, Literal

ToolSideEffect = Literal["pure", "read", "write", "irreversible", "unspecified"]


class PolicyDisposition(StrEnum):
    """The only outcomes a broker may receive from policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    WAITING_APPROVAL = "waiting_approval"


@dataclass(frozen=True)
class ToolDefinition:
    """Host-declared metadata for one executable tool surface.

    ``side_effect='unspecified'`` is intentional for legacy plugins: their
    direct ``PluginRegistry.call`` path remains compatible, while durable runs
    fail closed until the host classifies them.
    """

    name: str
    source: str
    input_schema: Mapping[str, Any]
    side_effect: ToolSideEffect = "unspecified"
    required_capabilities: tuple[str, ...] = ()
    code_digest: str = ""
    version: str = ""
    resource_scope: Mapping[str, Any] = field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    max_output_bytes: int | None = None
    cost_hint: float | None = None
    supports_idempotency: bool = False
    supports_reconcile: bool = False
    supports_cancel: bool = False
    sandbox_profile: str = ""
    approval_policy: str = "default"
    trusted: bool = False
    allowed_tools: tuple[str, ...] = ()
    timeout_s: float | None = None


@dataclass(frozen=True)
class ToolProposal:
    """A model or workflow request, before policy has authorized execution."""

    tool_call_id: str
    node_name: str
    definition: ToolDefinition
    arguments: Mapping[str, Any]
    idempotency_key: str | None = None
    approved: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    """A deterministic policy outcome recorded by :class:`ToolBroker`."""

    disposition: PolicyDisposition
    reason: str
    max_attempts: int = 1

    @property
    def allowed(self) -> bool:
        return self.disposition is PolicyDisposition.ALLOW


DEFAULT_STEP_CAPABILITIES: dict[str, frozenset[str]] = {
    "plan_task": frozenset({"graph:read", "rag:read"}),
    "retrieve": frozenset({"graph:read", "rag:read", "plugin:search_papers"}),
    "identify_gaps": frozenset({"graph:read", "rag:read"}),
    "critique": frozenset({"graph:read", "rag:read"}),
    # External model/software capabilities are deliberately absent by default:
    # a durable host must explicitly name every compute plugin it allows.
    "compute": frozenset(),
    "verify": frozenset({"graph:read", "rag:read"}),
    "report": frozenset({"graph:read", "rag:read"}),
}


class ToolPolicy:
    """Capability, side-effect, approval, and input-schema policy.

    The default table is intentionally narrow.  Hosts can pass an explicit
    ``step_capabilities`` mapping to grant a classified capability, including a
    wildcard such as ``plugin:trusted_model_*``.  No implicit catch-all exists.
    """

    def __init__(
        self,
        *,
        step_capabilities: Mapping[str, Iterable[str]] | None = None,
        max_read_attempts: int = 2,
    ) -> None:
        raw = step_capabilities if step_capabilities is not None else DEFAULT_STEP_CAPABILITIES
        self._step_capabilities = {
            str(step): frozenset(str(capability) for capability in capabilities)
            for step, capabilities in raw.items()
        }
        self._max_read_attempts = max(1, int(max_read_attempts))

    def capabilities_for(self, node_name: str) -> frozenset[str]:
        """Return the explicit tool capabilities visible to one workflow node."""
        return self._step_capabilities.get(node_name, frozenset())

    def to_manifest(self) -> dict[str, Any]:
        """Return the stable, secret-free policy contract for checkpointing."""
        return {
            "step_capabilities": {
                step: sorted(capabilities)
                for step, capabilities in sorted(self._step_capabilities.items())
            },
            "max_read_attempts": self._max_read_attempts,
        }

    def is_visible(self, *, node_name: str, definition: ToolDefinition) -> bool:
        """Whether a classified tool belongs on this node's agent surface."""
        if node_name == "verify" and definition.side_effect not in {"pure", "read"}:
            return False
        if definition.side_effect == "unspecified":
            return False
        if definition.source == "mcp" and (not definition.trusted or not definition.allowed_tools):
            return False
        return self._has_capabilities(node_name, definition)

    def evaluate(self, proposal: ToolProposal) -> PolicyDecision:
        """Validate and authorize a concrete proposal without invoking a handler."""
        definition = proposal.definition
        if proposal.node_name == "verify" and definition.side_effect not in {"pure", "read"}:
            return PolicyDecision(
                PolicyDisposition.DENY,
                "verifier is read-only and cannot modify experiment artifacts",
            )
        try:
            validate_arguments(definition.input_schema, proposal.arguments)
        except ValueError as exc:
            return PolicyDecision(PolicyDisposition.DENY, f"schema validation failed: {exc}")

        if definition.side_effect == "unspecified":
            return PolicyDecision(
                PolicyDisposition.DENY,
                "legacy tool has no side-effect classification for durable execution",
            )
        if definition.source == "mcp" and (not definition.trusted or not definition.allowed_tools):
            return PolicyDecision(
                PolicyDisposition.DENY,
                "durable MCP execution requires an explicitly trusted non-empty allowlist",
            )
        if not self._has_capabilities(proposal.node_name, definition):
            return PolicyDecision(
                PolicyDisposition.DENY,
                f"node {proposal.node_name!r} lacks capability for {definition.name!r}",
            )

        approval_required = (
            definition.side_effect == "irreversible"
            or (definition.side_effect == "write" and not definition.supports_idempotency)
            or definition.approval_policy in {"required", "always"}
        )
        if approval_required and not proposal.approved:
            return PolicyDecision(
                PolicyDisposition.WAITING_APPROVAL,
                f"{definition.side_effect} tool requires explicit approval",
            )

        attempts = self._max_read_attempts if definition.side_effect in {"pure", "read"} else 1
        return PolicyDecision(PolicyDisposition.ALLOW, "policy allowed", max_attempts=attempts)

    def _has_capabilities(self, node_name: str, definition: ToolDefinition) -> bool:
        required = definition.required_capabilities
        if not required:
            return False
        granted = self.capabilities_for(node_name)
        return all(
            any(fnmatchcase(capability, pattern) for pattern in granted) for capability in required
        )


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    """Validate the JSON-Schema subset used by graph, plugin, and MCP tools."""
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be an object")
    if not schema:
        return
    expected_type = schema.get("type")
    if expected_type not in (None, "object"):
        raise ValueError("tool input schema must describe an object")
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            raise ValueError(f"missing required field {key!r}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        extras = set(arguments) - set(properties)
        if extras:
            raise ValueError(f"unexpected fields: {', '.join(sorted(str(key) for key in extras))}")
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if isinstance(property_schema, Mapping):
            # LLM 常为可选字段补发 null(OpenAI function calling 惯例);非必填字段
            # 的 None 视同缺省,跳过类型校验(必填字段的存在性已由上面 required 检查)。
            if value is None and key not in required:
                continue
            _validate_value(property_schema, value, path=str(key))


def _validate_value(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    expected = schema.get("type")
    if expected == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{path} must be an integer")
    if expected == "number" and (not isinstance(value, int | float) or isinstance(value, bool)):
        raise ValueError(f"{path} must be a number")
    if expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    if expected == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_value(item_schema, item, path=f"{path}[{index}]")
    if expected == "object" and not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
