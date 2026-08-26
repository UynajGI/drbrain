"""Static diagnostics for durable external-tool configuration."""

from __future__ import annotations

from typing import Any, cast

from drbrain.loop.policy import ToolDefinition, ToolPolicy, ToolSideEffect
from drbrain.rag.mcp_tools import mcp_server_id, validate_mcp_server

_DURABLE_SIDE_EFFECTS = frozenset({"pure", "read", "write", "irreversible"})


def preflight_mcp_servers(
    servers: list[dict[str, Any]] | None, *, tool_policy: ToolPolicy
) -> dict[str, Any]:
    """Explain whether configured MCP tools can enter a durable agent surface.

    This is deliberately static: it validates the host configuration and policy
    visibility but never starts an MCP process or discovers remote tools.
    """
    reports: list[dict[str, Any]] = []
    policy_steps = sorted(tool_policy.to_manifest()["step_capabilities"])
    for index, server in enumerate(servers or []):
        if not isinstance(server, dict):
            reports.append(
                {
                    "server_id": f"server-{index + 1}",
                    "status": "blocked",
                    "side_effect": "unspecified",
                    "issues": ["server configuration must be a mapping"],
                    "tools": [],
                }
            )
            continue
        server_id = mcp_server_id(server, fallback=f"server-{index + 1}")
        try:
            policy = validate_mcp_server(server, require_trusted=True)
        except ValueError as exc:
            reports.append(
                {
                    "server_id": server_id,
                    "status": "blocked",
                    "side_effect": str(server.get("side_effect") or "unspecified"),
                    "issues": [str(exc)],
                    "tools": [],
                }
            )
            continue

        side_effect = str(server.get("side_effect") or "unspecified")
        if side_effect not in _DURABLE_SIDE_EFFECTS:
            reports.append(
                {
                    "server_id": policy.server_id,
                    "status": "blocked",
                    "side_effect": side_effect,
                    "issues": ["side_effect must be classified for durable use"],
                    "tools": [],
                }
            )
            continue

        common_capabilities = _string_tuple(server.get("required_capabilities"))
        tool_reports: list[dict[str, Any]] = []
        for tool_name in sorted(policy.allowed_tools or ()):
            capabilities = common_capabilities or (f"mcp:{policy.server_id}:{tool_name}",)
            definition = ToolDefinition(
                name=tool_name,
                source="mcp",
                input_schema={},
                side_effect=cast(ToolSideEffect, side_effect),
                required_capabilities=capabilities,
                trusted=True,
                allowed_tools=tuple(policy.allowed_tools or ()),
            )
            visible_steps = [
                step
                for step in policy_steps
                if tool_policy.is_visible(node_name=step, definition=definition)
            ]
            tool_reports.append(
                {
                    "name": tool_name,
                    "required_capabilities": list(capabilities),
                    "visible_steps": visible_steps,
                }
            )
        issues = []
        if not any(tool["visible_steps"] for tool in tool_reports):
            issues.append("no allowed tool is visible to a workflow step")
        reports.append(
            {
                "server_id": policy.server_id,
                "status": "ready" if not issues else "blocked",
                "side_effect": side_effect,
                "issues": issues,
                "tools": tool_reports,
            }
        )

    return {
        "configured_servers": len(reports),
        "ready_servers": sum(report["status"] == "ready" for report in reports),
        "blocked_servers": sum(report["status"] == "blocked" for report in reports),
        "servers": reports,
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    values = [str(item).strip() for item in value if str(item).strip()]
    return tuple(sorted(values)) if isinstance(value, (set, frozenset)) else tuple(values)
