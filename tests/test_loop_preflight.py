"""Static preflight contracts for durable external-tool configuration."""

from __future__ import annotations

from drbrain.loop.policy import ToolPolicy
from drbrain.loop.preflight import preflight_mcp_servers


def test_mcp_preflight_reports_visible_classified_tools_without_contacting_server():
    report = preflight_mcp_servers(
        [
            {
                "id": "catalog",
                "command": "catalog-mcp",
                "trusted": True,
                "allowed_tools": ["search"],
                "side_effect": "read",
            }
        ],
        tool_policy=ToolPolicy(step_capabilities={"retrieve": {"mcp:catalog:search"}}),
    )

    assert report == {
        "configured_servers": 1,
        "ready_servers": 1,
        "blocked_servers": 0,
        "servers": [
            {
                "server_id": "catalog",
                "status": "ready",
                "side_effect": "read",
                "issues": [],
                "tools": [
                    {
                        "name": "search",
                        "required_capabilities": ["mcp:catalog:search"],
                        "visible_steps": ["retrieve"],
                    }
                ],
            }
        ],
    }
