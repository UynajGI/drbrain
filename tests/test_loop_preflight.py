"""Static preflight contracts for durable external-tool configuration."""

from __future__ import annotations

from drbrain.loop.policy import ToolPolicy
from drbrain.loop.preflight import preflight_mcp_servers
from drbrain.rag.mcp_tools import mcp_server_id


def test_mcp_preflight_reports_visible_classified_tools_without_contacting_server():
    report = preflight_mcp_servers(
        [
            {
                "id": " catalog ",
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
    assert mcp_server_id({"id": " catalog "}) == "catalog"


def test_mcp_preflight_sorts_set_capabilities_in_its_diagnostic_report():
    report = preflight_mcp_servers(
        [
            {
                "id": "catalog",
                "command": "catalog-mcp",
                "trusted": True,
                "allowed_tools": ["search"],
                "side_effect": "read",
                "required_capabilities": {"beta", "alpha"},
            }
        ],
        tool_policy=ToolPolicy(step_capabilities={"retrieve": {"alpha", "beta"}}),
    )

    assert report["servers"][0]["tools"][0]["required_capabilities"] == ["alpha", "beta"]


def test_mcp_preflight_uses_runtime_fallback_id_for_invalid_server_config():
    report = preflight_mcp_servers(
        [{"trusted": True, "allowed_tools": ["search"], "side_effect": "read"}],
        tool_policy=ToolPolicy(step_capabilities={}),
    )

    assert report["servers"] == [
        {
            "server_id": "mcp",
            "status": "blocked",
            "side_effect": "read",
            "issues": ["MCP server command is required"],
            "tools": [],
        }
    ]
