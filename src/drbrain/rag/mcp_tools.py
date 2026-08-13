"""Generic MCP bridge — connect to *any* MCP (stdio) server, discover + call its tools.

The agent's "connect to anything" capability, mirroring how Claude Code natively
consumes MCP servers. Servers are described as dicts (``command`` / ``args`` /
``env``); this module knows only stdio (the common local case) and never names a
specific server — a server is just configuration, discovered and called at
runtime.

Session lifetime: these helpers connect/disconnect per operation (correct, not
high-throughput). A persistent session manager that keeps servers alive for a
whole agent run is a follow-up; the discover/call contract stays identical.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


def discover_mcp_tools(server: dict[str, Any]) -> list[dict[str, Any]]:
    """Connect to a stdio MCP server and return its tool descriptors.

    Returns ``[{"name", "description", "inputSchema"}, ...]``; raises on a
    connection/transport failure so callers can decide to skip that server.
    """
    return asyncio.run(_discover(server))


async def _discover(server: dict[str, Any]) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=server["command"],
        args=list(server.get("args") or []),
        env=server.get("env"),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.input_schema,
                }
                for tool in result.tools
            ]


def call_mcp_tool(
    server: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Connect to a stdio MCP server, call one tool, return its text output."""
    return asyncio.run(_call(server, tool_name, arguments))


async def _call(
    server: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=server["command"],
        args=list(server.get("args") or []),
        env=server.get("env"),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            parts: list[str] = []
            for item in result.content:
                if getattr(item, "type", "") == "text":
                    parts.append(getattr(item, "text", "") or "")
                else:
                    parts.append(str(item))
            return "\n".join(parts)


def load_mcp_tools(servers: list[dict[str, Any]] | None) -> list:
    """Bridge every tool from every MCP server to a LlamaIndex ``FunctionTool``.

    Graceful: a server that fails to connect is skipped (one bad server never
    aborts the rest). Each tool's handler reconnects per call — correct, but
    see the module docstring on session lifetime for the high-throughput path.
    """
    if not servers:
        return []
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        return []

    tools: list = []
    for server in servers:
        try:
            descriptors = discover_mcp_tools(server)
        except Exception as exc:  # noqa: BLE001 — a bad server must not break assembly
            log.warning("[mcp] discover failed for %r: %s", server.get("command"), exc)
            continue
        for descriptor in descriptors:
            tools.append(_to_function_tool(server, descriptor, FunctionTool))
    return tools


def _to_function_tool(
    server: dict[str, Any],
    descriptor: dict[str, Any],
    function_tool_cls: Any,
) -> Any:
    schema = descriptor.get("inputSchema") or {}
    model = _schema_to_model(descriptor["name"], schema)

    def _fn(**kwargs: Any) -> str:
        return call_mcp_tool(server, descriptor["name"], dict(kwargs))

    return function_tool_cls.from_defaults(
        fn=_fn,
        name=descriptor["name"],
        description=descriptor.get("description") or "",
        fn_schema=model,
    )


def _schema_to_model(name: str, schema: dict[str, Any]) -> type | None:
    """Build a pydantic model from a JSON Schema for ``FunctionTool.fn_schema``."""
    try:
        from pydantic import create_model
    except ImportError:
        return None
    properties = schema.get("properties") or {}
    if schema.get("type") != "object" or not properties:
        return None
    required = set(schema.get("required") or [])
    _json_to_py: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    fields: dict[str, Any] = {}
    for key, prop in properties.items():
        ptype = _json_to_py.get(prop.get("type"), Any)
        fields[key] = (ptype, ...) if key in required else (ptype | None, None)
    return create_model(f"mcp_{name}_Input", **fields)
