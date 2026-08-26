"""Trusted stdio-MCP bridge for the RAG agent.

Low-level discovery and invocation remain available for local administrative
scripts. Agent callers can opt into a strict mode where each server must be
marked trusted and pin the tools it grants to the model. The model never
chooses a command, environment, server, or tool outside that host-owned policy.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 300.0


class MCPTrustError(ValueError):
    """The host configuration does not grant a requested MCP capability."""


class MCPTimeoutError(TimeoutError):
    """A bounded MCP discovery or invocation exceeded its configured timeout."""


@dataclass(frozen=True)
class MCPServerPolicy:
    """Validated host-owned policy for one stdio MCP server."""

    server_id: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] | None
    allowed_tools: frozenset[str] | None
    timeout_seconds: float


def validate_mcp_server(
    server: dict[str, Any], *, require_trusted: bool = False
) -> MCPServerPolicy:
    """Validate one host-owned MCP server configuration without contacting it."""
    return _policy_from_server(server, require_trusted=require_trusted)


def mcp_server_id(server: Mapping[str, Any], *, fallback: str = "mcp") -> str:
    """Normalize the host-owned identifier used in MCP tool capabilities."""
    return str(server.get("id") or server.get("name") or server.get("command") or fallback).strip()


def _policy_from_server(
    server: dict[str, Any], *, require_trusted: bool = False
) -> MCPServerPolicy:
    """Validate a server dict and derive its immutable execution policy.

    Legacy direct calls may omit the trust fields; agent assembly passes
    ``require_trusted=True`` and therefore fails closed unless the application
    explicitly marks the server trusted and provides a non-empty tool allowlist.
    """
    if not isinstance(server, dict):
        raise MCPTrustError("MCP server configuration must be a dict")
    command = str(server.get("command") or "").strip()
    if not command:
        raise MCPTrustError("MCP server command is required")
    raw_args = server.get("args") or []
    if not isinstance(raw_args, (list, tuple)) or not all(isinstance(arg, str) for arg in raw_args):
        raise MCPTrustError("MCP server args must be a list of strings")
    raw_env = server.get("env")
    if raw_env is not None and (
        not isinstance(raw_env, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_env.items()
        )
    ):
        raise MCPTrustError("MCP server env must be a string-to-string mapping")
    raw_timeout = server.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(raw_timeout, bool):
        raise MCPTrustError("MCP timeout_seconds must be a positive number")
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise MCPTrustError("MCP timeout_seconds must be a positive number") from exc
    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise MCPTrustError(f"MCP timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS:g}] seconds")

    raw_allowed = server.get("allowed_tools")
    if raw_allowed is None:
        allowed_tools = None
    elif isinstance(raw_allowed, (list, tuple, set)) and all(
        isinstance(name, str) and name.strip() for name in raw_allowed
    ):
        allowed_tools = frozenset(name.strip() for name in raw_allowed)
    else:
        raise MCPTrustError("MCP allowed_tools must be a collection of non-empty strings")

    if require_trusted:
        if server.get("trusted") is not True:
            raise MCPTrustError("MCP server is not explicitly trusted")
        if not allowed_tools:
            raise MCPTrustError("Trusted MCP servers require a non-empty allowed_tools list")

    server_id = mcp_server_id(server, fallback=command)
    return MCPServerPolicy(
        server_id=server_id,
        command=command,
        args=tuple(raw_args),
        env=dict(raw_env) if raw_env is not None else None,
        allowed_tools=allowed_tools,
        timeout_seconds=timeout_seconds,
    )


def _require_allowed_tool(policy: MCPServerPolicy, tool_name: str) -> None:
    if policy.allowed_tools is not None and tool_name not in policy.allowed_tools:
        raise MCPTrustError(
            f"MCP tool {tool_name!r} is not allowed for server {policy.server_id!r}"
        )


def _run_coro(coro: Any) -> Any:
    """Run a coroutine, tolerating an already-running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def discover_mcp_tools(
    server: dict[str, Any], *, require_trusted: bool = False
) -> list[dict[str, Any]]:
    """Discover a server's allowed tools within its configured timeout.

    ``require_trusted=False`` preserves the direct local-administration API.
    Agent assembly uses ``True`` so only explicitly trusted, allowlisted tools
    reach the model's tool surface.
    """
    return _run_coro(_discover(_policy_from_server(server, require_trusted=require_trusted)))


async def _discover(policy: MCPServerPolicy) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(command=policy.command, args=list(policy.args), env=policy.env)
    try:
        async with asyncio.timeout(policy.timeout_seconds):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
    except TimeoutError as exc:
        raise MCPTimeoutError(f"MCP discovery timed out for {policy.server_id!r}") from exc
    descriptors: list[dict[str, Any]] = []
    for tool in result.tools:
        if policy.allowed_tools is not None and tool.name not in policy.allowed_tools:
            log.info("[mcp] tool %s from %s excluded by allowlist", tool.name, policy.server_id)
            continue
        descriptors.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.input_schema,
            }
        )
    return descriptors


def call_mcp_tool(
    server: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    require_trusted: bool = False,
) -> str:
    """Call one allowed stdio MCP tool within the host-configured timeout."""
    policy = _policy_from_server(server, require_trusted=require_trusted)
    _require_allowed_tool(policy, tool_name)
    return _run_coro(_call(policy, tool_name, arguments))


async def _call(policy: MCPServerPolicy, tool_name: str, arguments: dict[str, Any]) -> str:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(command=policy.command, args=list(policy.args), env=policy.env)
    try:
        async with asyncio.timeout(policy.timeout_seconds):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
    except TimeoutError as exc:
        raise MCPTimeoutError(f"MCP tool {tool_name!r} timed out for {policy.server_id!r}") from exc
    parts: list[str] = []
    for item in result.content:
        if getattr(item, "type", "") == "text":
            parts.append(getattr(item, "text", "") or "")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def load_mcp_tools(
    servers: list[dict[str, Any]] | None,
    *,
    require_trusted: bool = False,
    call_override: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], bool], Any]
    | None = None,
    include: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> list:
    """Bridge configured MCP tools, optionally requiring explicit trust.

    The default preserves historic local configuration behavior. Production
    hosts pass ``require_trusted=True``; then every server must use
    ``trusted: true`` plus a non-empty ``allowed_tools`` list. An allowlist,
    when supplied, is enforced even in compatibility mode. ``include`` and
    ``call_override`` are optional durable-loop hooks; omitting them preserves
    direct MCP execution.
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
            descriptors = discover_mcp_tools(server, require_trusted=require_trusted)
        except Exception as exc:  # noqa: BLE001 — a bad server must not break assembly
            label = server.get("id") or server.get("name") or server.get("command")
            log.warning(
                "[mcp] discovery failed (trusted=%s) for %r: %s", require_trusted, label, exc
            )
            continue
        for descriptor in descriptors:
            if include is not None and not include(server, descriptor):
                continue
            tools.append(
                _to_function_tool(
                    server,
                    descriptor,
                    FunctionTool,
                    require_trusted,
                    call_override=call_override,
                )
            )
    return tools


def _to_function_tool(
    server: dict[str, Any],
    descriptor: dict[str, Any],
    function_tool_cls: Any,
    require_trusted: bool,
    *,
    call_override: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], bool], Any]
    | None = None,
) -> Any:
    schema = descriptor.get("inputSchema") or {}
    model = _schema_to_model(descriptor["name"], schema)

    fn: Callable[..., Any]
    if call_override is not None:

        async def _brokered_fn(**kwargs: Any) -> str:
            result = call_override(server, descriptor, dict(kwargs), require_trusted)
            if inspect.isawaitable(result):
                result = await result
            return str(result)

        fn = _brokered_fn

    else:

        def _direct_fn(**kwargs: Any) -> str:
            return call_mcp_tool(
                server, descriptor["name"], dict(kwargs), require_trusted=require_trusted
            )

        fn = _direct_fn

    return function_tool_cls.from_defaults(
        fn=fn,
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
    json_to_py: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    fields: dict[str, Any] = {}
    for key, prop in properties.items():
        ptype = json_to_py.get(prop.get("type"), Any)
        fields[key] = (ptype, ...) if key in required else (ptype | None, None)
    return create_model(f"mcp_{name}_Input", **fields)
