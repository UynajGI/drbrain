"""Trusted MCP bridge for the RAG agent.

Low-level discovery and invocation remain available for local administrative
scripts. Agent callers can opt into a strict mode where each server must be
marked trusted and pin the tools it grants to the model. The model never
chooses a command, environment, server, or tool outside that host-owned policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from drbrain.capabilities import (
    CapabilityAnnotations,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityProvenance,
    InvocationResult,
    InvocationStatus,
    descriptor_id,
    runtime_fingerprint,
)

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 300.0


class MCPTrustError(ValueError):
    """The host configuration does not grant a requested MCP capability."""


class MCPTimeoutError(TimeoutError):
    """A bounded MCP discovery or invocation exceeded its configured timeout."""


@dataclass(frozen=True)
class MCPServerPolicy:
    """Validated host-owned policy for one MCP server and transport."""

    server_id: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] | None
    allowed_tools: frozenset[str] | None
    timeout_seconds: float
    transport: str = "stdio"
    url: str | None = None
    headers: dict[str, str] | None = None


def validate_mcp_server(
    server: dict[str, Any], *, require_trusted: bool = False
) -> MCPServerPolicy:
    """Validate one host-owned MCP server configuration without contacting it."""
    return _policy_from_server(server, require_trusted=require_trusted)


def mcp_server_id(server: Mapping[str, Any], *, fallback: str = "mcp") -> str:
    """Normalize the host-owned identifier used in MCP tool capabilities."""
    return str(server.get("id") or server.get("name") or server.get("command") or fallback).strip()


def normalize_mcp_strings(value: Any) -> tuple[str, ...]:
    """Normalize host-owned MCP metadata while making unordered inputs stable."""
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    values = [str(item).strip() for item in value if str(item).strip()]
    return tuple(sorted(values)) if isinstance(value, (set, frozenset)) else tuple(values)


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
    transport = str(
        server.get("transport") or ("streamable_http" if server.get("url") else "stdio")
    )
    transport = transport.strip().lower().replace("-", "_")
    if transport in {"http", "streamablehttp", "streamable_http"}:
        transport = "streamable_http"
    elif transport != "stdio":
        raise MCPTrustError(f"unsupported MCP transport: {transport!r}")
    command = str(server.get("command") or "").strip()
    url = str(server.get("url") or "").strip() or None
    if transport == "stdio" and not command:
        # Keep the established diagnostic stable for preflight/audit clients.
        raise MCPTrustError("MCP server command is required")
    if transport == "streamable_http" and not url:
        raise MCPTrustError("MCP server url is required for Streamable HTTP transport")
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
    raw_headers = server.get("headers")
    if raw_headers is not None and (
        not isinstance(raw_headers, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_headers.items()
        )
    ):
        raise MCPTrustError("MCP server headers must be a string-to-string mapping")
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

    server_id = mcp_server_id(server, fallback=command or url or "mcp")
    return MCPServerPolicy(
        server_id=server_id,
        command=command,
        args=tuple(raw_args),
        env=dict(raw_env) if raw_env is not None else None,
        allowed_tools=allowed_tools,
        timeout_seconds=timeout_seconds,
        transport=transport,
        url=url,
        headers=dict(raw_headers) if raw_headers is not None else None,
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
    server: dict[str, Any], *, require_trusted: bool = False, namespace: bool = False
) -> list[dict[str, Any]]:
    """Discover a server's allowed tools within its configured timeout.

    ``require_trusted=False`` preserves the direct local-administration API.
    Agent assembly uses ``True`` so only explicitly trusted, allowlisted tools
    reach the model's tool surface.
    """
    return _run_coro(
        _discover(_policy_from_server(server, require_trusted=require_trusted), namespace=namespace)
    )


@contextlib.asynccontextmanager
async def _session(policy: MCPServerPolicy):
    """Open either supported MCP transport behind one adapter boundary."""
    from mcp import ClientSession

    try:
        if policy.transport == "stdio":
            from mcp import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=policy.command, args=list(policy.args), env=policy.env
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

            http_client = create_mcp_http_client(headers=policy.headers)
            try:
                async with streamable_http_client(policy.url or "", http_client=http_client) as (
                    read,
                    write,
                ):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            finally:
                await http_client.aclose()
    except TimeoutError as exc:
        raise MCPTimeoutError(f"MCP operation timed out for {policy.server_id!r}") from exc


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


async def _discover(policy: MCPServerPolicy, *, namespace: bool = False) -> list[dict[str, Any]]:
    from mcp import types

    descriptors: list[dict[str, Any]] = []
    cursor: str | None = None
    async with asyncio.timeout(policy.timeout_seconds):
        async with _session(policy) as session:
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                result = await session.list_tools(params=params)
                for tool in result.tools:
                    if policy.allowed_tools is not None and tool.name not in policy.allowed_tools:
                        log.info(
                            "[mcp] tool %s from %s excluded by allowlist",
                            tool.name,
                            policy.server_id,
                        )
                        continue
                    canonical_name = descriptor_id(f"mcp:{policy.server_id}", str(tool.name))
                    descriptor = {
                        "id": canonical_name,
                        "canonicalName": canonical_name if namespace else str(tool.name),
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": _jsonable(
                            getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {})
                        ),
                    }
                    output_schema = getattr(tool, "outputSchema", None) or getattr(
                        tool, "output_schema", None
                    )
                    annotations = getattr(tool, "annotations", None)
                    meta = getattr(tool, "meta", None) or getattr(tool, "_meta", None)
                    if output_schema is not None:
                        descriptor["outputSchema"] = _jsonable(output_schema)
                    if annotations is not None:
                        descriptor["annotations"] = _jsonable(annotations)
                    if meta is not None:
                        descriptor["_meta"] = _jsonable(meta)
                    descriptors.append(descriptor)
                cursor = getattr(result, "nextCursor", None) or getattr(result, "next_cursor", None)
                if not cursor:
                    break
    return descriptors


def _call_result_from_mcp(result: Any, policy: MCPServerPolicy, tool_name: str) -> InvocationResult:
    content = tuple(
        item if isinstance(item, dict) else _jsonable(item)
        for item in (_jsonable(getattr(result, "content", None) or []) or [])
    )
    content = tuple(item if isinstance(item, dict) else {"value": item} for item in content)
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", False)
    status = (
        InvocationStatus.ERROR
        if is_error
        else (
            InvocationStatus.OK if structured is not None or content else InvocationStatus.NO_RESULT
        )
    )
    error = None
    if is_error:
        error = "MCP tool returned isError=true"
    task = getattr(result, "task", None) or getattr(result, "task_id", None)
    task_id = getattr(task, "taskId", None) or getattr(task, "task_id", None) or task
    if task_id is not None:
        task_id = str(task_id)
    return InvocationResult(
        status=status,
        data=structured,
        content=content,
        structured_content=structured,
        error=error,
        job_id=task_id,
        evidence={
            "capability_id": descriptor_id(f"mcp:{policy.server_id}", tool_name),
            "server_id": policy.server_id,
            "tool": tool_name,
            "transport": policy.transport,
            "isError": bool(is_error),
            "runtime": runtime_fingerprint(),
        },
    )


def call_mcp_tool(
    server: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    require_trusted: bool = False,
) -> str:
    """Call one allowed MCP tool within the host-configured timeout."""
    policy = _policy_from_server(server, require_trusted=require_trusted)
    _require_allowed_tool(policy, tool_name)
    result = _run_coro(_call(policy, tool_name, arguments))
    if result.status is InvocationStatus.ERROR:
        return result.error or "MCP tool failed"
    parts = []
    for item in result.content:
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        elif "value" in item:
            parts.append(str(item["value"]))
        else:
            parts.append(str(item))
    if parts:
        return "\n".join(parts)
    if result.structured_content is not None:
        import json

        return json.dumps(result.structured_content, ensure_ascii=False, default=str)
    return ""


def call_mcp_tool_result(
    server: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    require_trusted: bool = False,
) -> InvocationResult:
    """Call an MCP tool while preserving structured content and error flags."""
    policy = _policy_from_server(server, require_trusted=require_trusted)
    _require_allowed_tool(policy, tool_name)
    return _run_coro(_call(policy, tool_name, arguments))


async def _call(
    policy: MCPServerPolicy, tool_name: str, arguments: dict[str, Any]
) -> InvocationResult:
    try:
        async with asyncio.timeout(policy.timeout_seconds):
            async with _session(policy) as session:
                result = await session.call_tool(tool_name, arguments)
    except TimeoutError as exc:
        raise MCPTimeoutError(f"MCP tool {tool_name!r} timed out for {policy.server_id!r}") from exc
    return _call_result_from_mcp(result, policy, tool_name)


def load_mcp_tools(
    servers: list[dict[str, Any]] | None,
    *,
    require_trusted: bool = False,
    call_override: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], bool], Any]
    | None = None,
    include: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    namespace: bool = False,
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
    seen_names: set[str] = set()
    for server in servers:
        try:
            descriptors = discover_mcp_tools(
                server, require_trusted=require_trusted, namespace=namespace
            )
        except Exception as exc:  # noqa: BLE001 — a bad server must not break assembly
            label = server.get("id") or server.get("name") or server.get("command")
            log.warning(
                "[mcp] discovery failed (trusted=%s) for %r: %s", require_trusted, label, exc
            )
            continue
        for descriptor in descriptors:
            if include is not None and not include(server, descriptor):
                continue
            if (
                descriptor["name"] in seen_names
                and descriptor.get("canonicalName") == descriptor["name"]
            ):
                descriptor = {
                    **descriptor,
                    "canonicalName": descriptor.get("id", descriptor["name"]),
                }
            seen_names.add(str(descriptor["name"]))
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
        name=str(descriptor.get("canonicalName") or descriptor["name"]),
        description=descriptor.get("description") or "",
        fn_schema=model,
    )


def mcp_descriptor_to_capability(
    server: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> CapabilityDescriptor:
    """Translate a discovered MCP tool into the neutral capability model."""
    server_id = mcp_server_id(server)
    name = str(descriptor.get("name") or "").strip()
    canonical_name = str(descriptor.get("id") or descriptor_id(f"mcp:{server_id}", name))
    raw_annotations = descriptor.get("annotations")
    annotations = CapabilityAnnotations.from_dict(_jsonable(raw_annotations))
    return CapabilityDescriptor(
        id=canonical_name,
        name=name,
        description=str(descriptor.get("description") or ""),
        kind="mcp_tool",
        version=str(server.get("version") or ""),
        input_schema=dict(descriptor.get("inputSchema") or {}),
        output_schema=(
            dict(descriptor["outputSchema"])
            if isinstance(descriptor.get("outputSchema"), dict)
            else None
        ),
        annotations=annotations,
        execution=CapabilityExecution(
            timeout_seconds=float(server["timeout_seconds"])
            if isinstance(server.get("timeout_seconds"), (int, float))
            and not isinstance(server.get("timeout_seconds"), bool)
            else None,
            supports_cancel=server.get("supports_cancel") is True,
            supports_idempotency=server.get("supports_idempotency") is True,
            supports_reconcile=server.get("supports_reconcile") is True,
        ),
        permissions=tuple(normalize_mcp_strings(server.get("required_capabilities"))),
        metadata={
            "server_id": server_id,
            "annotations": _jsonable(raw_annotations),
            "_meta": _jsonable(descriptor.get("_meta")),
        },
        provenance=CapabilityProvenance(
            source=f"mcp:{server_id}",
            version=str(server.get("version") or ""),
            code_digest=str(server.get("code_digest") or ""),
            runtime=runtime_fingerprint(),
        ),
        resource_scope={
            "server_id": server_id,
            "transport": str(server.get("transport") or "stdio"),
        },
    )


def mcp_capability_catalog(servers: list[dict[str, Any]], *, require_trusted: bool = True) -> Any:
    """Return a shared catalog containing all discoverable MCP tools."""
    from drbrain.capabilities import CapabilityCatalog

    catalog = CapabilityCatalog()
    catalog.register_mcp_servers(servers, require_trusted=require_trusted)
    return catalog


def _schema_to_model(name: str, schema: dict[str, Any]) -> type | None:
    """Build a pydantic model from a JSON Schema for ``FunctionTool.fn_schema``.

    Delegates to the shared plugins-layer bridge so MCP tools get the same
    parameter constraints as plugin tools: per-parameter description/enum/
    default, nested objects recursed, ``additionalProperties: false``.
    """
    from drbrain.plugins.registry import json_schema_to_model

    return json_schema_to_model(name, schema)
