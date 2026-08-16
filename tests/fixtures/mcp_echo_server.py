"""Minimal stdio MCP server for testing the generic MCP bridge (echo tool).

Spawned as a subprocess by ``StdioServerParameters(command=sys.executable,
args=[this_file])`` in tests/test_mcp_tools.py. Uses the v2 ``MCPServer``
high-level API (the ``@app.tool()`` decorator infers name/description/schema
from the function signature).
"""

from __future__ import annotations

import asyncio

from mcp.server import MCPServer

app = MCPServer(name="echo-server")


@app.tool()
async def echo(text: str) -> str:
    """Echo back the text argument"""
    return f"echo: {text}"


if __name__ == "__main__":
    asyncio.run(app.run_stdio_async())
