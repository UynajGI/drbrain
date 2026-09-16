"""The index-model proxy reads request bodies and forwards headers safely.

Regression (PR #77 review): the parser looked up ``Content-Length``
case-sensitively, so a lowercase header dropped the body, and hop-by-hop
headers (Connection, Transfer-Encoding, …) were forwarded upstream.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "serve_llm_proxy", _ROOT / "scripts" / "serve_llm_proxy.py"
)
assert _SPEC is not None and _SPEC.loader is not None
proxy = importlib.util.module_from_spec(_SPEC)
sys.modules["serve_llm_proxy"] = proxy
_SPEC.loader.exec_module(proxy)


def _read(payload: bytes) -> tuple[Any, ...]:
    async def run() -> tuple[Any, ...]:
        reader = asyncio.StreamReader()
        reader.feed_data(payload)
        reader.feed_eof()
        return await proxy._read_request(reader)

    return asyncio.run(run())


def _request(content_length_header: str) -> bytes:
    return (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1:8099\r\n"
        f"{content_length_header}\r\n"
        "\r\n"
        "hello"
    ).encode("latin-1")


def test_lowercase_content_length_still_reads_the_body() -> None:
    method, _path, headers, body = _read(_request("content-length: 5"))
    assert method == "POST"
    assert headers["content-length"] == "5"
    assert body == b"hello"


def test_canonical_content_length_still_reads_the_body() -> None:
    *_head, body = _read(_request("Content-Length: 5"))
    assert body == b"hello"


def test_forward_headers_drop_the_hop_by_hop_set_and_the_connection_list() -> None:
    headers = {
        "Host": "127.0.0.1:8099",
        "Content-Type": "application/json",
        "Connection": "keep-alive, X-Internal",
        "X-Internal": "drop-me",
        "Keep-Alive": "timeout=5",
        "Transfer-Encoding": "chunked",
        "Authorization": "Bearer local",
    }
    assert proxy._forward_headers(headers) == {
        "Content-Type": "application/json",
        "Authorization": "Bearer local",
    }
