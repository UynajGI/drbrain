#!/usr/bin/env python3
"""Round-robin proxy in front of several local index-model instances.

``drbrain`` resolves one ``base_url`` per role, and the endpoint's
``max_concurrent`` gate caps in-flight calls.  Running three model instances
on three GPUs therefore needs one address that fans out — this proxy picks
the upstream with the fewest in-flight requests (ties broken round-robin)
and forwards the request with proxy-safe headers.

Usage::

    python scripts/serve_llm_proxy.py 8099 8010 8011 8012

The client is configured with ``base_url: http://127.0.0.1:8099/v1`` and a
``max_concurrent`` equal to the total per-instance allowance (e.g. 6 for
three instances at two each).  ``GET /healthz`` reports per-upstream
in-flight counts; every request logs one line with its upstream and status.
"""

from __future__ import annotations

import asyncio
import http.client
import itertools
import sys
import time

HOST = "127.0.0.1"
UPSTREAM_TIMEOUT = 900.0
MAX_HEADER_BYTES = 64 * 1024

# Hop-by-hop headers a proxy must not forward (RFC 7230 §6.1); "host" is
# rebuilt by http.client from the upstream address.  Names listed in a
# Connection header are dropped as well.
_HOP_BY_HOP = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Forward headers minus the hop-by-hop ones (and the Connection list)."""
    drop = set(_HOP_BY_HOP)
    for key, value in headers.items():
        if key.lower() == "connection":
            drop.update(part.strip().lower() for part in value.split(",") if part.strip())
    return {key: value for key, value in headers.items() if key.lower() not in drop}


class Router:
    """Least-in-flight upstream selection with a round-robin tie-break."""

    def __init__(self, ports: list[int]) -> None:
        self.ports = list(ports)
        self.inflight = {port: 0 for port in self.ports}
        self._rr = itertools.count()

    def pick(self) -> int:
        fewest = min(self.inflight.values())
        candidates = [port for port in self.ports if self.inflight[port] == fewest]
        return candidates[next(self._rr) % len(candidates)]

    def snapshot(self) -> str:
        return ", ".join(f"{port}:{self.inflight[port]}" for port in self.ports)


def _forward(port: int, method: str, path: str, headers: dict[str, str], body: bytes):
    conn = http.client.HTTPConnection(HOST, port, timeout=UPSTREAM_TIMEOUT)
    try:
        conn.request(method, path, body=body or None, headers=headers)
        response = conn.getresponse()
        return response.status, response.getheaders(), response.read()
    finally:
        conn.close()


def _response_bytes(status: int, headers: list[tuple[str, str]], body: bytes) -> bytes:
    skip = {"transfer-encoding", "content-length", "connection"}
    lines = [f"HTTP/1.1 {status} {http.client.responses.get(status, '')}".rstrip()]
    for key, value in headers:
        if key.lower() in skip:
            continue
        lines.append(f"{key}: {value}")
    lines.append(f"Content-Length: {len(body)}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


async def _read_request(reader: asyncio.StreamReader):
    head = await reader.readuntil(b"\r\n\r\n")
    if len(head) > MAX_HEADER_BYTES:
        raise ValueError("request head too large")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip()] = value.strip()
    lowered = {key.lower(): value for key, value in headers.items()}
    length = int(lowered.get("content-length", "0") or 0)
    body = await reader.readexactly(length) if length else b""
    return method, path, headers, body


async def serve(port: int, router: Router) -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            method, path, headers, body = await _read_request(reader)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            writer.close()
            return
        if path.startswith("/healthz"):
            payload = f"ok upstreams=[{router.snapshot()}]\n".encode()
            writer.write(_response_bytes(200, [("Content-Type", "text/plain")], payload))
            await writer.drain()
            writer.close()
            return
        upstream = router.pick()
        router.inflight[upstream] += 1
        started = time.monotonic()
        try:
            forwarded = _forward_headers(headers)
            status, out_headers, out_body = await asyncio.to_thread(
                _forward, upstream, method, path, forwarded, body
            )
        except Exception as exc:  # noqa: BLE001 - reported to the client as 502
            status, out_headers = 502, [("Content-Type", "text/plain")]
            out_body = f"upstream {upstream} failed: {exc}".encode()
        finally:
            router.inflight[upstream] -= 1
        elapsed = (time.monotonic() - started) * 1000
        print(
            f"[proxy] {method} {path} -> {upstream} {status} {elapsed:.0f}ms [{router.snapshot()}]",
            flush=True,
        )
        writer.write(_response_bytes(status, out_headers, out_body))
        try:
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, HOST, port)
    print(f"[proxy] listening on http://{HOST}:{port} -> {router.ports}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    port = int(argv[1])
    upstreams = [int(value) for value in argv[2:]] or [8010, 8011, 8012]
    try:
        asyncio.run(serve(port, Router(upstreams)))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
