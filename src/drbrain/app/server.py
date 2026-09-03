"""Minimal HTTP server for the DrBrain WebUI (standard library only).

Routes (all JSON unless noted)::

    GET  /                              index.html
    GET  /api/dashboard                 KPI counters + recent runs
    GET  /api/search?q=&limit=&type=    BM25 search  (== drbrain search)
    POST /api/ask         {question, top_k}          (== drbrain ask)
    GET  /api/runs                      autoresearch runs in the ledger
    POST /api/runs        {topic, max_cycles}        (== drbrain autoresearch run, background)
    GET  /api/runs/<id>/events?after=   append-only event stream (poll)
    GET  /api/runs/<id>/claims          proposals + reviews + settlement
    GET  /api/run-status?topic=         background thread status
    GET  /api/experiments[?run_id=]     compute jobs
    GET  /api/plugins                   discovered plugins
    GET  /api/assets                    database / ledger / plugins / exports
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from drbrain.app import service

STATIC_DIR = Path(__file__).parent / "static"


class WebUIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], cfg: Any, run_manager: service.RunManager | None = None
    ):
        super().__init__(address, WebUIHandler)
        self.cfg = cfg
        self.run_manager = run_manager or service.RunManager()


class WebUIHandler(BaseHTTPRequestHandler):
    server: WebUIServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return  # keep the CLI quiet; errors are returned to the client as JSON

    # ── plumbing ──
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in path.parents or not path.is_file():
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript",
            ".css": "text/css",
            ".svg": "image/svg+xml",
        }.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        return data if isinstance(data, dict) else {}

    # ── routing ──
    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        cfg = self.server.cfg
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/") :])
            if url.path == "/api/dashboard":
                return self._json(service.dashboard(cfg))
            if url.path == "/api/search":
                return self._json(
                    {
                        "query": q.get("q", ""),
                        "results": service.search(
                            cfg,
                            q.get("q", ""),
                            limit=int(q.get("limit", 10)),
                            type_filter=q.get("type") or None,
                        ),
                    }
                )
            if url.path == "/api/runs":
                return self._json(service.runs(cfg))
            if url.path == "/api/run-status":
                return self._json(self.server.run_manager.status(q.get("topic", "")))
            parts = url.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                run_id, leaf = parts[2], parts[3]
                if leaf == "events":
                    return self._json(
                        service.run_events(
                            cfg,
                            run_id,
                            after=int(q.get("after", 0)),
                            limit=int(q.get("limit", 200)),
                        )
                    )
                if leaf == "claims":
                    return self._json(service.run_claims(cfg, run_id))
            if url.path == "/api/experiments":
                return self._json(service.experiments(cfg, run_id=q.get("run_id") or None))
            if url.path == "/api/plugins":
                return self._json(service.plugins(cfg))
            if url.path == "/api/assets":
                return self._json(service.assets(cfg))
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
            return self._json(
                {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        url = urlparse(self.path)
        cfg = self.server.cfg
        try:
            body = self._body()
            if url.path == "/api/ask":
                result = service.ask(
                    cfg, str(body.get("question", "")), top_k=int(body.get("top_k", 5))
                )
                status = HTTPStatus.SERVICE_UNAVAILABLE if result.get("unavailable") else 200
                return self._json(result, status)
            if url.path == "/api/runs":
                mc = body.get("max_cycles")
                max_cycles = int(mc) if isinstance(mc, (int, str)) and str(mc).strip() else None
                try:
                    started = self.server.run_manager.start(
                        cfg, str(body.get("topic", "")), max_cycles=max_cycles
                    )
                except (ValueError, RuntimeError) as exc:
                    return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return self._json(started, HTTPStatus.ACCEPTED)
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            return self._json(
                {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )


def serve(
    cfg: Any, host: str = "127.0.0.1", port: int = 8765, *, block: bool = True
) -> WebUIServer:
    """Start the WebUI. With ``block=False`` the server runs in a daemon thread."""
    server = WebUIServer((host, port), cfg)
    if block:
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return server
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
