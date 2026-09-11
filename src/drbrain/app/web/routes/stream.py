"""Server-Sent Events for live research-run observation.

Implements the design contract (§6.2): ``id`` carries ``event_seq``, reconnects
resume via ``Last-Event-ID`` (or ``after`` on first connect), heartbeats keep
proxies from buffering, and a revoked/expired login terminates the stream
instead of leaking events to a stale tab.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from drbrain.app import auth, service
from drbrain.app.web import deps

router = APIRouter(dependencies=[Depends(deps.authenticate)])

_TERMINAL = {"succeeded", "failed", "cancelled"}
_POLL_SECONDS = 1.0
_HEARTBEAT_EVERY = 15
_AUTH_RECHECK_EVERY = 30


def _sse(event: str, data: object, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n\n"


@router.get("/api/runs/{run_id}/stream")
async def stream_run(
    request: Request,
    run_id: str,
    after: int = Query(0, ge=0),
    project_id: str = Query(""),
) -> StreamingResponse:
    cfg = deps.get_cfg(request)
    try:
        pid = project_id or service.run_project(cfg, run_id)
    except service.RunNotFoundError:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="unknown research run") from None
    project = deps.resolve_project(request, pid)
    scoped_project = project["project_id"]
    # EventSource reconnect: the browser sends the last event id back.
    last_event_id = request.headers.get("last-event-id")
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))
    cookie_value = request.cookies.get(auth.SESSION_COOKIE)
    bearer = request.headers.get("authorization", "")

    async def generate() -> AsyncIterator[str]:
        cursor = after
        polls = 0
        try:
            while True:
                if await request.is_disconnected():
                    return
                events = await asyncio.to_thread(
                    service.run_events, cfg, run_id, cursor, 200, scoped_project
                )
                for event in events:
                    cursor = int(event["seq"])
                    yield _sse("message", event, event_id=cursor)
                detail = await asyncio.to_thread(service.run_detail, cfg, run_id, scoped_project)
                yield _sse(
                    "status",
                    {
                        "seq": cursor,
                        "status": detail["status"],
                        "display_status": detail["display_status"],
                        "events": detail["events"],
                        "claims": detail["claims"],
                        "verified": detail["verified"],
                        "experiments": detail["experiments"],
                    },
                )
                if detail["status"] in _TERMINAL:
                    yield _sse("end", {"status": detail["status"]})
                    return
                polls += 1
                if polls % _HEARTBEAT_EVERY == 0:
                    yield ": keep-alive\n\n"
                if polls % _AUTH_RECHECK_EVERY == 0:
                    if cookie_value:
                        if auth.resolve_session(cfg, cookie_value) is None:
                            yield _sse("auth-expired", {"reason": "login expired"})
                            return
                    elif bearer.lower().startswith("bearer "):
                        token = bearer.split(" ", 1)[1].strip()
                        if not auth.verify_bootstrap_token(cfg, token):
                            yield _sse("auth-expired", {"reason": "token rotated"})
                            return
                await asyncio.sleep(_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
