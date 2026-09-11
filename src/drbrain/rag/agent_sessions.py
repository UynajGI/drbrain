"""Session history projection and persistence for the research assistant."""
from __future__ import annotations
import json
import uuid
from typing import Any
from drbrain.config import Config
from drbrain.security import public_model_configs, redact_sensitive
try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
except ImportError:
    ChatMessage = MessageRole = None
SESSION_TOKEN_BUDGET = 8000
SESSION_KEEP_RECENT = 6
MAX_RESULT_SUMMARY_CHARS = 800


def _history_summary(messages: list[dict]) -> str:
    """Plain-text summary of a message list for context compression.

    Mirrors ``SessionAgent._build_summary_text``: tool results are reduced to
    a char count, assistant tool-call messages to the tool names, everything
    else to a 200-char content preview.
    """
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if role == "tool":
            parts.append(f"[Tool result: {len(content)} chars]")
        elif role == "assistant" and m.get("tool_calls"):
            names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
            parts.append(f"Assistant called: {', '.join(names)}")
        elif content:
            parts.append(f"[{role}] {content[:200]}")
    return "\n".join(parts)

def _session_principal_matches(db: Any, session_id: str, principal: str | None) -> bool:
    """Check a session owner when a caller supplies an authenticated principal.

    ``principal=None`` deliberately preserves the historic local-CLI behavior.
    Once a principal is supplied, unowned legacy sessions are denied rather
    than silently claimed by the first caller.
    """
    if principal is not None and not str(principal).strip():
        return False
    checker = getattr(db, "session_principal_matches", None)
    if callable(checker):
        return bool(checker(session_id, principal))
    row = db.conn.execute(
        "SELECT owner_principal FROM agent_sessions WHERE session_id = ? AND status != 'deleted'",
        (session_id,),
    ).fetchone()
    return row is not None and (principal is None or str(row[0] or "") == principal)

def load_session_history(
    db: Any,
    session_id: str,
    token_budget: int = SESSION_TOKEN_BUDGET,
    *,
    principal: str | None = None,
) -> list[ChatMessage]:
    """Restore a session's prior turns as LlamaIndex ``ChatMessage``s (T9).

    Reads ``agent_messages`` (the shape both ``SessionAgent`` and
    ``_persist_reason_session`` write) in ``seq`` order and converts each row
    to a :class:`ChatMessage` for injection as ``agent.run(chat_history=...)``:

    * ``system`` rows are skipped — ``build_agent`` re-injects the same
      system prompt, so including the stored copy would duplicate it;
    * assistant tool-call rows keep ``additional_kwargs["tool_calls"]`` (the
      stored single dict is normalized to a list) and tool rows keep
      ``tool_call_id``/``name``, so the drbrain fallback chain receives a
      valid OpenAI tool loop;
    * long histories are compressed exactly like ``SessionAgent._maybe_compress``
      — once the estimated token count (``len(content)//4``) exceeds
      ``token_budget``, everything but the last :data:`SESSION_KEEP_RECENT`
      messages is collapsed into a leading ``[Context summary]`` system
      message.

    Returns ``[]`` for sessions with no (non-system) messages.
    """
    if db is None:
        return []
    if principal is not None and not _session_principal_matches(db, session_id, principal):
        raise PermissionError(f"Session access denied: {session_id}")
    rows = db.conn.execute(
        "SELECT role, content, tool_calls_json, tool_call_id, tool_name "
        "FROM agent_messages WHERE session_id = ? AND role != 'system' ORDER BY seq",
        (session_id,),
    ).fetchall()

    messages: list[dict[str, Any]] = []
    for r in rows:
        msg: dict[str, Any] = {
            "role": r[0],
            "content": redact_sensitive(r[1]) or "",
        }
        if r[0] == "assistant" and r[2]:
            try:
                msg["tool_calls"] = redact_sensitive(json.loads(r[2]))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        if r[0] == "tool" and r[3]:
            msg["tool_call_id"] = redact_sensitive(r[3]) or ""
            if r[4]:
                msg["name"] = redact_sensitive(r[4]) or ""
        messages.append(msg)

    if not messages:
        return []

    # SessionAgent._maybe_compress parity: keep the recent tail, summarize
    # the middle once the estimated size exceeds the budget.
    if len(messages) >= 8:
        estimated = sum(len(m.get("content", "")) // 4 for m in messages)
        if estimated >= token_budget:
            keep = min(SESSION_KEEP_RECENT, len(messages) - 3)
            recent = messages[-keep:] if keep > 0 else messages
            middle = messages[:-keep] if keep > 0 else []
            summary = _history_summary(middle)
            messages = [{"role": "system", "content": f"[Context summary]\n{summary}"}] + recent

    out: list[ChatMessage] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append(ChatMessage(role=MessageRole.SYSTEM, content=m["content"]))
        elif role == "user":
            out.append(ChatMessage(role=MessageRole.USER, content=m["content"]))
        elif role == "assistant":
            ak: dict[str, Any] = {}
            if m.get("tool_calls"):
                tc = m["tool_calls"]
                ak["tool_calls"] = tc if isinstance(tc, list) else [tc]
            out.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=m["content"], additional_kwargs=ak)
            )
        elif role == "tool":
            ak = {"tool_call_id": m.get("tool_call_id", "")}
            if m.get("name"):
                ak["name"] = m["name"]
            out.append(
                ChatMessage(role=MessageRole.TOOL, content=m["content"], additional_kwargs=ak)
            )
    return out

def _persist_reason_session(
    cfg: Config,
    db: Any,
    session_id: str | None,
    question: str,
    system_prompt: str,
    answer: str,
    tool_calls: list[dict[str, Any]],
    models: list[dict],
    *,
    principal: str | None = None,
) -> str | None:
    """Append one reasoning run to ``agent_sessions``/``agent_messages``.

    Write-only, best-effort, in a shape ``SessionAgent.load_session`` can read
    back (assistant messages carry ``tool_calls_json``, tool messages carry
    ``tool_call_id`` + ``tool_name``). ``"new"`` creates a session; an unknown
    existing id raises ``ValueError``. Read/restore + compression live in
    :func:`load_session_history` (T9).
    """
    if db is None or session_id is None:
        return None

    created = False
    if session_id == "new":
        session_id = "sess-" + uuid.uuid4().hex[:8]
        db.insert_agent_session(
            session_id,
            title="reason",
            system_prompt=redact_sensitive(system_prompt) or "",
            # Persist only non-secret routing metadata.  A later invocation
            # supplies the current runtime model list (and its credentials).
            model_config=json.dumps(public_model_configs(models), ensure_ascii=False),
            owner_principal=principal or "",
        )
        created = True
    else:
        if not _session_principal_matches(db, session_id, principal):
            if principal is not None:
                raise PermissionError(f"Session access denied: {session_id}")
            raise ValueError(f"Session not found: {session_id}")

    seq_row = db.conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    seq = seq_row[0] if seq_row else 0

    if created:
        db.insert_agent_message(
            session_id,
            seq,
            "system",
            content=redact_sensitive(system_prompt) or "",
        )
        seq += 1
    db.insert_agent_message(
        session_id,
        seq,
        "user",
        content=redact_sensitive(question) or "",
    )
    seq += 1
    for tc in tool_calls:
        call_id = f"call_{seq}"
        tcall = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(redact_sensitive(tc.get("args") or {}), ensure_ascii=False),
            },
        }
        safe_tcall_json = json.dumps(redact_sensitive(tcall), ensure_ascii=False)
        db.insert_agent_message(
            session_id, seq, "assistant", content="", tool_calls_json=safe_tcall_json
        )
        seq += 1
        db.insert_agent_message(
            session_id,
            seq,
            "tool",
            content=(redact_sensitive(str(tc.get("result_summary") or "")) or "")[
                :MAX_RESULT_SUMMARY_CHARS
            ],
            tool_call_id=redact_sensitive(call_id) or "",
            tool_name=redact_sensitive(str(tc.get("name", ""))) or "",
        )
        seq += 1
    db.insert_agent_message(
        session_id,
        seq,
        "assistant",
        content=redact_sensitive(answer or "") or "",
    )
    db.touch_session(session_id)
    db.commit()
    return session_id


# ── public entry point ──────────────────────────────────────────────────────
