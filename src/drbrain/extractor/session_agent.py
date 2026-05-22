"""Session-aware agent with persistent multi-turn conversation.

SessionAgent extends the tool-calling pattern from ReasonerAgent with:
  - Persistent sessions stored in agent_sessions / agent_messages tables
  - Cross-CLI-invocation context continuity
  - Automatic context compression when conversation exceeds token budget
  - ask (single-turn) and chat (interactive loop) modes
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drbrain.extractor.agent_tools import TOOL_DEFINITIONS, execute_tool
from drbrain.extractor.llm_client import acall_with_messages

if TYPE_CHECKING:
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning assistant. "
    "Use the provided tools to explore the graph and answer questions. "
    "Explain your reasoning step by step."
)

# Approximate token budget before compression kicks in
DEFAULT_TOKEN_BUDGET = 8000


class SessionAgent:
    """Stateful agent backed by persistent DB session storage.

    Usage:
        agent = SessionAgent()
        sid = agent.create_session(db, title="Explore transformers")
        agent.load_session(db, sid)
        answer = await agent.ask("What are the key innovations in attention?")
        # ... later, in a different CLI invocation ...
        agent.load_session(db, sid)
        followup = await agent.ask("And how do they compare to convolutions?")
    """

    def __init__(self) -> None:
        self.db: Database | None = None
        self.graph: GraphEngine | None = None
        self.models: list[dict] = []
        self.closure_context: str = ""

        self.session_id: str = ""
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.messages: list[dict] = []
        self._token_budget: int = DEFAULT_TOKEN_BUDGET

    # ── Session lifecycle ────────────────────────────────────────────────

    def create_session(
        self,
        db: Database,
        *,
        title: str = "",
        system_prompt: str = "",
        models: list[dict] | None = None,
    ) -> str:
        """Create a new session and return its ID.

        Args:
            db: Database instance.
            title: Human-readable session label.
            system_prompt: Custom system prompt (defaults to reasoning assistant).
            models: LLM model config list.

        Returns:
            The new session_id.
        """
        self.db = db
        self.models = models or []
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.session_id = _new_session_id()

        db.conn.execute(
            """INSERT INTO agent_sessions (session_id, title, system_prompt, model_config)
               VALUES (?, ?, ?, ?)""",
            (
                self.session_id,
                title,
                self.system_prompt,
                json.dumps(self.models),
            ),
        )
        db.commit()

        # Insert system message as seq=0
        self._persist_message("system", self.system_prompt, seq=0)
        self.messages = [{"role": "system", "content": self.system_prompt}]

        log.info("[session] created %s", self.session_id)
        return self.session_id

    def load_session(
        self,
        db: Database,
        session_id: str,
        *,
        graph: GraphEngine | None = None,
        models: list[dict] | None = None,
        closure_context: str = "",
    ) -> bool:
        """Load an existing session from DB, restoring full message history.

        Args:
            db: Database instance.
            session_id: The session to load.
            graph: Optional graph engine for tool execution.
            models: Override models from session config.
            closure_context: Optional closure-inferred relations.

        Returns:
            True if session was found and loaded, False otherwise.
        """
        self.db = db
        self.graph = graph
        self.closure_context = closure_context
        self.session_id = session_id

        # Load session metadata
        row = db.conn.execute(
            "SELECT title, system_prompt, model_config, status FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            log.warning("[session] %s not found", session_id)
            return False

        if row[3] == "deleted":
            log.warning("[session] %s is deleted", session_id)
            return False

        self.system_prompt = row[1] or DEFAULT_SYSTEM_PROMPT

        # Use provided models, or fall back to stored config
        if models:
            self.models = models
        else:
            try:
                self.models = json.loads(row[2]) if row[2] else []
            except json.JSONDecodeError:
                self.models = []

        # Load messages
        rows = db.conn.execute(
            "SELECT role, content, tool_calls_json, tool_call_id, tool_name "
            "FROM agent_messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()

        self.messages = []
        for r in rows:
            msg = {"role": r[0], "content": r[1] or ""}
            if r[0] == "assistant" and r[2]:
                try:
                    msg["tool_calls"] = json.loads(r[2])
                except json.JSONDecodeError:
                    pass
            if r[0] == "tool" and r[3]:
                msg["tool_call_id"] = r[3]
                if r[4]:
                    msg["name"] = r[4]
            self.messages.append(msg)

        log.info("[session] loaded %s — %d messages", session_id, len(self.messages))
        return True

    def delete_session(self, db: Database, session_id: str) -> bool:
        """Mark a session as deleted (soft delete)."""
        db.conn.execute(
            "UPDATE agent_sessions SET status='deleted', updated_at=? WHERE session_id=?",
            (_now_iso(), session_id),
        )
        db.commit()
        if self.session_id == session_id:
            self.session_id = ""
            self.messages = []
        return True

    # ── Core interaction ──────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        max_turns: int = 8,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> str:
        """Answer a question within the current session context.

        Appends the question to the conversation history, runs the tool-calling
        loop, and persists all intermediate messages to DB.

        Args:
            question: The user's question.
            max_turns: Maximum LLM tool-calling rounds.
            token_budget: Token budget before context compression.

        Returns:
            The final assistant response text.
        """
        if not self.session_id:
            return "No active session. Create or load one first."

        if not self.models:
            return "No LLM models configured."

        self._token_budget = token_budget

        # Append user message
        self._append_and_persist("user", question)

        # Compress if needed
        self._maybe_compress()

        tools = list(TOOL_DEFINITIONS)

        for _turn in range(max_turns):
            result = await acall_with_messages(
                messages=self.messages,
                models=self.models,
                tools=tools,
                max_tokens=1024,
                temperature=0.3,
            )

            if result is None:
                return "LLM call failed after exhausting all models."

            text = result.get("text", "")
            tool_calls = result.get("tool_calls")

            if tool_calls:
                # Persist assistant message with tool_calls
                self._persist_message("assistant", text, tool_calls_json=json.dumps(tool_calls))
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": tool_calls,
                    }
                )

                # Execute each tool and append results
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_result = execute_tool(
                        tool_name,
                        args,
                        db=self.db,
                        graph=self.graph,
                        papers_dir=self._papers_dir(),
                    )
                    result_content = json.dumps(tool_result, ensure_ascii=False, default=str)

                    self._persist_message(
                        "tool",
                        result_content,
                        tool_call_id=tc.get("id", ""),
                        tool_name=tool_name,
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result_content,
                        }
                    )

                # Compress after tool results added
                self._maybe_compress()
            else:
                # Final answer — persist and return
                self._persist_message("assistant", text)
                self.messages.append({"role": "assistant", "content": text})
                self._touch_session()
                return text

        return "Unable to answer after maximum reasoning turns."

    async def chat(
        self,
        *,
        on_message: Any = None,
        max_turns_per_question: int = 8,
    ) -> None:
        """Interactive chat loop (for CLI interactive mode).

        Reads stdin line by line. Type /exit or /quit to end.

        Args:
            on_message: Optional callback(role, content) for real-time output.
            max_turns_per_question: Max tool-calling rounds per question.
        """
        print(f"Session: {self.session_id}")
        print("Type your questions. /exit to quit, /history to show context.\n")

        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            line = line.strip()
            if not line:
                continue
            if line.lower() in ("/exit", "/quit"):
                print("Exiting chat.")
                break
            if line.lower() == "/history":
                self._print_history()
                continue

            answer = await self.ask(line, max_turns=max_turns_per_question)
            if on_message:
                on_message("assistant", answer)
            else:
                print(f"\n{answer}\n")

    # ── Context management ────────────────────────────────────────────────

    def _maybe_compress(self) -> None:
        """Compress conversation history if estimated token count exceeds budget.

        Strategy: keep the system message and last K non-system messages,
        compress everything in between into a summary system message.
        """
        if len(self.messages) < 8:
            return

        estimated = sum(len(m.get("content", "")) // 4 for m in self.messages)
        if estimated < self._token_budget:
            return

        # Keep system (msg 0) and last 6 messages
        keep = min(6, len(self.messages) - 3)
        to_compress = self.messages[1:-keep]
        recent = self.messages[-keep:]

        summary_text = _build_summary_text(to_compress)

        # Replace in-memory: system + summary + recent
        self.messages = [
            self.messages[0],
            {"role": "system", "content": f"[Context summary]\n{summary_text}"},
            *recent,
        ]

        log.info(
            "[session] compressed %d messages → summary (%d chars), kept %d recent",
            len(to_compress),
            len(summary_text),
            len(recent),
        )

    def _append_and_persist(self, role: str, content: str) -> None:
        """Append a message to in-memory list and persist to DB."""
        self.messages.append({"role": role, "content": content})
        self._persist_message(role, content)

    def _persist_message(
        self,
        role: str,
        content: str,
        *,
        tool_calls_json: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        seq: int | None = None,
    ) -> None:
        """Write a single message to agent_messages table."""
        if not self.db or not self.session_id:
            return

        if seq is None:
            row = self.db.conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            seq = row[0] if row else 0

        self.db.conn.execute(
            """INSERT INTO agent_messages
               (session_id, seq, role, content, tool_calls_json, tool_call_id, tool_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (self.session_id, seq, role, content, tool_calls_json, tool_call_id, tool_name),
        )
        self.db.commit()

    def _touch_session(self) -> None:
        """Update session updated_at timestamp."""
        if self.db and self.session_id:
            self.db.conn.execute(
                "UPDATE agent_sessions SET updated_at=? WHERE session_id=?",
                (_now_iso(), self.session_id),
            )
            self.db.commit()

    def _papers_dir(self) -> Path | None:
        """Resolve papers data directory from DB config."""
        if not self.db:
            return None
        return self.db.path.parent / "papers"

    def _print_history(self) -> None:
        """Print conversation summary to stdout."""
        print(f"\n--- Session: {self.session_id} ({len(self.messages)} messages) ---")
        for i, m in enumerate(self.messages):
            role = m.get("role", "?")
            content = m.get("content", "")
            preview = content[:120].replace("\n", " ")
            if m.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
                preview = f"[tool_calls: {', '.join(names)}]"
            elif role == "tool":
                preview = f"[tool result: {len(content)} chars]"
            print(f"  {i:3d} [{role:9s}] {preview}")
        print("---\n")


# ── Helpers ──────────────────────────────────────────────────────────────


def _new_session_id() -> str:
    """Generate a short, human-friendly session ID."""
    return "sess-" + uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return datetime.now(datetime.UTC).isoformat()


def _build_summary_text(messages: list[dict]) -> str:
    """Build a plain-text summary of a message list for context compression."""
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
