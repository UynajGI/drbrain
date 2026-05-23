"""Tests for SessionAgent persistent reasoning system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from drbrain.extractor.session_agent import SessionAgent, _build_summary_text, _new_session_id
from drbrain.storage.database import Database

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sess_db():
    """Temporary DB for session tests."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        yield db
        db.close()


@pytest.fixture
def fake_models():
    return [{"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"}]


def _make_mock_response(text="Answer", tool_calls=None):
    """Build a mock for acall_with_messages return value."""
    result = {
        "text": text,
        "tool_calls": tool_calls,
        "usage": {"in": 100, "out": 50},
    }
    return result


# ── Session lifecycle tests ───────────────────────────────────────────────


def test_create_session(sess_db, fake_models):
    """create_session writes to agent_sessions and returns a session ID."""
    agent = SessionAgent()
    sid = agent.create_session(sess_db, title="Test Session", models=fake_models)

    assert sid.startswith("sess-")
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"

    # Verify DB row
    row = sess_db.conn.execute(
        "SELECT title, status FROM agent_sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    assert row is not None
    assert row[0] == "Test Session"
    assert row[1] == "active"


def test_load_session(sess_db, fake_models):
    """load_session restores session metadata and full message history."""
    agent = SessionAgent()
    sid = agent.create_session(sess_db, title="Loadable", models=fake_models)

    # Simulate adding messages
    agent._append_and_persist("user", "Hello")
    agent._append_and_persist("assistant", "Hi there!")

    # Reload from fresh agent
    agent2 = SessionAgent()
    ok = agent2.load_session(sess_db, sid, models=fake_models)
    assert ok is True
    assert agent2.session_id == sid
    assert len(agent2.messages) == 3  # system + user + assistant
    assert agent2.messages[1]["role"] == "user"
    assert agent2.messages[1]["content"] == "Hello"


def test_load_session_not_found(sess_db, fake_models):
    """load_session returns False for nonexistent session."""
    agent = SessionAgent()
    ok = agent.load_session(sess_db, "nonexistent", models=fake_models)
    assert ok is False


def test_delete_session(sess_db, fake_models):
    """delete_session soft-deletes and clears local state."""
    agent = SessionAgent()
    sid = agent.create_session(sess_db, title="Deletable", models=fake_models)

    agent.delete_session(sess_db, sid)

    row = sess_db.conn.execute(
        "SELECT status FROM agent_sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    assert row[0] == "deleted"
    assert agent.session_id == ""
    assert agent.messages == []


# ── Ask tests (with mocked LLM) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_simple(sess_db, fake_models):
    """ask() returns LLM text response and persists conversation."""
    agent = SessionAgent()
    sid = agent.create_session(sess_db, title="Ask Test", models=fake_models)

    with mock.patch(
        "drbrain.extractor.session_agent.acall_with_messages",
        return_value=_make_mock_response("The answer is 42."),
    ):
        answer = await agent.ask("What is the meaning of life?")

    assert answer == "The answer is 42."
    # system + user + assistant = 3 messages
    assert len(agent.messages) == 3
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"] == "The answer is 42."

    # Verify persistence
    msg_count = sess_db.conn.execute(
        "SELECT COUNT(*) FROM agent_messages WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    assert msg_count == 3


@pytest.mark.asyncio
async def test_ask_with_tool_calls(sess_db, fake_models):
    """ask() executes tool calls and returns final answer."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Tool Test", models=fake_models)

    # Mock: first call returns tool_calls, second call returns final answer
    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "search_concepts", "arguments": '{"query": "transformer"}'},
    }

    calls = [
        _make_mock_response("", tool_calls=[tool_call]),
        _make_mock_response("Found transformer concept."),
    ]

    with mock.patch(
        "drbrain.extractor.session_agent.acall_with_messages",
        side_effect=calls,
    ):
        answer = await agent.ask("Find transformer")

    assert "Found transformer concept." in answer


@pytest.mark.asyncio
async def test_ask_no_session(sess_db, fake_models):
    """ask() without a session returns error message."""
    agent = SessionAgent()
    agent.models = fake_models
    answer = await agent.ask("Hello")
    assert "No active session" in answer


# ── Context compression tests ────────────────────────────────────────────


def test_context_compression_triggers(sess_db, fake_models):
    """_maybe_compress() compresses when message count and tokens exceed budget."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Compress", models=fake_models)

    # Add many messages to trigger compression
    long_text = "A" * 2000  # ~500 tokens per message at 4 chars/token
    for i in range(12):
        agent._append_and_persist("user", f"Question {i}: {long_text}")
        agent._append_and_persist("assistant", f"Answer {i}: {long_text}")

    before = len(agent.messages)
    agent._maybe_compress()
    after = len(agent.messages)

    # Should have compressed: system + summary + last 6
    assert after < before
    assert after <= 8


def test_context_compression_noop_when_small(sess_db, fake_models):
    """_maybe_compress() is a no-op for short conversations."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Small", models=fake_models)
    agent._append_and_persist("user", "Short question")

    before = len(agent.messages)
    agent._maybe_compress()
    assert len(agent.messages) == before


# ── Session persistence across calls ─────────────────────────────────────


@pytest.mark.asyncio
async def test_session_persistence_across_calls(sess_db, fake_models):
    """Two separate ask() calls maintain conversation continuity."""
    agent = SessionAgent()
    sid = agent.create_session(sess_db, title="Persist", models=fake_models)

    # First question
    with mock.patch(
        "drbrain.extractor.session_agent.acall_with_messages",
        return_value=_make_mock_response("Answer 1"),
    ):
        await agent.ask("Question 1")

    # Simulate a new CLI invocation — load session fresh
    agent2 = SessionAgent()
    agent2.load_session(sess_db, sid, models=fake_models)

    with mock.patch(
        "drbrain.extractor.session_agent.acall_with_messages",
        return_value=_make_mock_response("Answer 2"),
    ):
        await agent2.ask("Question 2")

    # The second agent should have context from both questions
    user_msgs = [m for m in agent2.messages if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert "Question 1" in user_msgs[0]["content"]
    assert "Question 2" in user_msgs[1]["content"]


# ── Helper tests ──────────────────────────────────────────────────────────


def test_new_session_id_format():
    """Session IDs follow the sess-XXXXXXXX pattern."""
    sid = _new_session_id()
    assert sid.startswith("sess-")
    assert len(sid) == 13  # sess- + 8 hex chars


def test_build_summary_text():
    """_build_summary_text produces readable summary."""
    msgs = [
        {"role": "user", "content": "What is attention?"},
        {"role": "assistant", "content": "Attention is a mechanism..."},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "search_concepts", "arguments": "{}"}}],
        },
        {"role": "tool", "content": '{"results": []}'},
    ]
    summary = _build_summary_text(msgs)
    assert "What is attention" in summary
    assert "Attention is a mechanism" in summary
    assert "search_concepts" in summary


# ── Inject context tests ────────────────────────────────────────────────


def test_inject_context(sess_db, fake_models):
    """inject_context() appends system message without LLM call."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Inject", models=fake_models)

    # Inject a build summary
    agent.inject_context("Paper X extracted 5 concepts, 3 relations.", label="build:X")

    # system + injected = 2 messages
    assert len(agent.messages) == 2
    assert agent.messages[1]["role"] == "system"
    assert "[build:X]" in agent.messages[1]["content"]
    assert "5 concepts" in agent.messages[1]["content"]

    # Verify persistence
    msg_count = sess_db.conn.execute(
        "SELECT COUNT(*) FROM agent_messages WHERE session_id = ?",
        (agent.session_id,),
    ).fetchone()[0]
    assert msg_count == 2


def test_inject_context_no_session_is_noop(sess_db, fake_models):
    """inject_context() without session is a silent no-op."""
    agent = SessionAgent()
    agent.inject_context("Should not persist", label="test")
    assert len(agent.messages) == 0


def test_inject_context_triggers_compression(sess_db, fake_models):
    """Many large injections trigger _maybe_compress()."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Compress Inject", models=fake_models)
    agent._token_budget = 2000  # lower budget so compression triggers

    long_text = "B" * 2000  # ~500 tokens each
    for i in range(14):
        agent.inject_context(f"Injection {i}: {long_text}", label=f"build:{i}")

    # After compression: system + summary + last 6
    assert len(agent.messages) <= 10


# ── Bidirectional reasoning tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_reason_bidirectional_consistent(sess_db, fake_models):
    """reason_bidirectional returns immediately when hypothesis is consistent."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Bidir", models=fake_models)

    with (
        mock.patch(
            "drbrain.extractor.session_agent.acall_with_messages",
            return_value=_make_mock_response("Method X addresses Problem Y."),
        ),
        mock.patch(
            "drbrain.extractor.session_agent.kg_validate",
            return_value={"consistent": True, "violations": [], "patterns": []},
        ),
    ):
        result = await agent.reason_bidirectional("What addresses Problem Y?")

    assert result["answer"] == "Method X addresses Problem Y."
    assert result["rounds"] == 1
    assert len(result["hypotheses"]) == 1


@pytest.mark.asyncio
async def test_reason_bidirectional_with_violations(sess_db, fake_models):
    """reason_bidirectional runs revision rounds when KG finds violations."""
    agent = SessionAgent()
    agent.create_session(sess_db, title="Bidir Violations", models=fake_models)

    # Round 1: inconsistent -> Round 2: consistent
    ask_responses = [
        _make_mock_response("Hypothesis A: X causes Y."),
        _make_mock_response("Hypothesis B: X correlates with Y."),
    ]

    validations = [
        {
            "consistent": False,
            "violations": [{"type": "tbox", "edge": {}, "reason": "causes not valid for type"}],
            "patterns": [],
        },
        {"consistent": True, "violations": [], "patterns": []},
    ]

    with (
        mock.patch(
            "drbrain.extractor.session_agent.acall_with_messages",
            side_effect=ask_responses,
        ),
        mock.patch(
            "drbrain.extractor.session_agent.kg_validate",
            side_effect=validations,
        ),
    ):
        result = await agent.reason_bidirectional("What is the relation between X and Y?")

    assert result["rounds"] == 2
    assert len(result["hypotheses"]) == 2
    assert result["answer"] == "Hypothesis B: X correlates with Y."


def test_reason_bidirectional_no_session(sess_db, fake_models):
    """reason_bidirectional returns error without active session."""
    agent = SessionAgent()
    agent.models = fake_models

    import asyncio

    result = asyncio.run(agent.reason_bidirectional("test?"))
    assert "error" in result
    assert "No active session" in result["error"]


# ── Tool-calling integration (no LLM) ───────────────────────────────────


def test_execute_tool_dispatch():
    """execute_tool routes to correct handler by name."""
    from drbrain.extractor.agent_tools import execute_tool

    # search_concepts with no DB returns []
    result = execute_tool("search_concepts", {"query": "test"}, db=None)
    assert result == []

    # Unknown tool returns []
    result = execute_tool("unknown_tool", {}, db=None)
    assert result == []
