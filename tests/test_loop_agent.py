"""Agent-backed node tests: the loop builds + runs a FunctionAgent.

Proves the loop layer's "independent agent" machinery — ``build_node_agent``
assembles the full tool surface (graph tools + external plugins) and
``run_agent`` runs it to a text answer. The LLM is stubbed via
``llm_client.acall_with_messages`` (offline, mirrors test_rag_agent.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.loop import ResearchLoopWorkflow

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None
pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]


def _cfg() -> Config:
    c = Config(llamaindex=LlamaIndexConfig(enabled=True))
    c.llm.models = list(MODELS)
    c.api.cache_ttl = 0
    c.llamaindex.storage_dir = "/nonexistent-rag-index"
    return c


def _scripted_llm(monkeypatch, script):
    calls = [0]

    async def fake(messages, models, tools=None, **kw):
        step = script[min(calls[0], len(script) - 1)]
        calls[0] += 1
        return dict(step)

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    return fake


def _write_search_plugin(tmp_path) -> str:
    """Drop a mock ``search_papers`` plugin returning two fixed paper titles."""
    (tmp_path / "search_papers.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(Plugin(name='search_papers', description='d',\n"
        "        input_schema={'type':'object','properties':{'query':{'type':'string'},\n"
        "        'limit':{'type':'integer'}}}),\n"
        "        lambda a: {'papers': [{'title': 'Paper A'}, {'title': 'Paper B'}]})\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _write_compute_plugin(tmp_path) -> str:
    """Drop a mock ``run_python`` plugin so ``_has_compute_tools()`` turns on.

    The T4 gate only inspects the on-disk job artifacts a real async run would
    leave behind — the plugin handler itself is never called by these tests.
    """
    (tmp_path / "run_python.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(Plugin(name='run_python', description='d',\n"
        "        input_schema={'type':'object','properties':{'code':{'type':'string'},\n"
        "        'mode':{'type':'string'}}}),\n"
        "        lambda a: {'job_id': 'stub', 'mode': 'async'})\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _write_job(run_dir, job_id: str, log_text: str = "result = 1.0") -> Path:
    """Pre-write a fake async job's on-disk artifacts (``<job_id>.json`` + ``<job_id>.log``).

    Mirrors what ``run_python(mode=async)`` leaves in ``$DRBRAIN_RUN_DIR``: a
    meta json (pid / log_path) plus the captured stdout log.
    """
    jobs = Path(run_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    log_file = jobs / f"{job_id}.log"
    log_file.write_text(log_text, encoding="utf-8")
    (jobs / f"{job_id}.json").write_text(
        json.dumps(
            {"job_id": job_id, "pid": 12345, "log_path": str(log_file)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return jobs


def _compute_loop_script(verify_text: str) -> list[dict]:
    """Full agent-backed loop script with compute tools present (run_python)."""
    return [
        {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
        {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
        {
            "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
            "tool_calls": None,
            "usage": None,
        },
        {"text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
         "tool_calls": None, "usage": None},
        {"text": verify_text, "tool_calls": None, "usage": None},
    ]


def test_build_node_agent_assembles_full_tool_surface(tmp_path):
    (tmp_path / "foo_plugin.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(Plugin(name='foo', description='d', input_schema={}), lambda a: {})\n",
        encoding="utf-8",
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=str(tmp_path))
    agent = wf.build_node_agent()
    assert agent is not None
    names = {t.metadata.name for t in agent.tools}
    assert "search_concepts" in names  # built-in graph tool
    assert "foo" in names  # external plugin tool


def test_build_node_agent_none_without_cfg():
    wf = ResearchLoopWorkflow()
    assert wf.build_node_agent() is None


def test_run_agent_returns_answer(monkeypatch):
    _scripted_llm(monkeypatch, [{"text": "Hello from agent.", "tool_calls": None, "usage": None}])
    wf = ResearchLoopWorkflow(cfg=_cfg())
    agent = wf.build_node_agent()
    answer = asyncio.run(wf.run_agent(agent, "hi"))
    assert answer == "Hello from agent."


def test_run_agent_none_returns_none():
    wf = ResearchLoopWorkflow()
    answer = asyncio.run(wf.run_agent(None, "hi"))
    assert answer is None


def test_retrieve_node_uses_agent(monkeypatch, tmp_path):
    """The retrieve node distills the task and fetches candidates via the plugin."""
    _scripted_llm(
        monkeypatch, [{"text": '{"query": "flat band"}', "tool_calls": None, "usage": None}]
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "candidates=2" in result


def test_identify_gaps_node_proposes_hypotheses(monkeypatch, tmp_path):
    """identify_gaps runs the agent, parsing structured JSON into gaps + hypotheses."""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": (
                    '{"gaps": ["gap1", "gap2"], '
                    '"hypotheses": [{"statement": "h1", "conditions": {}}, '
                    '{"statement": "h2", "conditions": {}}]}'
                ),
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "gaps=2" in result
    assert "hypotheses=2" in result


def test_parse_json_lenient():
    from drbrain.loop.workflow import _parse_json_lenient

    assert _parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_lenient('Here is the result: {"a": 1}') == {"a": 1}
    assert _parse_json_lenient("no json here") is None


def test_full_agent_backed_loop(monkeypatch, tmp_path):
    """End-to-end: retrieve → extract → identify_gaps → critique → verify all run the agent."""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {
                "text": '{"entities": ["flat band", "kagome"]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0, "computed": "1.0", "value": 1.0}]}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "candidates=2" in result
    assert "gaps=1" in result
    assert "hypotheses=1" in result
    assert "verified=1" in result


def test_settle_persists_claims_to_db(tmp_path):
    from drbrain.loop.events import ResearchState
    from drbrain.storage.database import Database

    db = Database(str(tmp_path / "t.db"))
    try:
        wf = ResearchLoopWorkflow(db=db)
        state = ResearchState(task="flat band", verified=["h1 confirmed"], predictions=["p1"])
        wf._persist_claims(state)
        rows = db.conn.execute(
            "SELECT claim_text, claim_type FROM claims ORDER BY claim_text"
        ).fetchall()
        assert ("h1 confirmed", "Conclusion") in rows
        assert ("p1", "Prediction") in rows
    finally:
        db.close()


def test_settle_no_db_is_noop():
    from drbrain.loop.events import ResearchState

    wf = ResearchLoopWorkflow()
    wf._persist_claims(ResearchState(verified=["x"], predictions=["y"]))


def test_full_loop_persists_verified_claims(tmp_path, monkeypatch):
    from drbrain.storage.database import Database

    db = Database(str(tmp_path / "t.db"))
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {
                "text": '{"entities": ["flat band", "kagome"]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0, "computed": "1.0", "value": 1.0}]}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), db=db, plugins_dir=_write_search_plugin(tmp_path))
    try:

        async def _go() -> str:
            handler = wf.run(task="flat band")
            return await handler

        result = asyncio.run(_go())
        assert "verified=1" in result
        rows = db.conn.execute("SELECT claim_text FROM claims").fetchall()
        assert ("h1",) in rows
    finally:
        db.close()


# ── T4: 实算证据门（结果文件硬产出，杜绝编造数值） ─────────────────────────────


def test_verify_requires_real_job_files(monkeypatch, tmp_path):
    """正路径：compute 工具在场 + job_id 指向真实作业文件（json+log 含数值）→ verified。"""
    jobs = _write_job(tmp_path / "jobs", "job-ok", log_text="converged energy -12.34")
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0, "computed": "-12.34", "value": -12.34, "job_id": "job-ok"}]}'
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert "verified=1" in result
    assert state.verifications[0].status == "verified"
    assert state.verifications[0].job_id == "job-ok"


def test_verify_downgrades_without_job_evidence(monkeypatch, tmp_path):
    """负路径：computed/value 填了数值但没有 job_id → 降级 prediction（防编造核心）。"""
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0, "computed": "1.0", "value": 1.0}]}'
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    # hypotheses=1 且 supports=1/refutes=0 —— 唯一被拦的理由就是 T4 实算证据门
    assert "hypotheses=1" in result
    assert "verified=0" in result
    assert state.verifications[0].status == "prediction"
    assert state.verifications[0].job_id == ""


def test_verify_downgrades_when_job_files_missing(monkeypatch, tmp_path):
    """负路径：job_id 非空但作业文件不存在 → 降级 prediction。"""
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0, "computed": "9.9", "value": 9.9, "job_id": "ghost"}]}'
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert "verified=0" in result
    assert state.verifications[0].status == "prediction"
    assert state.verifications[0].job_id == "ghost"


def test_verify_downgrades_when_job_log_has_no_number(monkeypatch, tmp_path):
    """负路径：作业文件在，但日志里没有任何可 parse 的数值 → 降级 prediction。"""
    jobs = _write_job(tmp_path / "jobs", "job-text", log_text="finished without a numeric output")
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0, "computed": "9.9", "value": 9.9, "job_id": "job-text"}]}'
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert "verified=0" in result
    assert state.verifications[0].status == "prediction"
    assert state.verifications[0].job_id == "job-text"


def test_verify_unchanged_without_compute_tools(monkeypatch, tmp_path):
    """无计算工具时保持现状：没有 run_python 插件 → 老 stub（无 job_id）仍可 verified。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {"text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
             "tool_calls": None, "usage": None},
            {
                "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
                '"orthogonal": 0, "computed": "1.0", "value": 1.0}]}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "verified=1" in result


# ── T7: per-role cross-cycle memory injection ─────────────────────────────────


def test_role_memory_injected_into_node_prompts(monkeypatch, tmp_path):
    """T7: critique/verify user messages include recent per-role history.

    The director writes ``knowledge/role-{critic|verifier}.md``; when the
    workflow is pointed at that dir, the critic and verifier nodes inject the
    recent tail of their own history into the prompt (avoiding repeated
    identical judgments across cycles).
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "role-critic.md").write_text(
        "- [cycle 1] h1（score=0.90, verdict=KEEP）\n"
        "- [cycle 2] h2（score=0.10, verdict=DISCARD）\n",
        encoding="utf-8",
    )
    (knowledge / "role-verifier.md").write_text(
        "- [cycle 1] h1：supports=1, refutes=0, orthogonal=0 → verified\n",
        encoding="utf-8",
    )
    seen: list = []
    calls = [0]
    script = [
        {"text": '{"query": "q1"}', "tool_calls": None, "usage": None},
        {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
        {
            "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
            "tool_calls": None,
            "usage": None,
        },
        {"text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
         "tool_calls": None, "usage": None},
        {
            "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0}]}',
            "tool_calls": None,
            "usage": None,
        },
    ]

    async def fake(messages, models, tools=None, **kw):
        seen.append(list(messages))
        step = script[min(calls[0], len(script) - 1)]
        calls[0] += 1
        return dict(step)

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="some research question", role_memory_dir=str(knowledge))
        return await handler

    asyncio.run(_go())

    def _user_text(messages) -> str:
        parts: list[str] = []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        parts.append(str(blk.get("text", "")))
                    else:
                        parts.append(str(getattr(blk, "text", "")))
        return "\n".join(parts)

    # agent calls: 0 retrieve, 1 extract, 2 identify_gaps, 3 critique, 4 verify,
    # 5 report (report is also agent-backed)
    assert len(seen) == 6
    critic_text = _user_text(seen[3])
    verify_text = _user_text(seen[4])
    assert "以往轮次评审过的假设" in critic_text
    assert "h2（score=0.10, verdict=DISCARD）" in critic_text
    assert "以往轮次核验过的假设" in verify_text
    assert "supports=1, refutes=0" in verify_text
