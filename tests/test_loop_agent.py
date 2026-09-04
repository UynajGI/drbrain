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
from drbrain.loop.workflow import ComputeToolsUnavailableError

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


def _write_job(
    run_dir, job_id: str, log_text: str = '{"value": 1.0, "quantity": "q", "unit": "u"}'
) -> Path:
    """Pre-write a fake async job's on-disk artifacts (``<job_id>.json`` + ``<job_id>.log``).

    Mirrors what ``run_python(mode=async)`` leaves in ``$DRBRAIN_RUN_DIR``: a
    meta json (pid / log_path) plus the captured stdout log. The log carries the
    T4 result contract (a JSON with a finite numeric ``value``).
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


def _compute_loop_script(verify_text: str, compute_text: str) -> list[dict]:
    """Full agent-backed loop script with compute tools present (run_python).

    Node order is retrieve → extract → identify_gaps → critique → **compute** →
    verify → report. The compute node (ROLE-GPU) reports per-hypothesis
    ``job_id``s before verify; verify consumes those for its T4 gate, so the
    verify stub only carries evidence counts.
    """
    return [
        {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
        {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
        {
            "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
            "tool_calls": None,
            "usage": None,
        },
        {
            "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
            "tool_calls": None,
            "usage": None,
        },
        {"text": compute_text, "tool_calls": None, "usage": None},
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
                    '"hypotheses": [{"statement": "h1", "prediction": "p1", '
                    '"falsification": "f1", "conditions": {}}, '
                    '{"statement": "h2", "prediction": "p2", '
                    '"falsification": "f2", "conditions": {}}]}'
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


def test_identify_gaps_no_filler_when_agent_empty(monkeypatch, tmp_path):
    """Agent 提不出假设 → 空轮次，绝不塞占位（ROLE-ANALYST Rule 2）。

    The old fallback fabricated a "缺少关于X的机制" gap + placeholder hypothesis,
    which the critic then DISCARDed at ~0 score — pure NO_GAIN churn. Now an
    empty agent answer stays empty: the cycle goes NO_GAIN with zero filler.
    """
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {"text": '{"gaps": [], "hypotheses": []}', "tool_calls": None, "usage": None},
            {"text": "cycle report", "tool_calls": None, "usage": None},
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "gaps=0" in result
    assert "hypotheses=0" in result
    assert "verified=0" in result
    assert "缺少" not in result  # no placeholder ever entered the report


def test_identify_gaps_drops_hypothesis_without_prediction(monkeypatch, tmp_path):
    """Analyst gate（ROLE-ANALYST Step 0.3）：缺 prediction 的「假设」不是假设，代码直接过滤。

    A hypothesis without a falsifiable prediction is dropped in code — only the
    well-formed one survives to the critic.
    """
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": (
                    '{"gaps": [], "hypotheses": ['
                    '{"statement": "h-no-prediction", "conditions": {}}, '
                    '{"statement": "h-ok", "prediction": "p", '
                    '"falsification": "f", "conditions": {}}]}'
                ),
                "tool_calls": None,
                "usage": None,
            },
            {"text": '{"hypotheses": []}', "tool_calls": None, "usage": None},
            {"text": "cycle report", "tool_calls": None, "usage": None},
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "hypotheses=1" in result


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
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
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
        # v12 knowledge_snapshots 的生产写入者：settle 落库即登记快照
        snaps = db.conn.execute(
            "SELECT snapshot_id, revision_id, description FROM knowledge_snapshots"
        ).fetchall()
        assert len(snaps) == 1
        assert snaps[0][0].startswith("snap-")
        assert "verified=1" in snaps[0][2] and "predictions=1" in snaps[0][2]
        # 幂等：同一 outcome 重放不再新增行
        wf._persist_claims(state)
        snaps_after = db.conn.execute("SELECT COUNT(*) FROM knowledge_snapshots").fetchone()[0]
        assert snaps_after == 1
    finally:
        db.close()


def test_record_claim_preserves_provenance_when_re_recorded_without_it(tmp_path):
    """OCR r5 bug·high：无溯源参数的重写不得抹掉已有 v19 审计列。"""
    from drbrain.storage.database import Database

    db = Database(str(tmp_path / "t.db"))
    try:
        cid = db.record_claim(
            "task",
            "statement X",
            claim_type="Conclusion",
            run_id="run-1",
            cycle=2,
            job_id="job-9",
            claim_ledger_id="cl-abc",
            model="test-model",
            prompt_hash="ph-1",
            evidence_node_ids="n1,n2",
        )
        # record_answer 走的路径：不带任何 v19 参数重写同一 claim
        db.record_claim("task", "statement X", claim_type="Conclusion", confidence=0.5)
        row = db.conn.execute(
            "SELECT run_id, cycle, job_id, claim_ledger_id, model, prompt_hash, "
            "evidence_node_ids FROM claims WHERE claim_id = ?",
            (cid,),
        ).fetchone()
        assert row == ("run-1", 2, "job-9", "cl-abc", "test-model", "ph-1", "n1,n2")

        # 带新值的重写仍然生效（显式覆盖优先）
        db.record_claim(
            "task", "statement X", claim_type="Conclusion", run_id="run-2", job_id="job-10"
        )
        row2 = db.conn.execute(
            "SELECT run_id, job_id, model FROM claims WHERE claim_id = ?", (cid,)
        ).fetchone()
        assert row2 == ("run-2", "job-10", "test-model")
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
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
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
    """正路径：compute 节点产出的 job_id 指向真实作业文件（json+log 含数值）→ verified。"""
    jobs = _write_job(
        tmp_path / "jobs",
        "job-ok",
        log_text='{"value": -12.34, "quantity": "energy", "unit": "eV"}',
    )
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0}]}',
            '{"results": [{"statement": "h1", "job_id": "job-ok", "computed": "-12.34"}]}',
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir, n_critics=1)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert "verified=1" in result
    assert state.verifications[0].status == "verified"
    assert state.verifications[0].job_id == "job-ok"
    # the compute node's summary rides into the verification's computed field
    assert state.verifications[0].computed == "-12.34"


def test_verify_downgrades_without_job_evidence(monkeypatch, tmp_path):
    """负路径：compute 节点没产出任何 job_id → 降级 prediction（防编造核心）。"""
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0}]}',
            '{"results": []}',
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir, n_critics=1)

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
    """负路径：compute 节点报了 job_id 但作业文件不存在 → 降级 prediction。"""
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0}]}',
            '{"results": [{"statement": "h1", "job_id": "ghost"}]}',
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir, n_critics=1)

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
    """负路径：compute 的作业文件在，但日志里没有任何可 parse 的数值 → 降级 prediction。"""
    jobs = _write_job(tmp_path / "jobs", "job-text", log_text="finished without a numeric output")
    monkeypatch.setenv("DRBRAIN_RUN_DIR", str(jobs))
    _scripted_llm(
        monkeypatch,
        _compute_loop_script(
            '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, '
            '"orthogonal": 0}]}',
            '{"results": [{"statement": "h1", "job_id": "job-text"}]}',
        ),
    )
    plugins_dir = _write_search_plugin(tmp_path)
    _write_compute_plugin(tmp_path)
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=plugins_dir, n_critics=1)

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
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
                "tool_calls": None,
                "usage": None,
            },
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


def test_strict_compute_mode_pauses_without_compute_tools(monkeypatch, tmp_path):
    """T4 严格模式：有候选但无计算工具 → 硬暂停报错，不再静默走无实算路径。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        require_compute_tools=True,
    )

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    with pytest.raises(ComputeToolsUnavailableError):
        asyncio.run(_go())


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
            "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
            "tool_calls": None,
            "usage": None,
        },
        {
            "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
            "tool_calls": None,
            "usage": None,
        },
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

    # agent calls: 0 retrieve, 1 extract, 2 identify_gaps, 3 critic-1, 4 critic-2,
    # 5 critic-3, 6 verify, 7 report (report is also agent-backed)
    assert len(seen) == 8
    critic_text = _user_text(seen[3])
    verify_text = _user_text(seen[6])
    assert "以往轮次评审过的假设" in critic_text
    assert "h2（score=0.10, verdict=DISCARD）" in critic_text
    assert "以往轮次核验过的假设" in verify_text
    assert "supports=1, refutes=0" in verify_text


# ── Discussion-Before-Queuing 门（多 critic 异步评论 + 非作者门 + 入队） ───────


def test_critique_discussion_gate_multiple_critics(monkeypatch, tmp_path):
    """讨论门正路径：多个 critic 并发评论，非作者评论满足门 → 入队（可 claim）。"""
    from drbrain.loop.discussion import POST_PROPOSAL

    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", '
                '"prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.9, "flaw": "minor"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.8, "flaw": "ok"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.7, "flaw": "fine"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"verifications": [{"statement": "h1", "supports": 1, '
                '"refutes": 0, "orthogonal": 0}]}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go():
        handler = wf.run(task="flat band")
        return await handler

    asyncio.run(_go())

    # 讨论门：h1 收到 3 条非作者评论（critic-1/2/3）。
    proposals = wf._board.list_posts(POST_PROPOSAL)
    assert len(proposals) == 1
    non_author = wf._board.non_author_comments(proposals[0].id, author="analyst")
    assert len(non_author) == 3

    # 入队且 discussion_pending=False（可被 compute claim）。
    pending = wf._queue.list_pending()
    claimed = wf._queue.list_claimed()
    queued = pending + claimed
    assert len(queued) == 1
    assert all(not i.discussion_pending for i in queued)
    # 3 个 critic 打分均值 (0.9+0.8+0.7)/3 = 0.8 → critiqued。
    assert queued[0].hypothesis.status == "critiqued"


def test_critique_discards_low_scored_proposal(monkeypatch, tmp_path):
    """讨论门负路径：所有非作者 reviewer 打低分 → DISCARD，不入队。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", '
                '"prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.1, "flaw": "weak"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.2, "flaw": "vague"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": 0.3, "flaw": "thin"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"verifications": []}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    _, state = asyncio.run(_go())
    # mean (0.1+0.2+0.3)/3 = 0.2 < 0.4 → 全体 DISCARD → 不入队。
    assert len(wf._queue.list_pending()) + len(wf._queue.list_claimed()) == 0
    assert all(h.status == "discarded" for h in state.hypotheses)


def test_critique_pending_when_no_non_author_comment(monkeypatch, tmp_path):
    """讨论门负路径：critic 无有效评论 → 不满足门 → discussion_pending=True。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", '
                '"prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {"text": "not json", "tool_calls": None, "usage": None},
            {"text": "not json", "tool_calls": None, "usage": None},
            {
                "text": '{"verifications": []}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path), n_critics=1)

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    _, state = asyncio.run(_go())
    # critic 唯一一次评审返回非 JSON（retry 后仍失败）→ 无非作者评论 → pending。
    queued = wf._queue.list_pending() + wf._queue.list_claimed()
    assert len(queued) == 1
    assert queued[0].discussion_pending is True
    assert all(h.status == "proposed" for h in state.hypotheses)


# ── 通用结果契约 + claim 置信度诚实 + 脏输出鲁棒性 ────────────────────────────


def test_t4_gate_rejects_domain_keys_and_accepts_value_contract(tmp_path):
    """T4 结果契约：只认 ``value``（有限数值）；领域专名键一律无效。"""
    from drbrain.loop.workflow import _extract_result_payload

    # 新契约：value 有限数值 → 通过
    assert _extract_result_payload('{"quantity": "gap", "value": 0.3, "unit": "eV"}') is not None
    # 字符串 / bool / 缺失 value → 拒绝
    assert _extract_result_payload('{"value": "big"}') is None
    assert _extract_result_payload('{"value": true}') is None
    assert _extract_result_payload('{"quantity": "gap"}') is None
    # 任何领域专名键（旧竞赛残留）都不是证据
    assert _extract_result_payload('{"min_bandwidth_ev": 1.0}') is None


def test_persist_claims_confidence_by_evidence_shape(tmp_path):
    """无实算产物的 Conclusion/Rejected 置信度封顶 0.6，Prediction 0.3；
    job_id 通过 T4 落盘校验才 1.0（OCR r6：字符串本身不是证据）。"""
    from drbrain.loop.events import ResearchState, Verification
    from drbrain.storage.database import Database

    db = Database(str(tmp_path / "t.db"))
    jobs = _write_job(tmp_path / "jobs", "job-9")
    try:
        wf = ResearchLoopWorkflow(db=db, jobs_dir=str(jobs))
        state = ResearchState(
            task="generic research task",
            verified=["lit-supported conclusion"],
            falsified=["lit-countered claim"],
            predictions=["bare guess"],
            verifications=[
                Verification(statement="lit-supported conclusion", supports=1, job_id=""),
                Verification(statement="lit-countered claim", refutes=2, job_id=""),
            ],
        )
        wf._persist_claims(state)
        rows = dict(db.conn.execute("SELECT claim_text, confidence FROM claims").fetchall())
        assert rows["lit-supported conclusion"] == pytest.approx(0.6)
        assert rows["lit-countered claim"] == pytest.approx(0.6)
        assert rows["bare guess"] == pytest.approx(0.3)

        # 无产物的 job_id（LLM 抄写字符串）不算实算证据
        state_ghost = ResearchState(
            task="generic research task",
            verified=["ghost-backed conclusion"],
            verifications=[
                Verification(statement="ghost-backed conclusion", supports=1, job_id="job-ghost")
            ],
        )
        wf._persist_claims(state_ghost)
        rows_ghost = dict(db.conn.execute("SELECT claim_text, confidence FROM claims").fetchall())
        assert rows_ghost["ghost-backed conclusion"] == pytest.approx(0.6)

        state2 = ResearchState(
            task="generic research task",
            verified=["job-backed conclusion"],
            verifications=[
                Verification(statement="job-backed conclusion", supports=1, job_id="job-9")
            ],
        )
        wf._persist_claims(state2)
        rows2 = dict(db.conn.execute("SELECT claim_text, confidence FROM claims").fetchall())
        assert rows2["job-backed conclusion"] == pytest.approx(1.0)
    finally:
        db.close()


def test_critic_non_numeric_score_does_not_crash(monkeypatch, tmp_path):
    """脏输出：critic 把 score 写成字符串（"high"）→ 不崩溃，按 0 分处理走 DISCARD。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "q"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", '
                '"prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": '{"hypotheses": [{"statement": "h1", "score": "high", "flaw": "x"}]}',
                "tool_calls": None,
                "usage": None,
            },
            {"text": "cycle report", "tool_calls": None, "usage": None},
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path), n_critics=1)

    async def _go():
        handler = wf.run(task="some research question")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())  # 之前这里会 ValueError → 整轮失败
    assert "hypotheses=1" in result
    assert all(h.status == "discarded" for h in state.hypotheses)


def test_critic_join_by_claim_id_survives_statement_paraphrase(monkeypatch, tmp_path):
    """脏输出：critic 改写了 statement 但原样回抄 claim_id → 评论仍能匹配上（不再作废）。"""
    from drbrain.loop.workflow import _claim_id

    claim_id = _claim_id("h1")
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "q"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["g1"], "hypotheses": [{"statement": "h1", '
                '"prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            {
                "text": json.dumps(
                    {
                        "hypotheses": [
                            {
                                "claim_id": claim_id,
                                "statement": "h1 改写版：机制完全不同的复述",
                                "score": 0.9,
                                "flaw": "fine, kept with residual risk noted",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                "tool_calls": None,
                "usage": None,
            },
            {"text": "cycle report", "tool_calls": None, "usage": None},
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path), n_critics=1)

    async def _go():
        handler = wf.run(task="some research question")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    _, state = asyncio.run(_go())
    # statement 字面已对不上，但 claim_id 命中 → 评审生效（critiqued 而非 pending）。
    assert all(h.status == "critiqued" for h in state.hypotheses)
    # §7.3: 最终分 = 0.7×0.9(critic) + 0.3×0.5(novelty unknown 中性) = 0.78
    assert state.hypotheses[0].score == pytest.approx(0.78)


def test_retrieve_reports_rag_status_in_state(monkeypatch, tmp_path):
    """检索故障进 state.retrieval_status：报告可见，绝不与「语料无内容」混淆。"""
    _scripted_llm(monkeypatch, [{"text": '{"query": "q"}', "tool_calls": None, "usage": None}])
    wf = ResearchLoopWorkflow(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        evidence_recorder=lambda _bundle: None,  # 使 RAG 路径可用（否则是 unavailable）
    )
    # 强制 RAG 路径以 error 收场：generation 存在 → 进入检索，retrieve_documents 抛异常
    wf._rag_generation = "gen-1"
    monkeypatch.setattr(
        "drbrain.rag.agent.retrieve_documents",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("index outage")),
    )

    async def _go():
        handler = wf.run(task="some research question")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert state.retrieval_status == "error"
    assert "rag=error" in result


def test_dirty_llm_outputs_survive_the_cycle(monkeypatch, tmp_path):
    """L-I8: 端到端「故意脏输出」用例——critic 给字符串分数、verifier 返回空
    verifications、语句被改写。循环必须完成且候选降级为 prediction，不允许
    ValueError 崩轮、也不允许候选无声蒸发。"""
    _scripted_llm(
        monkeypatch,
        [
            {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
            {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
            {
                "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
                "tool_calls": None,
                "usage": None,
            },
            # 脏输出①：score 是字符串且 claim_id 被改写
            {
                "text": '{"hypotheses": [{"claim_id": "cl-WRONG", "statement": "h1 paraphrased!", "score": "high", "flaw": "looks fine overall but unverified"}]}',
                "tool_calls": None,
                "usage": None,
            },
            # 脏输出②：verifications 为空列表（L-E8：不算 handled）
            {
                "text": '{"verifications": []}',
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path))

    async def _go():
        handler = wf.run(task="flat band")
        result = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        return result, state

    result, state = asyncio.run(_go())
    assert state is not None
    assert not state.verifications  # empty counts are not evidence
    assert state.hypotheses, "candidates must not silently vanish (L-E8)"
    # 脏 claim_id + 改写语句 → 评论无法归属 → discussion_pending（诚实降级，
    # 留给下一轮重审），而不是崩溃或无声消失。
    assert all(
        h.status in ("prediction", "critiqued", "discarded", "proposed") for h in state.hypotheses
    )
    assert "verified=0" in result


def test_structured_prediction_code_falsification(tmp_path):
    """§7.1: 结构化预测由代码判定——value vs threshold（含单位换算），计数仅旁证。

    T4 必须先过：数值证据必须来自真实作业日志，否则一律 prediction。
    """
    import json as _json

    from drbrain.loop.events import Hypothesis, Verification
    from drbrain.loop.workflow import _classify_verification

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "job-42.log").write_text('{"quantity": "gap", "value": 0.31}', encoding="utf-8")
    (jobs / "job-42.json").write_text(
        _json.dumps({"job_id": "job-42", "pid": 99999999, "log_path": str(jobs / "job-42.log")}),
        encoding="utf-8",
    )
    h = Hypothesis(
        statement="H",
        quantity="gap",
        comparator="<",
        threshold=0.5,
        unit="eV",
    )
    ver = Verification(
        statement="H",
        job_id="job-42",
        value=0.31,
        unit="eV",
        supports=1,
        refutes=0,
        computed="0.31",
    )
    run_dir = str(jobs)  # 生产里 _jobs_dir 即 jobs 目录本身

    # 数值 0.31 < 0.5（以日志落盘值为准）→ 判支持
    assert _classify_verification(ver, 0.9, True, run_dir, hypothesis=h) == "verified"

    # review round-3: LLM 谎报 value=0.71 也无效——判定只看日志里的 0.31
    ver_bad = Verification(
        statement="H",
        job_id="job-42",
        value=0.71,
        unit="eV",
        supports=1,
        refutes=0,
        computed="0.71",
    )
    assert _classify_verification(ver_bad, 0.9, True, run_dir, hypothesis=h) == "verified"

    # 数值 0.71 ≥ 0.5（第二个作业日志落盘 0.71）→ 代码证伪压过 supports=1
    (jobs / "job-43.log").write_text('{"quantity": "gap", "value": 0.71}', encoding="utf-8")
    (jobs / "job-43.json").write_text(
        _json.dumps({"job_id": "job-43", "pid": 99999999, "log_path": str(jobs / "job-43.log")}),
        encoding="utf-8",
    )
    ver_bad2 = Verification(
        statement="H",
        job_id="job-43",
        value=0.31,
        unit="eV",
        supports=1,
        refutes=0,
        computed="0.71",
    )
    assert _classify_verification(ver_bad2, 0.9, True, run_dir, hypothesis=h) == "falsified"

    # 幻觉 job_id（无落盘日志）→ structured None → T4 拦下 → prediction
    ver_phantom = Verification(
        statement="H",
        job_id="job-404",
        value=0.31,
        unit="eV",
        supports=1,
        refutes=0,
        computed="0.31",
    )
    assert _classify_verification(ver_phantom, 0.9, True, run_dir, hypothesis=h) == "prediction"

    # 混杂计数 + 可机检数值 + 真实作业 → 代码判定说了算
    ver_mixed = Verification(
        statement="H",
        job_id="job-42",
        value=0.31,
        unit="eV",
        supports=2,
        refutes=1,
        computed="0.31",
    )
    assert _classify_verification(ver_mixed, 0.9, True, run_dir, hypothesis=h) == "verified"

    # 预测单位 eV，作业单位 eV（同单位）→ 直接比较 0.31 < 0.5 → 支持
    ver_mev = Verification(
        statement="H",
        job_id="job-42",
        value=0.31,
        unit="eV",
        supports=1,
        refutes=0,
        computed="0.31 eV",
    )
    assert _classify_verification(ver_mev, 0.9, True, run_dir, hypothesis=h) == "verified"

    # 无结构化字段 → 走原计数路径
    h_free = Hypothesis(statement="H")
    assert _classify_verification(ver, 0.9, True, run_dir, hypothesis=h_free) == "verified"
