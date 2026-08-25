"""ResearchDirector tests — the continuous research loop (AutoScientists-style).

The director runs the 12-node workflow repeatedly until stagnation, keeping a
checkpointed champion/rejected/results state and a running report. LLM stubbed
via ``llm_client.acall_with_messages`` (offline, mirrors test_loop_agent.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.loop import ResearchDirector
from drbrain.loop.director import _default_state

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None
pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]


def _cfg() -> Config:
    c = Config(llamaindex=LlamaIndexConfig(enabled=True))
    c.llm.models = list(MODELS)
    c.api.cache_ttl = 0
    c.llamaindex.storage_dir = "/nonexistent-rag-index"
    return c


def test_checkpoint_manifest_tracks_typed_and_dict_llm_fallback_chains():
    """Resume rejects a changed CLI LLM chain without storing credentials."""
    typed_cfg = _cfg()
    typed_cfg.llm.models = [
        {"provider": "openai", "model": "gpt-4o", "api_key": "secret", "base_url": "a"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet", "api_key": "secret-2"},
    ]

    typed_manifest = ResearchDirector(cfg=typed_cfg)._checkpoint_manifest()
    assert typed_manifest.model_manifest["models"] == [
        {"provider": "openai", "model": "gpt-4o", "base_url": "a"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet"},
    ]
    assert "api_key" not in str(typed_manifest.model_manifest)

    dict_manifest = ResearchDirector(
        cfg={"llm": {"models": [{"provider": "openai", "model": "gpt-4o-mini"}]}}
    )._checkpoint_manifest()
    assert dict_manifest.model_manifest["models"] == [
        {"provider": "openai", "model": "gpt-4o-mini"}
    ]
    assert dict_manifest.model_manifest != typed_manifest.model_manifest


def _cyclic_llm(monkeypatch, script):
    """Scripted LLM that cycles through ``script`` (one entry per agent call)."""
    calls = [0]

    async def fake(messages, models, tools=None, **kw):
        step = script[calls[0] % len(script)]
        calls[0] += 1
        return dict(step)

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    return fake


_CYCLE_SCRIPT = [
    {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},  # retrieve distill
    {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},  # extract
    # identify_gaps: h1 gets verified in cycle 1 (champion); h2 is always
    # proposed (never a dup) and always DISCARDed by the critic, keeping the
    # no-gain → critic-veto → Phase 4 adapt path alive in later cycles.
    {
        "text": '{"gaps": ["gap1"], "hypotheses": ['
        '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
        '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
        "tool_calls": None,
        "usage": None,
    },  # critique
    {
        "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0, "computed": "1.0", "value": 1.0}]}',
        "tool_calls": None,
        "usage": None,
    },  # verify
    {"text": "cycle report", "tool_calls": None, "usage": None},  # report
]

# Same cycle, but the compute node stub carries the job_id of a real (pre-written)
# async job — the T4 evidence gate (fed from the Computed event) must accept it.
#
# The stub is LINEAR per cycle on purpose: with compute tools present the cycle
# makes 7 agent calls in cycle 1 (… critique → compute → verify → report) and 5
# in later cycles (compute/verify skip once the critic DISCARDs h2). A cyclic
# stub (script[calls % len]) would drift out of alignment, so the script below
# spells out the exact per-cycle call sequence; the run stops at cycle 3, i.e.
# exactly 7 + 5 + 5 = 17 calls, so no wrap-around ever happens.
_CYCLE_SCRIPT_COMPUTE = [
    # cycle 1: retrieve → extract → identify_gaps → critique → compute → verify → report
    {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
    {
        "text": '{"gaps": ["gap1"], "hypotheses": ['
        '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
        '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"results": [{"statement": "h1", "job_id": "job-1"}]}',
        "tool_calls": None,
        "usage": None,
    },  # compute
    {
        "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0, "computed": "1.0", "value": 1.0}]}',
        "tool_calls": None,
        "usage": None,
    },  # verify
    {"text": "cycle report", "tool_calls": None, "usage": None},  # report
    # cycle 2: retrieve → extract → identify_gaps → critique → report (h1 is a
    # champion dup → dropped; h2 is proposed and DISCARDed by the critic → no
    # compute/verify calls)
    {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
    {
        "text": '{"gaps": ["gap1"], "hypotheses": ['
        '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
        '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
        "tool_calls": None,
        "usage": None,
    },
    {"text": "cycle report", "tool_calls": None, "usage": None},
    # cycle 3: same shape as cycle 2 — no_gain hits the stagnation bar and the
    # critic's all-DISCARD round vetoes the direction → Phase 4 adapt → stop
    {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},
    {
        "text": '{"gaps": ["gap1"], "hypotheses": ['
        '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
        '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
        "tool_calls": None,
        "usage": None,
    },
    {"text": "cycle report", "tool_calls": None, "usage": None},
]


def _write_compute_plugin(tmp_path) -> str:
    """Drop a mock ``run_python`` plugin so ``_has_compute_tools()`` turns on."""
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


def _write_search_plugin(tmp_path) -> str:
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


# ── unit: state classification ────────────────────────────────────────────────


def test_absorb_classifies_champion_and_rejected():
    from drbrain.loop.events import ResearchState

    d = ResearchDirector(cfg=_cfg())
    state = _default_state("t")
    rs = ResearchState(
        verified=["h1"],
        falsified=["h2"],
        predictions=["p1"],
        hypotheses=[_h("h1"), _h("h2"), _h("h3 unresolved")],
    )
    d._absorb(state, "report", rs)
    assert [c["statement"] for c in state["champion"]] == ["h1"]
    assert "h2" in state["rejected"]  # h2 was falsified (DISCARD)
    assert "h1" not in state["rejected"]  # h1 was verified (KEEP)
    assert "h3 unresolved" not in state["rejected"]  # not falsified → not a dead end
    assert state["consecutive_no_gain"] == 0


def test_absorb_no_gain_increments_stagnation():
    from drbrain.loop.events import ResearchState

    d = ResearchDirector(cfg=_cfg())
    state = _default_state("t")
    state["champion"].append({"statement": "h1 confirmed", "cycle": 1, "confidence": 1.0})
    # same verified conclusion again → no new champion → stagnation++
    rs = ResearchState(verified=["h1 confirmed"], hypotheses=[_h("h1 confirmed")])
    d._absorb(state, "report", rs)
    assert state["consecutive_no_gain"] == 1
    assert len(state["champion"]) == 1


def _h(statement: str):
    from drbrain.loop.events import Hypothesis

    return Hypothesis(statement=statement)


def test_build_prior_context_lists_champion_and_rejected():
    state = _default_state("t")
    state["champion"] = [{"statement": "A 是 B", "cycle": 1, "confidence": 1.0}]
    state["rejected"] = ["X 是 Y"]
    prior = ResearchDirector._build_prior_context(state)
    assert "A 是 B" in prior
    assert "X 是 Y" in prior


def test_checkpoint_roundtrip(tmp_path):
    d = ResearchDirector(cfg=_cfg(), run_dir=str(tmp_path))
    state = _default_state("topological flat band")
    state["cycles"] = 2
    state["champion"].append({"statement": "h1", "cycle": 1, "confidence": 1.0})
    state["rejected"].append("h0 dead end")
    d._save_state("topological flat band", state)
    loaded = d._load_state("topological flat band")
    assert loaded["cycles"] == 2
    assert loaded["champion"][0]["statement"] == "h1"
    assert loaded["rejected"] == ["h0 dead end"]
    # canonical files exist (AutoScientists-style workspace, not a JSON blob)
    td = tmp_path / "topological-flat-band"
    assert (td / "champion.md").exists()
    assert (td / "dead_ends.md").exists()
    assert (td / "knowledge" / "patterns.md").exists()
    assert (td / "run.json").exists()


# ── integration: continuous loop to stagnation ────────────────────────────────


@pytest.mark.timeout(180)
def test_director_runs_cycles_to_stagnation(monkeypatch, tmp_path):
    _cyclic_llm(monkeypatch, _CYCLE_SCRIPT)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )

    async def _go():
        return await d.run(
            "topological flat band", max_cycles=10, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())

    # h1 verified every cycle → new champion in cycle 1, then no-gain → stagnation
    # → Phase 4 adapt #1 (pivot) → max_adaptations=1 → stop at cycle 3.
    assert state["cycles"] == 3
    assert [c["statement"] for c in state["champion"]] == ["h1"]
    assert state["adaptations"] == 1
    assert state["consecutive_no_gain"] == 0  # reset after pivot
    assert len(state["results"]) == 3
    # per-cycle evidence + canonical files + logs exist on disk
    run_dir = tmp_path / "runs" / "topological-flat-band"
    assert (run_dir / "champion.md").exists()
    assert (run_dir / "results" / "cycle-001.md").exists()
    assert (run_dir / "results" / "cycle-003.md").exists()
    assert (run_dir / "run.json").exists()
    # AutoScientists-style canonical logs: experiments.jsonl + sessions.jsonl
    assert (run_dir / "logs" / "experiments.jsonl").exists()
    assert (run_dir / "logs" / "sessions.jsonl").exists()
    import json

    exp_lines = [
        json.loads(ln) for ln in (run_dir / "logs" / "experiments.jsonl").read_text().splitlines()
    ]
    assert len(exp_lines) == 3
    assert {e["outcome"] for e in exp_lines} == {"KEEP", "NO_GAIN"}


@pytest.mark.timeout(180)
def test_director_honors_job_evidence_gate(monkeypatch, tmp_path):
    """T4: the director's DRBRAIN_RUN_DIR wiring feeds the verify job-evidence gate.

    Compute plugin present (run_python) + job_id pointing at real on-disk job
    artifacts → the verifier's claim passes the gate and h1 is KEEP (champion).
    Without the artifacts the same stub would be downgraded to prediction.
    """
    _cyclic_llm(monkeypatch, _CYCLE_SCRIPT_COMPUTE)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )
    _write_compute_plugin(tmp_path)
    # The director points DRBRAIN_RUN_DIR at <run_dir>/<topic>/jobs before the
    # cycle starts; pre-write the job artifacts there so the gate finds them.
    jobs_dir = tmp_path / "runs" / "topological-flat-band" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "job-1.log").write_text("computed 1.0", encoding="utf-8")
    (jobs_dir / "job-1.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "pid": 99999999,  # dead pid → job finished (T4 gate requires completion)
                "log_path": str(jobs_dir / "job-1.log"),
            }
        ),
        encoding="utf-8",
    )

    async def _go():
        return await d.run(
            "topological flat band", max_cycles=10, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())

    assert [c["statement"] for c in state["champion"]] == ["h1"]
    assert state["cycles"] == 3
    assert state["adaptations"] == 1


# ── T7: per-role cross-cycle memory (critic / verifier) ───────────────────────


@pytest.mark.timeout(180)
def test_director_writes_role_memory_files(monkeypatch, tmp_path):
    """T7: after each cycle the director appends critic/verifier history to knowledge/."""
    script = [
        {"text": '{"query": "q1"}', "tool_calls": None, "usage": None},
        {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
        {
            "text": '{"gaps": ["g1"], "hypotheses": ['
            '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
            '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
            "tool_calls": None,
            "usage": None,
        },
        {
            "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
            "tool_calls": None,
            "usage": None,
        },
        {
            "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0}]}',
            "tool_calls": None,
            "usage": None,
        },
        {"text": "cycle report", "tool_calls": None, "usage": None},
    ]
    _cyclic_llm(monkeypatch, script)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )

    async def _go():
        return await d.run(
            "a general research question", max_cycles=10, stagnation_cycles=2, max_adaptations=1
        )

    asyncio.run(_go())

    run_dir = tmp_path / "runs" / "a-general-research-question"
    critic = run_dir / "knowledge" / "role-critic.md"
    verifier = run_dir / "knowledge" / "role-verifier.md"
    assert critic.exists()
    assert verifier.exists()
    critic_text = critic.read_text(encoding="utf-8")
    verifier_text = verifier.read_text(encoding="utf-8")
    # critic history: hypotheses + score + verdict; verifier history: counts + status
    assert "[cycle 1] h1" in critic_text
    assert "score=0.90" in critic_text
    assert "verdict=KEEP" in critic_text
    assert "[cycle 1] h1" in verifier_text
    assert "supports=1" in verifier_text
    assert "→ verified" in verifier_text
    # h1 is a champion dup from cycle 2 on (T6) → only h2 gets re-proposed and
    # DISCARDed; the verifier records real evidence only in cycle 1 (h1).
    # cycle 1: h1 (KEEP 0.90) + h2 (DISCARD 0.00) → 2 critic lines; cycles 2-3:
    # h2 DISCARD → 1 critic line each → 4 total; verifier stays at 1.
    assert critic_text.count("[cycle") == 4
    assert verifier_text.count("[cycle") == 1


# ── T8: proposal board + critic reviews（Discussion-Before-Queuing 落盘） ───────

_DISCUSSION_SCRIPT = [
    {"text": '{"query": "q1"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
    {
        "text": '{"gaps": ["gap1"], "hypotheses": ['
        '{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}, '
        '{"statement": "h2", "prediction": "p2", "falsification": "f2", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"verifications": [{"statement": "h1", "supports": 1, "refutes": 0, "orthogonal": 0}]}',
        "tool_calls": None,
        "usage": None,
    },
    {"text": "cycle report", "tool_calls": None, "usage": None},
]


@pytest.mark.timeout(180)
def test_director_persists_proposals_and_reviews(monkeypatch, tmp_path):
    """T8: each cycle's hypotheses land in knowledge/proposals.md (propose role)
    and the critic's score+verdict land in knowledge/reviews.md (non-author role)."""
    _cyclic_llm(monkeypatch, _DISCUSSION_SCRIPT)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )

    async def _go():
        return await d.run(
            "a research question", max_cycles=2, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())
    assert state["cycles"] == 2  # bounded by max_cycles, no pivot needed

    knowledge = tmp_path / "runs" / "a-research-question" / "knowledge"
    proposals = knowledge / "proposals.md"
    reviews = knowledge / "reviews.md"
    assert proposals.exists()
    assert reviews.exists()
    p_text = proposals.read_text(encoding="utf-8")
    r_text = reviews.read_text(encoding="utf-8")
    # cycle 1: the scripted proposal + its critic review (KEEP, high score).
    # The reviewer is explicitly the critic role — the non-author reviewer
    # independent of the propose role that authored the proposal.
    assert "- [cycle 1] h1" in p_text
    assert "- [cycle 1] h1（reviewer=critic, score=0.90, verdict=KEEP）" in r_text
    assert "reviewer=critic" in r_text
    # cycle 2: h1 是 champion dup（T6）→ 只重新提 h2；critic 没评 h2 →
    # discussion_pending（未获非作者评论）→ 只落盘 proposals.md，不落 reviews.md。
    assert "- [cycle 2] h2" in p_text
    assert "h2" not in r_text  # pending 的 h2 没有 review


# ── T8: endorsement — Phase 4 adapt needs the critic's independent veto ────────


def test_critic_vetoes_direction_unit():
    """T8 unit: the endorsement gate reads the critic's latest review round."""
    from drbrain.loop.events import Hypothesis

    def _rs(*hypotheses):
        from drbrain.loop.events import ResearchState

        return ResearchState(hypotheses=list(hypotheses))

    def _h(statement, score, status):
        return Hypothesis(statement=statement, score=score, status=status)

    # critic endorses: high mean, nothing discarded → NO veto → no pivot
    assert ResearchDirector._critic_vetoes_direction(_rs(_h("h1", 0.9, "critiqued"))) is False
    assert (
        ResearchDirector._critic_vetoes_direction(
            _rs(_h("h1", 0.5, "critiqued"), _h("h2", 0.6, "critiqued"))
        )
        is False
    )
    # critic vetoes: mean score below the endorsement bar → veto
    assert ResearchDirector._critic_vetoes_direction(_rs(_h("h1", 0.2, "critiqued"))) is True
    # critic vetoes: every hypothesis of the round was DISCARDed → veto
    assert ResearchDirector._critic_vetoes_direction(_rs(_h("h1", 0.9, "discarded"))) is True
    # no critic opinion → not a veto (a structural change needs an explicit
    # independent objection)
    assert ResearchDirector._critic_vetoes_direction(_rs()) is False
    assert ResearchDirector._critic_vetoes_direction(None) is False


# Script: h1 is always proposed and survives the critic with score 0.9 (KEEP),
# but verification always yields a prediction (supports=0) — never a champion,
# so no-gain climbs while the critic keeps endorsing the direction.
_ENDORSE_SCRIPT = [
    {"text": '{"query": "q1"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
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
        "text": '{"verifications": [{"statement": "h1", "supports": 0, "refutes": 0, "orthogonal": 1}]}',
        "tool_calls": None,
        "usage": None,
    },
    {"text": "cycle report", "tool_calls": None, "usage": None},
]


@pytest.mark.timeout(180)
def test_director_keeps_cycling_when_critic_endorses_direction(monkeypatch, tmp_path):
    """T8: no-gain alone is not a structural change — with the critic endorsing
    (high mean score, nothing discarded) the director keeps cycling, no pivot."""
    _cyclic_llm(monkeypatch, _ENDORSE_SCRIPT)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )

    async def _go():
        return await d.run(
            "a research question", max_cycles=5, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())
    # no-gain climbs past the stagnation bar every cycle, but the critic scores
    # h1 at 0.9 → direction endorsed → no pivot; the loop only stops at max_cycles
    assert state["cycles"] == 5
    assert state["adaptations"] == 0
    assert not any(r.startswith("[stagnation]") for r in state["rejected"])


# Script: h1 is always proposed and the critic scores it 0.1 → DISCARD, so every
# cycle contributes no gain AND carries the critic's veto of the direction.
_VETO_SCRIPT = [
    {"text": '{"query": "q1"}', "tool_calls": None, "usage": None},
    {"text": '{"entities": ["e1"]}', "tool_calls": None, "usage": None},
    {
        "text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "prediction": "p1", "falsification": "f1", "conditions": {}}]}',
        "tool_calls": None,
        "usage": None,
    },
    {
        "text": '{"hypotheses": [{"statement": "h1", "score": 0.1}]}',
        "tool_calls": None,
        "usage": None,
    },
    {"text": "cycle report", "tool_calls": None, "usage": None},
]


@pytest.mark.timeout(180)
def test_director_pivots_when_critic_vetoes_direction(monkeypatch, tmp_path):
    """T8: stagnation + critic veto (score below the endorsement bar) → pivot."""
    _cyclic_llm(monkeypatch, _VETO_SCRIPT)
    d = ResearchDirector(
        cfg=_cfg(),
        plugins_dir=_write_search_plugin(tmp_path),
        run_dir=str(tmp_path / "runs"),
        n_critics=1,
    )

    async def _go():
        return await d.run(
            "a research question", max_cycles=10, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())
    # cycle 1: h1 score 0.1 → DISCARD (never verified) → no_gain=1
    # cycle 2: same → no_gain=2 → critic veto (all discarded) → pivot → stop
    assert state["cycles"] == 2
    assert state["adaptations"] == 1
    assert any(r.startswith("[stagnation]") for r in state["rejected"])


# ── T-janitor: stale async-job re-claim ───────────────────────────────────────


def test_janitor_flags_stale_jobs(monkeypatch, tmp_path):
    """T-janitor: jobs older than the threshold without a numeric log are flagged."""
    d = ResearchDirector(cfg=_cfg(), run_dir=str(tmp_path / "runs"))
    jobs = tmp_path / "runs" / "a-general-research-question" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    # started 1h ago, log has no number → stale
    (jobs / "job-stale.json").write_text(
        json.dumps({"job_id": "job-stale", "pid": 1, "started_at": time.time() - 3600}),
        encoding="utf-8",
    )
    (jobs / "job-stale.log").write_text("still computing, no number yet", encoding="utf-8")
    # started 1min ago, no number → not stale yet
    (jobs / "job-fresh.json").write_text(
        json.dumps({"job_id": "job-fresh", "pid": 2, "started_at": time.time() - 60}),
        encoding="utf-8",
    )
    (jobs / "job-fresh.log").write_text("no number either", encoding="utf-8")
    # started 2h ago but log has a number → completed, not stale
    (jobs / "job-done.json").write_text(
        json.dumps({"job_id": "job-done", "pid": 3, "started_at": time.time() - 7200}),
        encoding="utf-8",
    )
    (jobs / "job-done.log").write_text("computed 42.0", encoding="utf-8")

    state = _default_state("a general research question")
    state["cycles"] = 1
    # the stale job's hypothesis was recorded verified (no real value) → distrust
    state["results"] = [
        {
            "cycle": 1,
            "verifications": [
                {
                    "statement": "h1",
                    "supports": 1,
                    "refutes": 0,
                    "orthogonal": 0,
                    "status": "verified",
                    "job_id": "job-stale",
                }
            ],
        }
    ]
    d._janitor_scan("a general research question", state)

    log_path = tmp_path / "runs" / "a-general-research-question" / "logs" / "janitor.jsonl"
    assert log_path.exists()
    entries = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()]
    events = [e["event"] for e in entries]
    assert "[janitor] job job-stale stale" in events
    assert "job-fresh" not in events
    assert "job-done" not in events
    # stale + recorded-verified → explicit distrust warning for the next cycle
    distrust = [e for e in entries if e["event"] == "[janitor] stale job trusted"]
    assert len(distrust) == 1
    assert distrust[0]["job_id"] == "job-stale"
    assert distrust[0]["statement"] == "h1"
