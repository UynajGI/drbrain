"""End-to-end autoresearch integration tests (real db + real model).

These exercise the three layers together — RAG graph tools + external plugins
(local KG search, sciverse/arxiv/s2) + the 12-node loop — against the real
``data/drbrain.db`` and the primary model (OpenCode Zen ``deepseek-v4-flash``).
They are marked ``integration`` (excluded from the fast ``-m "not integration"``
suite) because they hit the network and a 16 GB database.

Run explicitly:
    uv run pytest tests/integration -m integration -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "drbrain.db"

pytestmark = pytest.mark.integration

_requires_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="data/drbrain.db not present"
)


@_requires_db
def test_local_kg_plugin_searches_real_db(monkeypatch):
    """The local KG plugin reads the real schema and returns flat-band results."""
    monkeypatch.setenv("DRBRAIN_DB", str(DB_PATH))
    from drbrain.plugins.registry import PluginRegistry

    reg = PluginRegistry()
    reg.discover(str(ROOT / "research" / "plugins"))

    papers = reg.call("search_papers", {"query": "topological flat band", "limit": 5})
    assert papers.status == "ok"
    assert papers.data and papers.data.get("count", 0) >= 1

    concepts = reg.call("search_concept_nodes", {"query": "kagome", "limit": 10})
    assert concepts.status == "ok"
    assert concepts.data and concepts.data.get("count", 0) >= 1


@_requires_db
def test_full_loop_flatband_end_to_end(monkeypatch):
    """The whole loop runs the flat-band research and returns a real report.

    This is the autoresearch acceptance test: with no human in the loop, the
    pipeline must progress from literature search → feature extraction →
    hypothesis proposal → critique → verification → a written report, and the
    report must carry the machine-readable summary line (candidates>0).
    """
    import asyncio

    monkeypatch.setenv("DRBRAIN_DB", str(DB_PATH))

    from drbrain.config import load_config
    from drbrain.loop import ResearchLoopWorkflow

    cfg = load_config(str(ROOT / "config.yaml"), str(ROOT / "config.local.yaml"))
    # db=None: data access goes through the read-only local-KG plugin
    # (DRBRAIN_DB above); the workflow's own `db` is only used for settle/claims
    # writes, which we deliberately keep out of the production database here.
    wf = ResearchLoopWorkflow(
        cfg=cfg,
        db=None,
        plugins_dir=str(ROOT / "research" / "plugins"),
    )

    async def _go() -> str:
        handler = wf.run(task="topological flat band")
        return await handler

    report = asyncio.run(_go())

    assert report
    # machine-readable summary line precedes the markdown report
    assert "candidates=" in report
    assert "candidates=0" not in report.split("\n")[0]
