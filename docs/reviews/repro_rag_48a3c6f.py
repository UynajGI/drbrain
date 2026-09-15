"""Independent review probes; synthetic SQLite/Zvec data, no user corpus or network.

Run explicitly with pytest, DRBRAIN_REVIEW_PROJECT and PYTHONPATH set.
The two missing-reader probes record the old checkout's prerequisite; exclude
them on e5b7a09, where three-leg retrieval is fixed. See the accompanying report.
The repro_ name keeps intentionally failing review probes out of default discovery.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

PROJECT = Path(os.environ["DRBRAIN_REVIEW_PROJECT"])
spec = importlib.util.spec_from_file_location(
    "review_cli_helpers", PROJECT / "tests/test_cli_index.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture(scope="module")
def indexed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rag-review-synthetic")
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("DRBRAIN_ROOT", raising=False)
        patch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
        configure(root)
        helpers._write_corpus(root)
        helpers._patch_build(patch)
        built = helpers._invoke(root, "index", "build", "--json")
        assert built.exit_code == 0, (built.stdout, built.stderr)
        assert json.loads(built.stdout)["ok"] is True
        assert not (root / "data/drbrain_rag.db").exists()
        yield root


def configure(root, *, backend="sqlite", retrievers=None, rerank=False):
    helpers._write_config(root)
    p = root / "config.yaml"
    value = yaml.safe_load(p.read_text())
    value["llamaindex"]["retrievers"] = retrievers or ["bm25", "vector", "tree"]
    value["llamaindex"]["rerank"] = rerank
    value["retrieval"] = {"vector_backend": backend}
    p.write_text(yaml.safe_dump(value), encoding="utf-8")


def tree_config(root, *, rerank=False):
    from drbrain.config import Config, EmbedConfig, LlamaIndexConfig

    cfg = Config()
    cfg.db.path = str(root / "data/drbrain.db")
    cfg.dirs.papers = str(root / "data/papers")
    cfg.embed = EmbedConfig(dim=3, provider="local", model="fake-embed")
    cfg.llm.models = []
    cfg.llamaindex = LlamaIndexConfig(
        enabled=True,
        rag_engine="sql",
        storage_dir=str(root / "data/llamaindex"),
        tree_storage=str(root / "data/tree"),
        retrievers=["tree"],
        rerank=rerank,
    )
    return cfg


@pytest.mark.parametrize("backend", ["sqlite", "zvec"])
def test_successful_build_makes_all_three_cli_routes_available(indexed_root, backend):
    configure(indexed_root, backend=backend)
    found = helpers._invoke(indexed_root, "search", "p1body0", "--json")
    payload = json.loads(found.stdout)
    legs = {leg["source"]: leg["status"] for leg in payload["legs"]}
    assert all(legs.get(name) in {"ok", "empty"} for name in ("bm25", "vector", "tree")), {
        "build": "ok",
        "backend": backend,
        "search_exit": found.exit_code,
        "legs": legs,
        "error": payload.get("error"),
    }


def test_readiness_cannot_certify_missing_bm25_vector_readers(indexed_root):
    configure(indexed_root)
    found = helpers._invoke(indexed_root, "search", "p1body0", "--json")
    legs = json.loads(found.stdout)["legs"]
    assert any(
        leg["source"] in {"bm25", "vector"} and leg["status"] == "unavailable" for leg in legs
    )
    status = helpers._invoke(indexed_root, "index", "status", "--json")
    report = json.loads(status.stdout)
    assert report["states"]["retrievable"]["ready"] is False, {
        "status": report["status"],
        "route": report["route"],
        "legs": legs,
    }


def test_verify_cannot_pass_when_requested_readers_are_missing(indexed_root):
    configure(indexed_root)
    verified = helpers._invoke(indexed_root, "index", "verify", "--json")
    report = json.loads(verified.stdout)
    assert report["ok"] is False and verified.exit_code == 1, {
        "verify_exit": verified.exit_code,
        "ok": report["ok"],
        "errors": report["errors"],
    }


def test_unified_query_honors_enabled_reranking(indexed_root, monkeypatch):
    from drbrain.rag.sql_retrie import retrieve_documents_sql
    from drbrain.storage.database import Database

    calls = []

    class Reranker:
        def rerank(self, query, texts):
            calls.append((query, list(texts)))
            return [float(i) for i in range(len(texts))]

    monkeypatch.setattr("drbrain.rag.sql_retrie._get_reranker", lambda cfg: Reranker())
    cfg = tree_config(indexed_root, rerank=True)
    db = Database(cfg.db.path)
    try:
        rows = retrieve_documents_sql(cfg, db, "p1body0", top_k=5)
    finally:
        db.close()
    assert rows, rows.result.to_json() if hasattr(rows.result, "to_json") else str(rows.result)
    assert calls, "rerank=true produced evidence without invoking the configured reranker"


def test_partial_navigation_is_not_reported_as_complete(indexed_root):
    from drbrain.tree.leg import run_tree_leg
    from drbrain.tree.navigator import HeuristicPlanner
    from drbrain.tree.tools import ToolBudget

    outcome = run_tree_leg(
        tree_config(indexed_root),
        query="p1body0",
        top_k=10,
        view="leaf",
        planner=HeuristicPlanner(),
        budget=ToolBudget(max_calls=1),
        storage_dir=indexed_root / "data/tree",
    )
    assert outcome.navigation_status == "partial", outcome.to_json()
    assert outcome.hits, outcome.to_json()
    assert outcome.status != "ok", outcome.to_json()


def test_paper_scope_is_enforced_on_navigation_reads(indexed_root):
    from drbrain.storage.database import Database
    from drbrain.tree.leg import run_tree_leg
    from drbrain.tree.navigator import ScriptedPlanner

    db = Database(indexed_root / "data/drbrain.db")
    try:
        forbidden = db.conn.execute(
            "SELECT node_id FROM tree_nodes WHERE kind='leaf' AND local_id='p2' LIMIT 1"
        ).fetchone()[0]
    finally:
        db.close()
    planner = ScriptedPlanner(
        [
            {"action": "read", "node_id": forbidden},
            {"action": "finish", "reason": "done"},
        ]
    )
    outcome = run_tree_leg(
        tree_config(indexed_root),
        query="p1body0",
        top_k=10,
        local_ids=["p1"],
        planner=planner,
        storage_dir=indexed_root / "data/tree",
    )
    assert all(hit.local_id == "p1" for hit in outcome.hits), {
        "scope": ["p1"],
        "read_papers": [hit.local_id for hit in outcome.hits],
        "status": outcome.status,
    }


def prepare_helpers():
    spec = importlib.util.spec_from_file_location(
        "review_prepare_helpers", PROJECT / "tests/tree/test_prepare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["lambda", "force"])
def test_rebuild_revisits_assigned_leaves(tmp_path, change):
    import dataclasses

    from drbrain.tree.prepare import prepare_unified_index

    h = prepare_helpers()
    db = h._setup(tmp_path)
    try:
        embed, model = h._FakeEmbedder(), h._FakeSummaryModel()
        first = h._prepare(db, tmp_path, embed=embed, model=model)
        assert first.ok and first.hierarchy["created"] > 0, first.to_json()
        assert not db.leaves_missing_parent(), "fixture must already have assigned every leaf"
        config = (
            dataclasses.replace(h.BUILDER_CONFIG, lam=0.0)
            if change == "lambda"
            else h.BUILDER_CONFIG
        )
        second = prepare_unified_index(
            db,
            storage_dir=tmp_path / "tree",
            profile=h.PROFILE,
            embed=embed,
            summary_model=model,
            builder_config=config,
            force=change == "force",
        )
        assert second.hierarchy.get("rounds"), {
            "change": change,
            "first": first.hierarchy,
            "second": second.hierarchy,
        }
    finally:
        db.close()


def test_publication_failure_can_resume_without_forcing_rebuild(tmp_path, monkeypatch):
    import drbrain.tree.prepare as prepare

    h = prepare_helpers()
    db = h._setup(tmp_path)
    actual_publish = prepare.publish_tree_generation

    def unavailable(*args, **kwargs):
        raise OSError("synthetic publication interruption")

    try:
        embed, model = h._FakeEmbedder(), h._FakeSummaryModel()
        monkeypatch.setattr(prepare, "publish_tree_generation", unavailable)
        first = h._prepare(db, tmp_path, embed=embed, model=model)
        assert first.publication["status"] == "failed", first.to_json()
        monkeypatch.setattr(prepare, "publish_tree_generation", actual_publish)
        second = h._prepare(db, tmp_path, embed=embed, model=model)
        assert second.published, second.to_json()
    finally:
        db.close()


def test_plain_markdown_ingest_does_not_require_global_llm_models(tmp_path, monkeypatch):
    """CPU-only canonical ingestion should not require a global chat fallback list."""
    configure(tmp_path)
    monkeypatch.setenv("DRBRAIN_OFFLINE", "1")
    source = tmp_path / "material.md"
    source.write_text(
        "# Generic methods note\n\n" + "This method has a specified boundary condition. " * 40,
        encoding="utf-8",
    )
    result = helpers._invoke(tmp_path, "ingest", str(source), "--json")
    report = json.loads(result.stdout)
    assert report.get("failed") == 0 and report.get("successful") == 1, report
