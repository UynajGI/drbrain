"""Production regressions from the unified RAG design review (synthetic corpus)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from drbrain.storage.database import Database


def _helpers(name):
    spec = importlib.util.spec_from_file_location(
        f"production_{name.replace('/', '_')}", Path(__file__).parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _helpers("test_cli_index")


def _configure(root, *, retrievers=None, backend="zvec", rerank=False):
    cli._write_config(root, retrievers=retrievers)
    path = root / "config.yaml"
    value = yaml.safe_load(path.read_text())
    value["retrieval"] = {"vector_backend": backend}
    value["llamaindex"]["rerank"] = rerank
    path.write_text(yaml.safe_dump(value))


def _config(root, *, retrievers=None, rerank=False):
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
        retrievers=retrievers or ["tree"],
        rerank=rerank,
    )
    return cfg


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("rag-production")
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("DRBRAIN_ROOT", raising=False)
        patch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
        _configure(root)
        cli._write_corpus(root)
        cli._patch_build(patch)
        built = cli._invoke(root, "index", "build", "--json")
        assert built.exit_code == 0, (built.stdout, built.stderr)
        assert json.loads(built.stdout)["ok"]
        assert not (root / "data/drbrain_rag.db").exists()
        yield root


@pytest.mark.parametrize("legs", [["bm25", "vector", "tree"], ["bm25", "vector"]])
def test_selected_unified_routes_are_readable_and_ready(indexed, legs):
    _configure(indexed, retrievers=legs)
    found = cli._invoke(indexed, "search", "p1body0", "--json")
    payload = json.loads(found.stdout)
    assert all(leg["status"] in {"ok", "empty", "partial"} for leg in payload["legs"])
    checked = cli._invoke(indexed, "index", "status", "--json")
    assert json.loads(checked.stdout)["states"]["retrievable"]["ready"]


def test_unified_reranker_is_called(indexed, monkeypatch):
    from drbrain.rag.sql_retrie import retrieve_documents_sql

    calls = []

    class Reranker:
        def rerank(self, query, texts):
            calls.append(list(texts))
            return [float(i) for i in range(len(texts))]

    monkeypatch.setattr("drbrain.rag.sql_retrie._get_reranker", lambda cfg: Reranker())
    db = Database(indexed / "data/drbrain.db")
    try:
        assert retrieve_documents_sql(_config(indexed, rerank=True), db, "p1body0")
        assert calls
    finally:
        db.close()


def test_tree_budget_exhaustion_stays_partial(indexed):
    from drbrain.tree.leg import run_tree_leg
    from drbrain.tree.navigator import HeuristicPlanner
    from drbrain.tree.tools import ToolBudget

    outcome = run_tree_leg(
        _config(indexed),
        query="p1body0",
        view="leaf",
        top_k=10,
        planner=HeuristicPlanner(),
        budget=ToolBudget(max_calls=1),
    )
    assert outcome.hits and outcome.navigation_status == "partial"
    assert outcome.status == "partial"


def test_navigation_cannot_read_outside_paper_scope(indexed):
    from drbrain.tree.leg import run_tree_leg
    from drbrain.tree.navigator import ScriptedPlanner

    db = Database(indexed / "data/drbrain.db")
    try:
        node = db.conn.execute(
            "SELECT node_id FROM tree_nodes WHERE kind='leaf' AND local_id='p2' LIMIT 1"
        ).fetchone()[0]
    finally:
        db.close()
    planner = ScriptedPlanner([{"action": "read", "node_id": node}, {"action": "finish"}])
    outcome = run_tree_leg(
        _config(indexed),
        query="p1body0",
        top_k=10,
        local_ids=["p1"],
        planner=planner,
    )
    assert all(hit.local_id == "p1" for hit in outcome.hits)
    assert not any(
        response.get("ok") for action, response in planner.observations if action.action == "read"
    )


def test_tree_fusion_preserves_read_receipts(indexed, monkeypatch):
    from drbrain.rag.sql_retrie import retrieve_documents_sql
    from drbrain.tree.navigator import ScriptedPlanner

    db = Database(indexed / "data/drbrain.db")
    try:
        node_id, text = db.conn.execute(
            "SELECT n.node_id, b.text FROM tree_nodes n JOIN content_blocks b "
            "ON b.block_id=n.block_id WHERE n.kind='leaf' AND LENGTH(b.text)>40 LIMIT 1"
        ).fetchone()
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": node_id, "char_start": 7, "char_end": 27},
                {"action": "finish"},
            ]
        )
        monkeypatch.setattr(
            "drbrain.tree.leg.resolve_navigation_planner", lambda cfg: (planner, "scripted")
        )
        rows = retrieve_documents_sql(_config(indexed), db, "p1body0")
        row = next(row for row in rows if row["node_id"] == node_id)
        assert (row["char_start"], row["char_end"], row["text"]) == (7, 27, text[7:27])
    finally:
        db.close()


def test_bm25_scope_precedes_limit(tmp_path):
    from drbrain.rag.sql_retrie import _unified_bm25_entries
    from drbrain.services.canonical_content import write_canonical_content

    db = Database(tmp_path / "scope.sqlite")
    try:
        for paper in ("inside", "outside"):
            db.insert_paper(paper, paper, 2024, "uploaded")
            write_canonical_content(
                db, paper, "scopedneedle has an identical passage.", media_type="md", parser="test"
            )
        hits = db.search_content('"scopedneedle"', limit=10)
        assert {hit["local_id"] for hit in hits} == {"inside", "outside"}
        scope = next(hit["local_id"] for hit in hits if hit["local_id"] != hits[0]["local_id"])
        assert _unified_bm25_entries(db, "scopedneedle", 1, {scope})
    finally:
        db.close()


def test_publication_switch_during_query_does_not_mix_generations(indexed, monkeypatch):
    import drbrain.rag.sql_retrie as retrieval
    from drbrain.tree.embedding_identity import profile_from_config
    from drbrain.tree.leg import active_tree_generation
    from drbrain.tree.publish import publish_tree_generation

    cfg = _config(indexed, retrievers=["bm25", "vector", "tree"])
    captured = active_tree_generation(cfg)
    original = retrieval._unified_bm25_entries
    db = Database(cfg.db.path)

    def switch(snapshot, *args, **kwargs):
        result = original(snapshot, *args, **kwargs)
        publish_tree_generation(
            db,
            indexed / "data/tree",
            profile_id=profile_from_config(cfg.embed).profile_id(),
            vector_dir=indexed / "data/tree/vectors",
        )
        return result

    monkeypatch.setattr(retrieval, "_unified_bm25_entries", switch)
    try:
        rows = retrieval.retrieve_documents_sql(cfg, db, "p1body0")
        assert rows and active_tree_generation(cfg) != captured
        assert rows.result.generation == captured
        assert rows.result.capabilities["tree"]["generation"] == captured
    finally:
        db.close()


def test_scoped_snapshot_guards_all_tools_and_outside_summaries(indexed):
    from drbrain.tree.leg import active_tree_generation
    from drbrain.tree.publish import resolve_tree_generation
    from drbrain.tree.reading import ReadOnlyTreeStore
    from drbrain.tree.tools import ToolError, ToolState, TreeTools

    generation = active_tree_generation(_config(indexed))
    snapshot = resolve_tree_generation(indexed / "data/tree", generation)["snapshot"]
    with ReadOnlyTreeStore(snapshot, local_ids=["p1"]) as store:
        outside = store.conn.execute(
            "SELECT node_id FROM tree_nodes WHERE kind='leaf' AND local_id='p2' LIMIT 1"
        ).fetchone()[0]
        tools = TreeTools(store)
        for action in (tools.read, tools.expand, tools.parents, tools.read_scope):
            with pytest.raises(ToolError):
                action(outside, ToolState())
        for row in store.conn.execute(
            "SELECT node_id FROM tree_nodes WHERE kind='region' AND state='ready'"
        ):
            node_id = row[0]
            origins = store.conn.execute(
                "WITH RECURSIVE m(id) AS (SELECT ? UNION SELECT c.child_id "
                "FROM tree_node_children c JOIN m ON c.parent_id=m.id) "
                "SELECT DISTINCT local_id FROM tree_nodes JOIN m ON node_id=m.id WHERE kind='leaf'",
                (node_id,),
            ).fetchall()
            if any(item[0] != "p1" for item in origins):
                assert store.get_tree_node(node_id) is None


def test_unpublished_content_cannot_enter_snapshot_results(indexed):
    from drbrain.rag.sql_retrie import retrieve_documents_sql
    from drbrain.services.canonical_content import write_canonical_content
    from drbrain.tree.leg import active_tree_generation
    from drbrain.tree.publish import resolve_tree_generation
    from drbrain.tree.reading import ReadOnlyTreeStore

    cfg = _config(indexed, retrievers=["bm25"])
    generation = active_tree_generation(cfg)
    db = Database(cfg.db.path)
    try:
        db.insert_paper("latepaper", "New", 2024, "uploaded")
        write_canonical_content(
            db,
            "latepaper",
            "postpublishuniqueterm added after publication.",
            media_type="md",
            parser="test",
        )
        resolved = resolve_tree_generation(indexed / "data/tree", generation)
        with ReadOnlyTreeStore(resolved["snapshot"]) as snap:
            assert not snap.conn.execute(
                "SELECT 1 FROM tree_nodes WHERE local_id='latepaper'"
            ).fetchone()
        rows = retrieve_documents_sql(cfg, db, "postpublishuniqueterm")
        assert not rows
        assert rows.result.generation == generation
    finally:
        db.close()


@pytest.mark.parametrize("change", ["lambda", "force"])
def test_completed_hierarchy_rebuilds_when_requested(tmp_path, change):
    from drbrain.tree.prepare import prepare_unified_index

    h = _helpers("tree/test_prepare")
    db = h._setup(tmp_path)
    try:
        embed, model = h._FakeEmbedder(), h._FakeSummaryModel()
        first = h._prepare(db, tmp_path, embed=embed, model=model)
        assert first.hierarchy.get("created") and not db.leaves_missing_parent()
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
        assert second.hierarchy.get("rounds"), second.to_json()
    finally:
        db.close()


def test_failed_publication_resumes_without_recomputing(tmp_path, monkeypatch):
    import drbrain.tree.prepare as prepare

    h = _helpers("tree/test_prepare")
    db = h._setup(tmp_path)
    actual_publish = prepare.publish_tree_generation

    def unavailable(*args, **kwargs):
        raise OSError("publication interrupted")

    try:
        embed, model = h._FakeEmbedder(), h._FakeSummaryModel()
        monkeypatch.setattr(prepare, "publish_tree_generation", unavailable)
        first = h._prepare(db, tmp_path, embed=embed, model=model)
        assert first.publication["status"] == "failed"
        calls = (embed.calls, model.calls)
        monkeypatch.setattr(prepare, "publish_tree_generation", actual_publish)
        second = h._prepare(db, tmp_path, embed=embed, model=model)
        assert second.published and second.ok
        assert (embed.calls, model.calls) == calls
    finally:
        db.close()


def test_offline_markdown_ingest_does_not_require_chat_models(tmp_path, monkeypatch):
    _configure(tmp_path)
    monkeypatch.setenv("DRBRAIN_OFFLINE", "1")
    source = tmp_path / "material.md"
    source.write_text("# Generic methods note\n\n" + "This method has a boundary condition. " * 40)
    result = cli._invoke(tmp_path, "ingest", str(source), "--json")
    report = json.loads(result.stdout)
    assert report["failed"] == 0 and report["successful"] == 1, report
    assert source.exists()
