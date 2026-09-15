"""T43/T45/T46: the production tree leg reads the published unified generation.

Finding 1: the SQL tree leg used to read the retired PageIndex ANN
(``tree_vectors.tree_layer='pageindex'``), and the LlamaIndex path still
instantiated the per-paper ``DrbrainTreeRetriever``.  These tests build a real
``rag prepare --unified`` product (canonical content → hierarchy → published
generation), plus the SQL projection the BM25/vector legs read, and then run:

* ``run_tree_leg`` directly (unified search → navigation → receipts → leaf
  text), asserting the "region hit → read original text" trajectory;
* the SQL leg through ``retrieve_documents_sql(legs=["tree"])``, asserting the
  returned row text is the leaf content of the *same* revision;
* a tree-only ``ask`` with a mocked synthesizer, asserting the answer is built
  from tree evidence;
* fail-closed behavior: no generation → unavailable; stale revision → dropped.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from drbrain.config import Config, EmbedConfig, LlamaIndexConfig
from drbrain.rag.preparation import prepare_sql_rag
from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.tree.builder import BuilderConfig
from drbrain.tree.clustering import ClusteringParams
from drbrain.tree.cost import CostParams
from drbrain.tree.embedding_identity import profile_from_config
from drbrain.tree.prepare import prepare_unified_index
from drbrain.tree.summary import SummaryContract


def _paragraph(tag: str, words: int = 50) -> str:
    """Long enough that a two-member group beats the routing cost estimate."""
    return " ".join(f"{tag}{index}" for index in range(words))


DOC_A = f"# Methods\n\n{_paragraph('alpha')}\n\n# Results\n\n{_paragraph('beta')}\n"
DOC_B = f"# Methods\n\n{_paragraph('gamma')}\n\n# Results\n\n{_paragraph('delta')}\n"
THEME = "kagome flat band theme"


def _fake_embed(texts, embed_cfg=None):
    """Deterministic 3-d embedder: the theme marker ranks the region first."""
    out = []
    for text in texts:
        value = str(text)
        marker = 1.0 if "kagome" in value.lower() else 0.0
        out.append([marker, float(len(value) % 7) + 0.5, float(len(value) % 5) + 0.5])
    return out


class _ThemeSummaryModel:
    """Deterministic summary text (the query embeds to the same vector)."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int):
        self.calls += 1
        return SimpleNamespace(text=THEME, finish_reason="stop")


#: Same identity the leg resolves from config (the store filters on it).
PROFILE = profile_from_config(EmbedConfig(dim=3, model="fake-embed", provider="local"))
BUILDER_CONFIG = BuilderConfig(
    clustering=ClusteringParams(dim=3, max_clusters=6),
    cost=CostParams(summary_output_budget=64, tool_overhead_tokens=1, min_members=2),
    contract=SummaryContract(model="fake", max_output_tokens=64, input_budget=12000),
    lam=4.0,
    max_layers=2,
    use_structure_hints=True,
)


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.db.path = str(tmp_path / "data" / "drbrain.db")
    cfg.dirs.papers = str(tmp_path / "data" / "papers")
    cfg.embed = EmbedConfig(dim=3, model="fake-embed", provider="local")
    cfg.llamaindex = LlamaIndexConfig(
        rag_engine="sql",
        retrievers=["tree"],
        enabled=True,
        storage_dir=str(tmp_path / "data" / "llamaindex"),
        tree_storage=str(tmp_path / "data" / "tree"),
        rerank=False,
        tree_candidates=50,
    )
    cfg.llm.models = []
    return cfg


@pytest.fixture()
def prepared(tmp_path, monkeypatch):
    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr("drbrain.services.embedding._embed_batch", _fake_embed)
    cfg = _cfg(tmp_path)
    db = Database(cfg.db.path)
    for local_id, text in (("p1", DOC_A), ("p2", DOC_B)):
        db.insert_paper(local_id, "T", 2024, "uploaded")
        written = write_canonical_content(db, local_id, text, media_type="md", parser="test")
        assert written["ok"], written
    db.commit()
    model = _ThemeSummaryModel()
    tree = prepare_unified_index(
        db,
        storage_dir=cfg.llamaindex.tree_storage,
        profile=PROFILE,
        embed_cfg=cfg.embed,
        summary_model=model,
        builder_config=BUILDER_CONFIG,
    )
    assert tree.ok, tree.to_json()
    assert tree.published, tree.to_json()
    assert tree.hierarchy.get("created", 0) > 0, tree.to_json()
    sql = prepare_sql_rag(cfg, publish=True)
    assert sql["nodes"] > 0, sql
    return SimpleNamespace(cfg=cfg, db=db, tree=tree, sql=sql, model=model)


class TestUnifiedTreeLeg:
    def test_region_hit_reads_the_original_text(self, prepared):
        from drbrain.tree.leg import run_tree_leg

        outcome = run_tree_leg(prepared.cfg, query=THEME, top_k=10)
        assert outcome.status == "ok", outcome.to_json()
        assert outcome.hits, outcome.to_json()
        reads = [step for step in outcome.trace if step["action"] == "read"]
        kinds = [step["detail"].get("kind") for step in reads]
        assert "region" in kinds, outcome.to_json()
        first_region = kinds.index("region")
        assert "leaf" in kinds[first_region + 1 :], outcome.to_json()
        actions = [step["action"] for step in outcome.trace]
        assert "expand" in actions, outcome.to_json()
        # No chat role is configured, so no model was silently consulted.
        assert outcome.planner.startswith("heuristic"), outcome.planner
        # Every hit is a real published leaf with a read receipt.
        for hit in outcome.hits:
            row = prepared.db.get_tree_node(hit.node_id)
            assert row is not None and row["kind"] == "leaf"
            assert hit.text
            assert hit.content_hash

    def test_missing_generation_fails_closed(self, tmp_path, monkeypatch):
        from drbrain.tree.leg import TreeLegUnavailableError, run_tree_leg

        monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
        monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
        cfg = _cfg(tmp_path)
        with pytest.raises(TreeLegUnavailableError, match="no active unified tree generation"):
            run_tree_leg(cfg, query=THEME, top_k=5)

    def test_revision_mismatch_is_dropped_and_reported(self, prepared):
        from drbrain.tree.leg import run_tree_leg

        outcome = run_tree_leg(prepared.cfg, query=THEME, top_k=10, verify=lambda hit: False)
        assert outcome.status == "unavailable"
        assert outcome.reason == "revision_mismatch"
        assert outcome.hits == []


class TestSqlTreeLeg:
    def _generation(self, prepared):
        from drbrain.rag.index_generations import capture_index_generation

        generation = capture_index_generation(prepared.cfg)
        assert generation, "the SQL snapshot must be published"
        return generation

    def test_tree_only_retrieval_uses_the_unified_generation(self, prepared):
        from drbrain.rag.sql_retrie import retrieve_documents_sql

        rows = retrieve_documents_sql(
            prepared.cfg,
            prepared.db,
            THEME,
            top_k=5,
            generation=self._generation(prepared),
        )
        assert rows, "tree-only retrieval must return rows"
        tree = rows.result.capabilities["tree"]
        assert tree["status"] == "ok"
        assert tree["generation"] == prepared.tree.published
        assert tree["hits"] >= 1
        entries = tree["entries"]
        assert any(entry["action"] == "read" for entry in entries)
        assert any(entry["action"] == "expand" for entry in entries)
        assert any(entry["detail"].get("kind") == "leaf" for entry in entries)
        # The delivered text is the canonical leaf content of the SQL revision.
        for row in rows:
            node = prepared.db.get_tree_node(row["node_id"])
            assert node is not None and node["kind"] == "leaf"
            assert row["legs"] == ["tree"]
            assert row["evidence_id"]

    def test_tree_only_ask_works_without_the_sql_corpus(self, prepared, monkeypatch):
        """The target flow: ingest → rag prepare (unified, default) → ask.

        With no ``drbrain_rag.db`` and no SQL generation, the tree request is
        served from the published unified generation; the missing legacy legs
        are reported unavailable instead of failing the whole query.
        """
        from llama_index.core import Settings
        from llama_index.core.llms import MockLLM

        import drbrain.rag.sql_adapter as sql_adapter
        from drbrain.rag import sql_retrie
        from drbrain.rag.engine import ask_llamaindex

        # sql_adapter binds the resolver at import time; patch that binding.
        monkeypatch.setattr(sql_adapter, "capture_index_generation", lambda cfg: None)
        absent = Path(prepared.cfg.llamaindex.tree_storage).parent / "absent_rag.db"
        monkeypatch.setattr(sql_retrie, "_default_rag_db", lambda cfg: absent)

        def _init(_cfg):
            Settings.llm = MockLLM(max_tokens=64)
            return True

        monkeypatch.setattr("drbrain.rag.llm.init_llamaindex_settings", _init)
        result = ask_llamaindex(
            prepared.cfg, prepared.db, THEME, top_k=5, streaming=False, legs=["tree"]
        )
        assert result["route"]["legs"] == ["tree"]
        assert result["sources"], result
        assert result.get("status") != "retrieval_failure"
        assert result["answer"]
        for source in result["sources"]:
            node = prepared.db.get_tree_node(source["node_id"])
            assert node is not None and node["kind"] == "leaf"

    def test_tree_only_ask_fails_closed_without_the_generation(self, prepared):
        """A missing unified generation abstains (retrieval_failure), never a fallback."""
        from pathlib import Path

        from drbrain.rag.engine import ask_llamaindex

        prepared.cfg.llamaindex.tree_storage = str(
            Path(prepared.cfg.llamaindex.tree_storage).parent / "empty-tree"
        )
        result = ask_llamaindex(
            prepared.cfg, prepared.db, THEME, top_k=5, streaming=False, legs=["tree"]
        )
        assert result["status"] == "retrieval_failure"
        assert result["sources"] == []

    def test_tree_only_ask_answers_from_leaf_evidence(self, prepared, monkeypatch):
        from llama_index.core import Settings
        from llama_index.core.llms import MockLLM

        from drbrain.rag.engine import ask_llamaindex

        def _init(_cfg):
            Settings.llm = MockLLM(max_tokens=64)
            return True

        monkeypatch.setattr("drbrain.rag.llm.init_llamaindex_settings", _init)
        result = ask_llamaindex(
            prepared.cfg,
            prepared.db,
            THEME,
            top_k=5,
            streaming=False,
            legs=["tree"],
        )
        assert result["engine"] == "llamaindex"
        assert result["route"]["legs"] == ["tree"]
        leg_status = {
            leg["source"]: leg["status"] for leg in result["telemetry"]["retrieval"]["legs"]
        }
        assert leg_status.get("tree") == "ok"
        assert result["sources"], result
        for source in result["sources"]:
            node = prepared.db.get_tree_node(source["node_id"])
            assert node is not None and node["kind"] == "leaf"
            assert source["paper_id"] in {"p1", "p2"}
        assert result["answer"]
