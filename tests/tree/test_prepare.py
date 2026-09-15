"""T45: one incremental prepare for FTS, shared vectors and the hierarchy."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import pytest
import typer

from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.storage.node_projection import read_node_text
from drbrain.tree.builder import BuilderConfig
from drbrain.tree.clustering import ClusteringParams
from drbrain.tree.cost import CostParams
from drbrain.tree.embedding_identity import EmbeddingProfile
from drbrain.tree.prepare import HIERARCHY_WATERMARK, prepare_unified_index
from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
from drbrain.tree.summary import SummaryContract

PROFILE = EmbeddingProfile(provider="local", model="qwen3-embedding-0.6b", dimension=3)

BUILDER_CONFIG = BuilderConfig(
    clustering=ClusteringParams(dim=3, max_clusters=6),
    cost=CostParams(summary_output_budget=64, tool_overhead_tokens=1, min_members=2),
    contract=SummaryContract(model="fake", max_output_tokens=64, input_budget=12000),
    lam=4.0,
    max_layers=2,
    use_structure_hints=True,
)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        texts = list(texts)
        self.calls += 1
        self.texts += len(texts)
        return [[float(len(text) % 7), float(len(text) % 5), 0.5] for text in texts]


class _FakeSummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int):
        self.calls += 1
        return SimpleNamespace(text="summary", finish_reason="stop")


def _doc(db: Database, local_id: str, sections: int = 4, words: int = 30) -> None:
    text = "".join(
        f"# Section {index}\n\n" + " ".join([f"{local_id}body{index}"] * words) + "\n\n"
        for index in range(sections)
    )
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _setup(tmp_path) -> Database:
    db = Database(tmp_path / "db.sqlite")
    _doc(db, "p1")
    _doc(db, "p2")
    return db


def _prepare(db, tmp_path, *, profile=PROFILE, embed=None, model=None, **kwargs):
    return prepare_unified_index(
        db,
        storage_dir=tmp_path / "tree",
        profile=profile,
        embed=embed,
        summary_model=model,
        builder_config=BUILDER_CONFIG,
        **kwargs,
    )


def _role_config(model: str) -> SimpleNamespace:
    """A real role config: ``llm.roles.index_model`` -> one named endpoint."""
    return SimpleNamespace(
        llm={
            "endpoints": {
                "idx": {
                    "provider": "openai",
                    "model": model,
                    "base_url": "http://127.0.0.1:9/v1",
                }
            },
            "roles": {"index_model": "idx"},
        }
    )


class _StubIndex:
    """Duck-typed ``IndexModel``: real resolved role, deterministic output."""

    def __init__(self, role) -> None:
        self.role = role
        self.calls = 0

    def call_text(self, prompt: str, max_tokens: int = 0) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(text=f"summary-{self.role.model}-{self.calls}", finish_reason="stop")


class TestPrepare:
    def test_first_prepare_fills_stages_and_publishes(self, tmp_path):
        db = _setup(tmp_path)
        embedder, model = _FakeEmbedder(), _FakeSummaryModel()
        outcome = _prepare(db, tmp_path, embed=embedder, model=model)
        assert outcome.ok, outcome.to_json()
        assert outcome.fts["status"] == "ok"
        nodes = outcome.vectors["nodes"]
        assert nodes > 0 and outcome.vectors["embedded"] == nodes
        assert outcome.hierarchy["created"] > 0 and model.calls > 0
        assert outcome.changed and outcome.published
        assert get_active_tree_generation(tmp_path / "tree") == outcome.published
        resolved = resolve_tree_generation(tmp_path / "tree", outcome.published)
        # The generation carries one vector per ready node (leaves + the
        # regions the hierarchy stage created and embedded).
        ready_nodes = db.conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE state = 'ready'"
        ).fetchone()[0]
        assert resolved["manifest"]["vector_count"] == ready_nodes > nodes

    def test_no_text_is_embedded_twice_by_the_pipeline(self, tmp_path):
        """T34/T45: the hierarchy reuses the vectors the stages already stored.

        The builder must read back the vector the vector stage stored for a
        leaf, and the region vector the previous round embedded, instead of
        recomputing the same text.  Counting from the prepare entry is the
        point: a builder-only count cannot see the duplicate.  The fixture
        keeps every node's text unique (headings carry the paper id) so a text
        identifies exactly one node.
        """
        db = Database(tmp_path / "db.sqlite")

        def _unique_doc(local_id: str, sections: int = 4) -> None:
            text = "".join(
                f"# {local_id} Section {index}\n\n"
                + " ".join([f"{local_id}body{index}"] * 30)
                + "\n\n"
                for index in range(sections)
            )
            if db.get_paper(local_id) is None:
                db.insert_paper(local_id, "T", 2024, "uploaded")
                db.commit()
            result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
            assert result["ok"]

        _unique_doc("p1")
        _unique_doc("p2")

        class _Recording:
            def __init__(self) -> None:
                self.seen: list[str] = []

            def __call__(self, texts):
                texts = list(texts)
                self.seen.extend(texts)
                return [[float(len(text) % 7), float(len(text) % 5), 0.5] for text in texts]

        class _UniqueSummaryModel:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str, *, max_tokens: int):
                self.calls += 1
                # One distinct text per call, so a text identifies one node.
                return SimpleNamespace(text=f"summary-{self.calls}", finish_reason="stop")

        embedder, model = _Recording(), _UniqueSummaryModel()
        outcome = _prepare(db, tmp_path, embed=embedder, model=model)
        assert outcome.ok, outcome.to_json()
        assert outcome.hierarchy["created"] > 0 and model.calls > 0

        counts = Counter(embedder.seen)
        leaf_texts = {
            str(read_node_text(db.conn, str(node_id)))
            for (node_id,) in db.conn.execute(
                "SELECT node_id FROM tree_nodes WHERE kind = 'leaf' AND state = 'ready'"
            ).fetchall()
        }
        assert leaf_texts
        assert all(counts[text] == 1 for text in leaf_texts)
        repeated = {text: count for text, count in counts.items() if count > 1}
        assert not repeated, f"texts embedded more than once: {sorted(repeated)[:2]}"

    def test_second_prepare_does_zero_model_and_embedding_work(self, tmp_path):
        db = _setup(tmp_path)
        embedder, model = _FakeEmbedder(), _FakeSummaryModel()
        first = _prepare(db, tmp_path, embed=embedder, model=model)
        assert first.ok and first.published
        embedder.calls = embedder.texts = 0
        model.calls = 0
        second = _prepare(db, tmp_path, embed=embedder, model=model)
        assert second.ok, second.to_json()
        assert embedder.calls == 0 and model.calls == 0
        assert second.vectors["embedded"] == 0
        assert second.hierarchy["status"] == "complete"
        assert not second.changed and second.published is None
        assert second.publication["status"] == "skipped"
        # The active pointer still names the first generation.
        assert get_active_tree_generation(tmp_path / "tree") == first.published

    def test_model_change_rebuilds_only_vectors(self, tmp_path):
        db = _setup(tmp_path)
        embedder, model = _FakeEmbedder(), _FakeSummaryModel()
        assert _prepare(db, tmp_path, embed=embedder, model=model).ok
        model.calls = 0
        embedder.calls = embedder.texts = 0
        other = dataclasses.replace(PROFILE, model="qwen3-embedding-other")
        outcome = _prepare(db, tmp_path, profile=other, embed=embedder, model=model)
        assert outcome.ok, outcome.to_json()
        assert outcome.vectors["embedded"] == outcome.vectors["nodes"] > 0
        assert embedder.calls > 0
        assert outcome.fts["status"] == "ok"  # FTS untouched
        assert outcome.hierarchy.get("created", 0) == 0  # hierarchy untouched
        assert model.calls == 0

    def test_failed_vector_stage_leaves_the_store_resumable(self, tmp_path):
        db = _setup(tmp_path)
        model = _FakeSummaryModel()

        class _Broken:
            def __call__(self, texts):
                raise RuntimeError("gpu down")

        failed = _prepare(db, tmp_path, embed=_Broken(), model=model)
        assert not failed.ok and failed.vectors["status"] == "failed"
        assert failed.published is None and failed.publication["status"] == "skipped"
        assert get_active_tree_generation(tmp_path / "tree") is None
        assert model.calls == 0

        embedder = _FakeEmbedder()
        resumed = _prepare(db, tmp_path, embed=embedder, model=model)
        assert resumed.ok, resumed.to_json()
        assert resumed.vectors["embedded"] == resumed.vectors["nodes"]
        assert resumed.published

    def test_model_failure_is_not_complete_and_the_same_command_recovers(self, tmp_path):
        """T36/T42: a broken endpoint is a state, not a finished stage.

        The failing run must not advance the hierarchy watermark nor publish,
        and it must stop instead of spending one failing call per candidate.
        The identical command then retries with the recovered model and builds
        the parents the failure had skipped.
        """
        db = _setup(tmp_path)

        class _Faulty:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str, *, max_tokens: int):
                self.calls += 1
                raise RuntimeError("index endpoint down")

        faulty = _Faulty()
        first = _prepare(db, tmp_path, embed=_FakeEmbedder(), model=faulty)
        assert not first.ok
        assert first.failed_stages == ["hierarchy"]
        assert first.hierarchy["status"] == "failed"
        assert first.hierarchy["created"] == 0
        assert first.hierarchy["reason"] == "summary_failed"
        assert first.hierarchy["failed"] == {"model_error: RuntimeError": 1}
        assert faulty.calls == 1  # fail fast, not one call per candidate
        assert first.published is None
        assert db.get_vector_metadata(HIERARCHY_WATERMARK) is None
        assert db.leaves_missing_parent()  # every leaf still awaits a parent

        recovered = _FakeSummaryModel()
        second = _prepare(db, tmp_path, embed=_FakeEmbedder(), model=recovered)
        assert second.ok, second.to_json()
        assert recovered.calls > 0
        assert second.hierarchy["created"] > 0
        assert second.published
        assert db.get_vector_metadata(HIERARCHY_WATERMARK) is not None

    def test_model_identity_change_invalidates_and_rebuilds(self, tmp_path):
        """T26/T36: a new index deployment must not reuse the old summaries.

        The identity is bound from the resolved role, so the same member sets
        against a second endpoint are different work: the cached summaries are
        not reused, the parents the first deployment built are retired instead
        of served, and the new model is actually called.
        """
        from drbrain.services.model_roles import ROLE_INDEX, resolve_model_role
        from drbrain.tree.prepare import IndexModelSummary

        db = _setup(tmp_path)
        embedder = _FakeEmbedder()

        def _model(name: str):
            role = resolve_model_role(_role_config(name), ROLE_INDEX)
            stub = _StubIndex(role)
            return IndexModelSummary(stub), stub

        first_model, first_stub = _model("index-a")
        first = _prepare(db, tmp_path, embed=embedder, model=first_model)
        assert first.ok, first.to_json()
        assert first_stub.calls > 0
        first_regions = [
            node_id
            for round_payload in first.hierarchy["rounds"]
            for node_id in round_payload["created_nodes"]
        ]
        assert first_regions

        second_model, second_stub = _model("index-b")
        second = _prepare(db, tmp_path, embed=embedder, model=second_model)
        assert second.ok, second.to_json()
        assert second.hierarchy.get("reason") != "signature-unchanged"
        assert second_stub.calls > 0  # cached summaries are not reused
        assert second.hierarchy["created"] > 0

        placeholders = ",".join("?" for _ in first_regions)
        states = {
            str(node_id): str(state)
            for node_id, state in db.conn.execute(
                f"SELECT node_id, state FROM tree_nodes WHERE node_id IN ({placeholders})",
                tuple(first_regions),
            ).fetchall()
        }
        assert set(states.values()) == {"stale"}  # retired, not served
        vector_states = {
            str(state)
            for (state,) in db.conn.execute(
                f"SELECT state FROM node_vectors WHERE node_id IN ({placeholders})",
                tuple(first_regions),
            ).fetchall()
        }
        assert vector_states == {"failed"}  # never ready again

    def test_partial_run_does_not_certify_the_next_contract(self, tmp_path):
        """T36: an unset watermark must not keep the old contract alive.

        A run that fails part way leaves the watermark untouched, so the next
        deployment starts from ``previous is None`` while regions of the old
        contract are already on disk.  Those have to be retired, otherwise the
        rebuild publishes a tree that mixes two contracts.
        """
        from drbrain.services.model_roles import ROLE_INDEX, resolve_model_role
        from drbrain.tree.prepare import IndexModelSummary

        class _FailingIndex(_StubIndex):
            def __init__(self, role, ok_calls: int) -> None:
                super().__init__(role)
                self.ok_calls = ok_calls

            def call_text(self, prompt: str, max_tokens: int = 0) -> SimpleNamespace:
                if self.calls >= self.ok_calls:
                    self.calls += 1
                    raise RuntimeError("index endpoint down")
                return super().call_text(prompt, max_tokens=max_tokens)

        db = _setup(tmp_path)
        embedder = _FakeEmbedder()
        failing = _FailingIndex(resolve_model_role(_role_config("index-a"), ROLE_INDEX), 1)
        first = _prepare(db, tmp_path, embed=embedder, model=IndexModelSummary(failing))
        assert not first.ok
        assert first.hierarchy["created"] > 0  # partial: some parents were built
        assert db.get_vector_metadata(HIERARCHY_WATERMARK) is None
        old_regions = [
            node_id
            for round_payload in first.hierarchy["rounds"]
            for node_id in round_payload["created_nodes"]
        ]
        assert old_regions

        healthy = _StubIndex(resolve_model_role(_role_config("index-b"), ROLE_INDEX))
        second = _prepare(db, tmp_path, embed=embedder, model=IndexModelSummary(healthy))
        assert second.ok, second.to_json()
        assert healthy.calls > 0

        placeholders = ",".join("?" for _ in old_regions)
        states = {
            str(state)
            for (state,) in db.conn.execute(
                f"SELECT state FROM tree_nodes WHERE node_id IN ({placeholders})",
                tuple(old_regions),
            ).fetchall()
        }
        assert states == {"stale"}, f"old-contract regions must not stay ready: {states}"
        ready_contracts = {
            str(contract)
            for (contract,) in db.conn.execute(
                "SELECT DISTINCT contract_json FROM tree_nodes "
                "WHERE state = 'ready' AND kind = 'region'"
            ).fetchall()
        }
        assert len(ready_contracts) == 1  # one contract in the published tree

    def test_fts_drift_is_repaired_by_the_stage(self, tmp_path):
        db = _setup(tmp_path)
        db.conn.execute("INSERT INTO content_fts(content_fts) VALUES('delete-all')")
        db.commit()
        assert db.content_fts_status()["consistent"] is False
        outcome = _prepare(db, tmp_path, embed=_FakeEmbedder(), model=_FakeSummaryModel())
        assert outcome.ok, outcome.to_json()
        assert outcome.fts["status"] == "rebuilt"
        assert db.content_fts_status()["consistent"] is True


class TestCli:
    def _ctx(self, tmp_path):
        ctx = mock.MagicMock(spec=typer.Context)
        ctx.obj = {
            "config": SimpleNamespace(
                db={"path": str(tmp_path / "db.sqlite")},
                dirs={},
                embed={"provider": "local", "model": "qwen3-embedding-0.6b", "dim": 3},
            )
        }
        return ctx

    def test_unified_flag_wires_the_incremental_prepare(self, tmp_path):
        from drbrain.cli import rag_commands

        outcome = SimpleNamespace(ok=True, to_json=lambda: {"ok": True, "changed": True})
        captured = {}

        def _fake_prepare(db, **kwargs):
            captured.update(kwargs)
            return outcome

        db = mock.MagicMock()
        with (
            mock.patch.object(rag_commands, "open_db", mock.MagicMock(return_value=db)),
            mock.patch("drbrain.tree.prepare.prepare_unified_index", side_effect=_fake_prepare),
            mock.patch("typer.echo") as echo,
        ):
            rag_commands.rag_prepare_cmd(
                self._ctx(tmp_path),
                force=False,
                paper=None,
                unified=True,
                tree_storage=str(tmp_path / "tree"),
                json_output=True,
            )
        assert captured["storage_dir"] == tmp_path / "tree"
        assert captured["profile"].model == "qwen3-embedding-0.6b"
        assert json.loads(echo.call_args[0][0])["ok"] is True

    def test_unified_rejects_paper_restriction(self, tmp_path):
        from drbrain.cli import rag_commands

        with pytest.raises(typer.BadParameter):
            rag_commands.rag_prepare_cmd(
                self._ctx(tmp_path),
                force=False,
                paper=["p1"],
                unified=True,
                tree_storage="",
                json_output=False,
            )
