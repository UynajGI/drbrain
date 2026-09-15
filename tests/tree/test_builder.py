"""T34/T35: one bounded builder over the shared node model."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.affinity import SourceProfile
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.builder import BuilderConfig, BuilderError, TreeBuilder
from drbrain.tree.clustering import ClusteringParams, FittedStage
from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id
from drbrain.tree.cost import CostParams
from drbrain.tree.summary import SummaryContract, SummaryResponse


def _count(text: str) -> int:
    return max(1, len(text.split()))


def _doc(db: Database, local_id: str, text: str):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=1, media_type="md", parser="test"
    )
    db.upsert_document_revision(
        local_id,
        1,
        source_hash=f"src-{local_id}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        backend="test",
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


def _leaf(block, local_id: str, layer: int = 0) -> NodeRecord:
    ref = LeafRef(
        local_id=local_id,
        revision=1,
        block_id=block.block_id,
        char_start=0,
        char_end=len(block.text),
    )
    return NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="ready",
        layer=layer,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


class FakeModel:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int) -> SummaryResponse:
        self.calls += 1
        # Keep the summary well below the members' read cost.
        return SummaryResponse(f"summary of {self.calls} members")


def _make_docs(db: Database, *, papers: int = 3, sections: int = 4) -> list[str]:
    """Create papers with several substantial sections; returns leaf ids."""
    leaves: list[str] = []
    for paper in range(papers):
        text = "".join(
            f"# Section {section}\n\n" + " ".join([f"paper{paper}section{section}"] * 30) + "\n\n"
            for section in range(sections)
        )
        blocks = _doc(db, f"p{paper}", text)
        for block in blocks:
            if block.kind != "paragraph":
                continue
            leaf = _leaf(block, f"p{paper}")
            db.insert_tree_node(leaf, publish=True)
            leaves.append(leaf.node_id)
    return leaves


def _builder(db: Database, **overrides) -> TreeBuilder:
    config = BuilderConfig(
        clustering=overrides.pop("clustering", ClusteringParams(dim=4, max_clusters=6)),
        cost=overrides.pop("cost", CostParams(summary_output_budget=16, tool_overhead_tokens=1)),
        contract=overrides.pop(
            "contract",
            SummaryContract(model="fake", max_output_tokens=16, input_budget=4000),
        ),
        lam=overrides.pop("lam", 4.0),
        max_layers=overrides.pop("max_layers", 3),
        use_structure_hints=overrides.pop("use_structure_hints", True),
        **overrides,
    )
    builder = TreeBuilder(
        db,
        config=config,
        count_tokens=_count,
        embed=lambda texts: [
            [float(len(text) % 7), float(len(text) % 5), 0.5, 0.25] for text in texts
        ],
    )
    builder.model = FakeModel()
    builder._model_impl = builder.model
    builder.profile_id = "emb-test"
    return builder


class TestSingleRound:
    def test_round_creates_region_nodes_over_leaves(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db)
        builder = _builder(db)
        result = builder.build(leaves)
        assert result.rounds, "at least one round must run"
        first = result.rounds[0]
        assert first.frontier_size == len(leaves)
        assert first.created_nodes, "expected accepted region nodes"
        # Every model call is either an accepted parent or an explicitly
        # recorded rejection (a summarised group whose post-check failed).
        rejected_summaries = sum(
            metrics.rejected.get("summary_rejected", 0) for metrics in result.rounds
        )
        assert builder.model.calls == sum(m.accepted for m in result.rounds) + rejected_summaries
        for node_id in first.created_nodes:
            row = db.get_tree_node(node_id)
            assert row["kind"] == "region" and row["state"] == "ready"
            assert row["layer"] >= 1
            children = db.get_tree_children(node_id)
            assert len(children) >= 2
            assert all(child["state"] == "ready" for child in children)

    def test_structure_and_semantic_groups_share_one_table(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=2, sections=3)
        builder = _builder(db)
        result = builder.build(leaves)
        origins = {db.get_tree_node(node_id)["origin"] for node_id in result.created_nodes}
        assert origins, "regions must record where they came from"
        assert origins <= {"structure", "semantic", "mixed", "manual"}

    def test_leaves_are_never_re_embedded(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db)
        calls: list[list[str]] = []

        def embed(texts):
            calls.append(list(texts))
            return [[float(len(text) % 7), 0.5, 0.25, 0.125] for text in texts]

        builder = _builder(db)
        builder.embed = embed
        result = builder.build(leaves)
        # First call embeds the frontier leaves; any later call must only carry
        # the new summary texts, not the leaves again.
        assert calls, "the frontier must be embedded at least once"
        assert set(calls[0]) and len(calls[0]) == len(leaves)
        leaf_texts = {builder._node_text(node_id) for node_id in leaves}
        for later in calls[1:]:
            assert not (set(later) & leaf_texts)
        assert result.created_nodes


def _global_fixture() -> tuple[FittedStage, dict[str, SourceProfile]]:
    """One global stage where the prior moves a row across the threshold."""
    fitted = FittedStage(
        stage="global",
        row_ids=("nl-a", "nl-b", "nl-c", "nl-d"),
        component_ids=("g0", "g1"),
        probs=((0.09, 0.91), (0.9, 0.0), (0.0, 0.5), (0.0, 0.5)),
        labels=((1,), (0,), (1,), (1,)),
        n_components=2,
        threshold=0.1,
    )
    profiles = {
        "nl-a": SourceProfile(parts=(("p0", ("Methods",), 100),)),
        "nl-b": SourceProfile(parts=(("p0", ("Methods",), 100),)),
        "nl-c": SourceProfile(parts=(("p1", ("Methods",), 100),)),
        "nl-d": SourceProfile(parts=(("p1", ("Methods",), 100),)),
    }
    return fitted, profiles


class TestTwoLevelConditioning:
    """T04/T28/T30: the structural prior enters at both stages, not just local."""

    def test_corrected_global_membership_seeds_the_local_subsets(self, tmp_path):
        """A row the prior moves globally must move with it into the local split.

        ``local_stages`` selects each local fit's rows from the global labels;
        feeding it the raw upstream labels would leave the prior inert at the
        global level.
        """
        db = Database(tmp_path / "db.sqlite")
        builder = _builder(db)
        builder.config = dataclasses.replace(builder.config, lam=6.0)
        fitted, profiles = _global_fixture()
        conditioned = builder._conditioned_global(fitted, profiles)
        assert fitted.labels == ((1,), (0,), (1,), (1,))
        # same-document evidence pushes nl-a past the 0.1 threshold into g0
        # alone (at lam=4 it would still hold both components)
        assert conditioned.labels == ((0,), (0,), (1,), (1,))
        assert conditioned.probs == fitted.probs  # raw posterior is untouched

    def test_lambda_zero_keeps_the_upstream_labels(self, tmp_path):
        """The documented ablation must hand the raw stage straight through."""
        db = Database(tmp_path / "db.sqlite")
        builder = _builder(db)
        fitted, profiles = _global_fixture()
        builder.config = dataclasses.replace(builder.config, lam=0.0)
        assert builder._conditioned_global(fitted, profiles) is fitted


class TestBoundsAndReachability:
    def test_build_stops_without_infinite_growth(self, tmp_path):
        db = Database(tmp_path / "sqlite" if False else tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=2, sections=7)
        builder = _builder(db, max_layers=4, min_frontier=2)
        result = builder.build(leaves)
        assert result.stop_reason
        assert len(result.rounds) <= 4
        for metrics in result.rounds:
            assert metrics.created_nodes or metrics.stop_reason

    def test_multi_root_result_keeps_every_leaf_reachable(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=3, sections=2)
        builder = _builder(db, max_layers=2)
        result = builder.build(leaves)
        # Every leaf is either a root itself or has a parent chain to a root.
        roots = set(result.roots)
        for node_id in leaves:
            assert node_id in roots or db.get_tree_parents(node_id), node_id
        # Unpromoted leaves are legitimate multi-roots, not lost evidence.
        assert set(db.leaves_missing_parent()) <= roots

    def test_no_parents_stops_cleanly(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=2, sections=3)
        builder = _builder(db, max_new_nodes_per_round=0)
        result = builder.build(leaves)
        assert result.stop_reason in {"no_parent", "budget_exhausted"}
        assert result.created_nodes == []

    def test_duplicate_groups_do_not_repeat_work(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db)
        builder = _builder(db)
        result = builder.build(leaves)
        keys = set()
        for node_id in result.created_nodes:
            children = db.get_tree_children(node_id)
            key = frozenset(child["child_id"] for child in children)
            assert key not in keys, "the same member set must not be published twice"
            keys.add(key)

    def test_empty_seed_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        builder = _builder(db)
        with pytest.raises(BuilderError):
            builder.build([])


class TestFailureStates:
    def test_missing_model_is_reported(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=1, sections=3)
        builder = _builder(db)
        del builder._model_impl
        builder.model = None
        with pytest.raises(BuilderError, match="index model"):
            builder.build(leaves)

    def test_summary_failures_leave_members_in_place(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=2, sections=4)
        builder = _builder(db)

        class Failing:
            def complete(self, prompt, *, max_tokens):
                return SummaryResponse("", finish_reason="length")

        builder._model_impl = Failing()
        result = builder.build(leaves)
        assert result.created_nodes == []
        for node_id in leaves:
            assert db.get_tree_node(node_id)["state"] == "ready"

    def test_metrics_are_recorded_per_round(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db)
        builder = _builder(db)
        result = builder.build(leaves)
        payload = result.to_json()
        assert payload["rounds"]
        first = payload["rounds"][0]
        assert first["frontier_size"] == len(leaves)
        assert set(first) >= {
            "round",
            "frontier_size",
            "components",
            "proposals",
            "accepted",
            "rejected",
            "created_nodes",
            "prompt_tokens",
            "summary_tokens",
            "duration_ms",
            "stop_reason",
        }


class TestBoundedScheduling:
    """T59 scheduling: per-run frontier cap and in-flight summary calls."""

    def test_frontier_limit_bounds_one_run_and_leaves_the_rest_roots(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _make_docs(db, papers=2, sections=3)
        builder = _builder(db, frontier_limit=3)
        result = builder.build(leaves)
        assert result.frontier_total == len(leaves)
        assert result.frontier_processed == 3
        assert all(metrics.frontier_size <= 3 for metrics in result.rounds)
        covered: set[str] = set()
        for node_id in result.created_nodes:
            covered.update(child["child_id"] for child in db.get_tree_children(node_id))
        assert covered <= set(leaves[:3])
        # Unprocessed seeds stay roots and re-enter the next run's frontier.
        assert set(result.roots) >= set(leaves[3:])
        assert len(db.leaves_missing_parent()) > 0

    def test_summary_workers_keep_the_serial_result(self, tmp_path):
        def run(workers: int):
            db = Database(tmp_path / f"db-{workers}.sqlite")
            leaves = _make_docs(db)
            builder = _builder(db, summary_workers=workers)
            result = builder.build(leaves)
            return (
                [list(metrics.created_nodes) for metrics in result.rounds],
                builder.model.calls,
                result.stop_reason,
            )

        assert run(1) == run(3)
