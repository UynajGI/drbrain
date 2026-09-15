"""T29/T30: source profiles from storage and structure-conditioned assignment."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.affinity import SourceProfile
from drbrain.tree.assign import (
    AssignmentError,
    assignment_coverage,
    node_source_profile,
    row_affinity,
    soft_assignment,
    stage_affinities,
)
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id
from drbrain.tree.posteriors import PosteriorStage


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


def _leaf(block, local_id: str = "p1") -> NodeRecord:
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
        layer=0,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


def _region(children, *, layer=1, summary="s", contract=None) -> NodeRecord:
    contract = contract or {"prompt": "p"}
    refs = tuple(
        ChildRef(child_id=child.node_id, child_revision=1, ordinal=i)
        for i, child in enumerate(children)
    )
    return NodeRecord(
        node_id=region_node_id(refs, contract),
        revision=1,
        kind="region",
        state="ready",
        layer=layer,
        content_hash=hashlib.sha256(summary.encode()).hexdigest(),
        summary=summary,
        children=refs,
        contract=contract,
    )


DOC_A = "# Methods\n\nsparse retrieval uses bm25 over tokens.\n\n# Results\n\nwe improve recall by twelve points.\n"
DOC_B = "# Methods\n\nirrigation scheduling for maize yields.\n"


class TestSourceProfiles:
    def test_leaf_profile_uses_block_range_and_heading(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", DOC_A)
        leaf = _leaf(blocks[0])
        db.insert_tree_node(leaf, publish=True)
        profile = node_source_profile(db, leaf.node_id, count_tokens=_count)
        assert profile.documents() == ("p1",)
        local_id, path, tokens = profile.parts[0]
        assert path == ("Methods",) and tokens > 0

    def test_region_profile_aggregates_unique_leaves(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks_a = _doc(db, "p1", DOC_A)
        blocks_b = _doc(db, "p2", DOC_B)
        leaf_a = _leaf(blocks_a[0], "p1")
        leaf_b = _leaf(blocks_b[0], "p2")
        for leaf in (leaf_a, leaf_b):
            db.insert_tree_node(leaf, publish=True)
        region = _region([leaf_a, leaf_b])
        db.insert_tree_node(region, publish=True)
        # A second parent over the same leaves must not double-count origins.
        other = _region([leaf_a, leaf_b], summary="other", contract={"prompt": "q"})
        db.insert_tree_node(other, publish=True)
        upper = _region([region], layer=2, summary="upper", contract={"prompt": "u"})
        db.insert_tree_node(upper, publish=True)
        profile = node_source_profile(db, upper.node_id, count_tokens=_count)
        assert set(profile.documents()) == {"p1", "p2"}
        merged = profile.merged()
        assert len(merged.parts) == 2  # one part per (document, path)
        leaf_total = sum(
            part[2]
            for part in node_source_profile(db, leaf_a.node_id, count_tokens=_count).merged().parts
        ) + sum(
            part[2]
            for part in node_source_profile(db, leaf_b.node_id, count_tokens=_count).merged().parts
        )
        assert merged.token_total() == pytest.approx(leaf_total)

    def test_unknown_node_raises(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        with pytest.raises(AssignmentError, match="unknown tree node"):
            node_source_profile(db, "nl-missing")


class TestAffinityWiring:
    def _stage(self):
        return PosteriorStage(
            stage="global",
            row_ids=("r1", "r2", "r3"),
            component_ids=("g0", "g1"),
            probs=((0.8, 0.2), (0.7, 0.3), (0.4, 0.6)),
        )

    def test_stage_affinity_matrix_matches_hand_values(self):
        stage = self._stage()
        profiles = {
            "r1": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r2": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r3": SourceProfile(parts=(("p2", ("Methods",), 100),)),
        }
        matrix = stage_affinities(stage, profiles)
        # picture g0 without r1 holds r2 (0.7*100, p1) and r3 (0.4*100, p2)
        assert matrix[0][0] == pytest.approx(70.0 / 110.0)
        # picture g1 without r1 holds r2 (0.3*100, p1) and r3 (0.6*100, p2)
        assert matrix[0][1] == pytest.approx(30.0 / 90.0)
        assert matrix[2][0] == pytest.approx(0.0)  # cross-document neutral

    def test_row_affinity_excludes_self(self):
        stage = self._stage()
        profiles = {
            "r1": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r2": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r3": SourceProfile(parts=(("p1", ("Methods",), 100),)),
        }
        values = row_affinity(profiles["r1"], stage, profiles, "r1")
        assert values[0] > 0
        solo = {"r1": profiles["r1"]}
        assert row_affinity(profiles["r1"], stage, solo, "r1") == (0.0, 0.0)


class TestSoftAssignment:
    def _profiles(self):
        return {
            "r1": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r2": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r3": SourceProfile(parts=(("p2", ("Methods",), 100),)),
        }

    def _stage(self):
        return PosteriorStage(
            stage="global",
            row_ids=("r1", "r2", "r3"),
            component_ids=("g0", "g1"),
            probs=((0.6, 0.4), (0.55, 0.45), (0.45, 0.55)),
        )

    def test_lambda_zero_reproduces_stage_membership(self):
        stage = self._stage()
        candidates = soft_assignment(stage, self._profiles(), lam=0.0)
        expected = stage.labels()
        for candidate in candidates:
            assert set(candidate.member_ids()) == {
                row_id for row_id in stage.row_ids if candidate.component_id in expected[row_id]
            }

    def test_structural_boost_prefers_same_document_group(self):
        stage = self._stage()
        candidates = soft_assignment(stage, self._profiles(), lam=6.0)
        by_id = {candidate.component_id: candidate for candidate in candidates}
        g0_ids = set(by_id["g0"].member_ids())
        assert {"r1", "r2"} <= g0_ids  # same-document rows stay together
        coverage = assignment_coverage(candidates)
        assert coverage["assigned_rows"] >= 2
        assert coverage["multi_parent_rows"] >= 1  # soft membership is preserved

    def test_lambda_crosses_the_membership_threshold(self):
        """The prior must be able to move a row, not only keep rows together.

        ``r1`` starts in ``g1`` alone (0.09 stays under the strict 0.1 on the
        g0 side); with same-document evidence for g0 and a cross-document
        picture for g1 the reweighted posterior puts it into ``g0`` alone.  An
        assertion on already-member rows cannot observe this — the lambda term
        would be inert and still pass.
        """
        stage = PosteriorStage(
            stage="global",
            row_ids=("r1", "r2", "r3"),
            component_ids=("g0", "g1"),
            probs=((0.09, 0.91), (0.9, 0.0), (0.0, 0.5)),
        )
        profiles = {
            "r1": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r2": SourceProfile(parts=(("p1", ("Methods",), 100),)),
            "r3": SourceProfile(parts=(("p2", ("Methods",), 100),)),
        }
        raw = {
            candidate.component_id: set(candidate.member_ids())
            for candidate in soft_assignment(stage, profiles, lam=0.0)
        }
        assert "r1" in raw["g1"] and "r1" not in raw["g0"]

        corrected = {
            candidate.component_id: dict(candidate.members)
            for candidate in soft_assignment(stage, profiles, lam=6.0)
        }
        assert "r1" not in corrected["g1"]
        assert corrected["g0"]["r1"] == pytest.approx(0.9756, abs=5e-4)
        assert "r3" in corrected["g1"]  # cross-document picture is unchanged

    def test_empty_assignment_is_carried(self):
        stage = PosteriorStage(
            stage="global",
            row_ids=("r1", "r2"),
            component_ids=("g0", "g1"),
            probs=((0.05, 0.05), (0.9, 0.05)),
        )
        profiles = {
            "r1": SourceProfile(parts=(("p1", ("A",), 10),)),
            "r2": SourceProfile(parts=(("p1", ("A",), 10),)),
        }
        candidates = soft_assignment(stage, profiles, lam=0.0)
        assigned = {node for candidate in candidates for node in candidate.member_ids()}
        assert "r1" not in assigned
        # Coverage helper keeps the unassigned row visible through the stage.
        assert candidates  # components still exist

    def test_lambda_bounds_enforced(self):
        with pytest.raises(ValueError, match="lambda"):
            soft_assignment(self._stage(), self._profiles(), lam=99.0)

    def test_affinity_can_be_disabled_for_ablation(self):
        stage = self._stage()
        raw = soft_assignment(stage, {}, lam=6.0, with_affinity=False)
        # every row keeps at least one component above the 0.1 threshold
        assert assignment_coverage(raw)["assigned_rows"] == 3
