"""T04/T05 fixture tests: affinity matrices, reweighting, cost gates.

The JSON fixtures under ``tests/tree/fixtures`` are hand-computed; the module
under test is the frozen reference implementation, not the builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drbrain.tree.affinity import SourceSpan, affinity_matrix, build_pictures, structural_affinity
from drbrain.tree.cost import (
    CostParams,
    estimate_multi_target_cost,
    post_check,
    pre_screen,
    unique_coverage,
)
from drbrain.tree.posteriors import PosteriorStage

FIXTURES = Path(__file__).parent / "fixtures"


def _spans(rows) -> dict[str, SourceSpan]:
    return {
        row["id"]: SourceSpan(
            local_id=row["local_id"],
            revision=row["revision"],
            block_id=row["block_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            tokens=row["tokens"],
            heading_path=tuple(row["heading_path"]),
        )
        for row in rows
    }


def _stage(rows, components, probs, lam=0.0, affinity=None) -> PosteriorStage:
    return PosteriorStage(
        stage="global",
        row_ids=tuple(row["id"] for row in rows),
        component_ids=tuple(components),
        probs=tuple(tuple(row) for row in probs),
        lam=lam,
        affinity=affinity,
    )


class TestAffinityFixtures:
    def load(self):
        return json.loads((FIXTURES / "affinity_cases.json").read_text(encoding="utf-8"))

    def test_hand_checked_affinity_matrices(self):
        for case in self.load()["affinity_cases"]:
            spans = _spans(case["rows"])
            stage = _stage(case["rows"], case["components"], case["probs"])
            matrix = affinity_matrix(stage, spans)
            for row_index, row in enumerate(case["rows"]):
                expected = case["expected"][row["id"]]
                for col, want in enumerate(expected):
                    assert matrix[row_index][col] == pytest.approx(want, rel=1e-9, abs=1e-12), (
                        case["name"],
                        row["id"],
                        col,
                    )

    def test_self_exclusion_changes_picture(self):
        """A single-row picture with the row excluded is empty -> neutral."""
        case = self.load()["affinity_cases"][0]
        spans = _spans(case["rows"])
        stage = _stage(case["rows"], case["components"], case["probs"])
        picture = build_pictures(stage, spans, exclude_row="a1")["g0"]
        assert picture.parts
        assert all(
            not (local_id == "p1" and path == ("Methods",) and weight == pytest.approx(80.0))
            for local_id, path, weight in picture.parts
        )
        with_self = build_pictures(stage, spans)["g0"]
        assert with_self.doc_mass.get("p1", 0.0) > picture.doc_mass.get("p1", 0.0)
        assert structural_affinity(spans["a1"], picture) < structural_affinity(
            spans["a1"], with_self
        )

    def test_lambda_zero_is_identity_and_reweighting_is_checkable(self):
        cases = self.load()["reweight_cases"]
        for case in cases:
            stage = PosteriorStage(
                stage="global",
                row_ids=tuple(f"r{i}" for i in range(len(case["probs"]))),
                component_ids=("g0", "g1"),
                probs=tuple(tuple(row) for row in case["probs"]),
                lam=case["lambda"],
                affinity=tuple(tuple(row) for row in case["affinity"]),
            )
            reweighted = stage.reweighted()
            if case["lambda"] == 0.0:
                assert reweighted is stage  # exact no-op object identity
            for row_idx, expected_row in enumerate(case["expected"]):
                for col, want in enumerate(expected_row):
                    assert reweighted.probs[row_idx][col] == pytest.approx(want, rel=1e-9)
            if "expected_labels" in case:
                assert reweighted.labels()["r0"] == tuple(case["expected_labels"][0])

    def test_lambda_range_enforced(self):
        with pytest.raises(ValueError, match="lambda"):
            PosteriorStage(
                stage="global",
                row_ids=("r0",),
                component_ids=("g0",),
                probs=((1.0,),),
                lam=99.0,
                affinity=((0.0,),),
            )

    def test_unassigned_rows_are_carried(self):
        stage = PosteriorStage(
            stage="global",
            row_ids=("r0", "r1"),
            component_ids=("g0", "g1"),
            probs=((0.05, 0.05), (0.9, 0.05)),
        )
        members = stage.membership()
        assert members["r0"] == ()
        assert members["r1"][0][0] == "g0"
        assert "r0" in stage.labels()  # explicit, never dropped


class TestCostFixtures:
    def load(self):
        return json.loads((FIXTURES / "cost_cases.json").read_text(encoding="utf-8"))

    def test_coverage_cases(self):
        data = self.load()
        for case in data["coverage_cases"]:
            spans = _spans([{**row, "id": f"s{i}"} for i, row in enumerate(case["spans"])])
            coverage = unique_coverage(spans.values())
            assert coverage.unique_tokens == case["expected_tokens"], case["name"]
            assert coverage.unique_chars == case["expected_chars"], case["name"]

    def test_pre_screen_cases(self):
        data = self.load()
        for case in data["pre_screen_cases"]:
            spans = _spans([{**row, "id": f"s{i}"} for i, row in enumerate(case["spans"])])
            coverage = unique_coverage(spans.values())
            decision = pre_screen(
                member_ids=[f"s{i}" for i in range(case["member_count"])],
                coverage=coverage,
                member_tokens=case["member_tokens"],
                seen_member_keys=set(),
                member_key="k",
            )
            assert decision.accepted == (case["expected_reason"] == ""), case["name"]
            assert decision.reason == case["expected_reason"], case["name"]

    def test_pre_screen_duplicate_group(self):
        spans = _spans(
            [
                {
                    "id": "a",
                    "local_id": "p1",
                    "revision": 1,
                    "block_id": "b1",
                    "char_start": 0,
                    "char_end": 100,
                    "tokens": 800,
                    "heading_path": [],
                },
                {
                    "id": "b",
                    "local_id": "p1",
                    "revision": 1,
                    "block_id": "b2",
                    "char_start": 0,
                    "char_end": 100,
                    "tokens": 800,
                    "heading_path": [],
                },
            ]
        )
        decision = pre_screen(
            member_ids=["a", "b"],
            coverage=unique_coverage(spans.values()),
            member_tokens=1600,
            seen_member_keys={"k"},
            member_key="k",
        )
        assert decision.reason == "duplicate_group"

    def test_post_check_cases(self):
        data = self.load()
        base_spans = [
            {
                "id": "s0",
                "local_id": "p1",
                "revision": 1,
                "block_id": "b1",
                "char_start": 0,
                "char_end": 100,
                "tokens": 800,
                "heading_path": [],
            },
            {
                "id": "s1",
                "local_id": "p1",
                "revision": 1,
                "block_id": "b2",
                "char_start": 0,
                "char_end": 100,
                "tokens": 800,
                "heading_path": [],
            },
        ]
        spans = _spans(base_spans)
        coverage = unique_coverage(spans.values())
        for case in data["post_check_cases"]:
            referenced = list(spans.values()) if case["referenced"] == "all" else [spans["s0"]]
            decision = post_check(
                summary_text=case["summary"],
                summary_tokens=case["summary_tokens"],
                finish_reason=case["finish_reason"],
                coverage=coverage,
                referenced_spans=referenced,
                params=CostParams(summary_output_budget=512),
            )
            assert decision.accepted == (case["expected_reason"] == ""), case["name"]
            assert decision.reason == case["expected_reason"], case["name"]

    def test_multi_target_estimate(self):
        case = self.load()["multi_target_case"]
        spans = _spans([{**row, "id": f"s{i}"} for i, row in enumerate(case["spans"])])
        coverage = unique_coverage(spans.values())
        estimate = estimate_multi_target_cost(
            coverage,
            branches=case["branches"],
            summary_tokens=case["summary_tokens"],
            params=CostParams(),
        )
        assert estimate == case["expected"]

    def test_member_text_survives_rejection(self):
        """A rejected group is a decision, not a mutation of its members."""
        spans = _spans(
            [
                {
                    "id": "a",
                    "local_id": "p1",
                    "revision": 1,
                    "block_id": "b1",
                    "char_start": 0,
                    "char_end": 100,
                    "tokens": 900,
                    "heading_path": [],
                },
            ]
        )
        coverage = unique_coverage(spans.values())
        decision = pre_screen(
            member_ids=["a"],
            coverage=coverage,
            member_tokens=900,
            seen_member_keys=set(),
            member_key="k",
        )
        assert not decision.accepted
        assert list(spans) == ["a"] and coverage.spans  # untouched

    def test_read_cost_uses_unique_members_not_paths(self):
        """Soft multi-parent paths must not inflate read cost."""
        two = _spans(
            [
                {
                    "id": "a",
                    "local_id": "p1",
                    "revision": 1,
                    "block_id": "b1",
                    "char_start": 0,
                    "char_end": 100,
                    "tokens": 100,
                    "heading_path": [],
                },
                {
                    "id": "b",
                    "local_id": "p1",
                    "revision": 1,
                    "block_id": "b1",
                    "char_start": 0,
                    "char_end": 100,
                    "tokens": 100,
                    "heading_path": [],
                },
            ]
        )
        assert unique_coverage(two.values()).unique_tokens == 100
