from __future__ import annotations

import pytest

from drbrain.loop.frontier import (
    ArtifactRef,
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    DoneContract,
    EvidenceRef,
    ExperimentSpec,
    ResearchFrontier,
    ResearchObjective,
)


def test_experiment_spec_fails_closed_when_preregistration_is_incomplete() -> None:
    spec = ExperimentSpec(metric="accuracy")
    with pytest.raises(ValueError, match="baseline"):
        spec.validate_for_execution()


def test_domain_objects_round_trip_as_json_compatible_dicts():
    branch = BranchSpec(
        branch_id="b1",
        objective_id="o1",
        hypothesis="A causes B",
        status=BranchStatus.SCREENED,
        experiment=ExperimentSpec(
            experiment_id="e1", branch_id="b1", metric="accuracy", seeds=[1, 2]
        ),
    )
    outcome = BranchOutcome(
        branch_id="b1",
        evidence=[EvidenceRef(evidence_id="ev1", source="paper")],
        artifacts=[ArtifactRef(artifact_id="a1", uri="file:///tmp/a")],
        proposed_children=[BranchSpec(branch_id="b2", hypothesis="A does not cause B")],
    )

    restored = BranchSpec.from_dict(branch.to_dict())
    restored_outcome = BranchOutcome.from_dict(outcome.to_dict())

    assert restored.status is BranchStatus.SCREENED
    assert restored.experiment is not None
    assert restored.experiment.seeds == [1, 2]
    assert restored_outcome.evidence[0].evidence_id == "ev1"
    assert restored_outcome.proposed_children[0].branch_id == "b2"
    assert ResearchObjective(question="q").to_dict()["question"] == "q"
    assert DoneContract().to_dict()["require_counter_evidence_search"] is True


def test_best_first_selection_is_deterministic_and_respects_active_limit():
    frontier = ResearchFrontier(max_active=1)
    frontier.add(BranchSpec(branch_id="low", hypothesis="low", novelty=0.1))
    frontier.add(BranchSpec(branch_id="high", hypothesis="high", novelty=0.9))

    selected = frontier.select_next()
    assert [branch.branch_id for branch in selected] == ["high"]
    assert frontier.branches["high"].status is BranchStatus.RUNNING
    assert frontier.pending()[0].branch_id == "low"


def test_progressive_widening_limits_children_and_applies_outcome():
    frontier = ResearchFrontier(initial_width=1, progressive_widening_alpha=0.5)
    root = frontier.add(BranchSpec(branch_id="root", hypothesis="root"))
    assert frontier.width_for(root.branch_id) == 1
    frontier.select_next()

    outcome = BranchOutcome(
        branch_id="root",
        status=BranchStatus.RETAINED,
        proposed_children=[
            BranchSpec(branch_id="c1", hypothesis="child 1"),
            BranchSpec(branch_id="c2", hypothesis="child 2"),
            BranchSpec(branch_id="c3", hypothesis="child 3"),
        ],
    )
    admitted = frontier.apply_outcome(outcome)
    assert [child.branch_id for child in admitted] == ["c1", "c2"]
    assert frontier.branches["root"].status is BranchStatus.RETAINED
    assert frontier.expansions["root"] == 1


def test_conflicting_replay_and_unknown_outcome_fail_closed():
    frontier = ResearchFrontier()
    frontier.add(BranchSpec(branch_id="b", hypothesis="same"))
    frontier.add(BranchSpec(branch_id="b", hypothesis="same"))
    try:
        frontier.add(BranchSpec(branch_id="b", hypothesis="different"))
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("conflicting replay was accepted")

    try:
        frontier.apply_outcome(BranchOutcome(branch_id="missing"))
    except KeyError:
        pass
    else:
        raise AssertionError("unknown branch outcome was accepted")
