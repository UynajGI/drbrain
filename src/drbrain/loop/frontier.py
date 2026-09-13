"""Research-frontier primitives for adaptive autoresearch orchestration.

The frontier is deliberately independent from the workflow runner.  A worker
can execute the existing loop for one :class:`BranchSpec`, while this module
keeps branch state, deterministic priority scoring, and progressive widening
in a small, serialisable data model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, TypeVar


class BranchStatus(StrEnum):
    """Lifecycle states accepted by :class:`ResearchFrontier`."""

    PROPOSED = "proposed"
    SCREENED = "screened"
    RUNNING = "running"
    VERIFYING = "verifying"
    RETAINED = "retained"
    PRUNED = "pruned"
    NEEDS_HUMAN = "needs_human"


T = TypeVar("T", bound="Serializable")


class Serializable:
    """Tiny stdlib-only JSON-compatible serialisation helper."""

    _nested: ClassVar[dict[str, type[Serializable]]] = {}

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, Serializable):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set, frozenset)):
                return [convert(v) for v in value]
            return value

        fields = getattr(self, "__dataclass_fields__", {})
        return {key: convert(getattr(self, key)) for key in fields if not key.startswith("_")}

    @classmethod
    def from_dict(cls: type[T], value: Mapping[str, Any] | None) -> T:
        raw = dict(value or {})
        for name, nested_type in cls._nested.items():
            item = raw.get(name)
            if isinstance(item, Mapping):
                raw[name] = nested_type.from_dict(item)
            elif isinstance(item, list):
                raw[name] = [
                    nested_type.from_dict(v) if isinstance(v, Mapping) else v for v in item
                ]
        return cls(**raw)


@dataclass
class ResearchObjective(Serializable):
    """A run-level question and its host-owned constraints."""

    objective_id: str = ""
    question: str = ""
    scope: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, int | float] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    termination_conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoneContract(Serializable):
    """Machine-checkable conditions under which an objective is complete."""

    min_supported_evidence: int = 1
    required_experiments: int = 0
    require_replication: bool = False
    require_counter_evidence_search: bool = True
    min_confidence: float = 0.0
    stop_conditions: list[str] = field(default_factory=list)
    human_review: bool = False


@dataclass
class ExperimentSpec(Serializable):
    """Immutable (by convention) pre-registration for one branch experiment."""

    experiment_id: str = ""
    branch_id: str = ""
    metric: str = ""
    baseline: str = ""
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    seeds: list[int] = field(default_factory=list)
    repetitions: int = 1
    stopping_rule: str = ""
    failure_criteria: list[str] = field(default_factory=list)
    falsification_criteria: list[str] = field(default_factory=list)
    budget: dict[str, int | float] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def validate_for_execution(self) -> None:
        """Fail closed before compute when the pre-registration is incomplete."""
        required = {
            "metric": self.metric,
            "baseline": self.baseline,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "stopping_rule": self.stopping_rule,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError("experiment preregistration missing: " + ", ".join(missing))
        if self.repetitions < 1:
            raise ValueError("experiment repetitions must be positive")
        if not self.seeds:
            raise ValueError("experiment preregistration requires at least one seed")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds):
            raise ValueError("experiment preregistration seeds must be integers")
        if len(self.seeds) < self.repetitions:
            raise ValueError(
                "experiment preregistration requires one deterministic seed per repetition"
            )
        if not self.falsification_criteria:
            raise ValueError("experiment preregistration requires falsification criteria")


@dataclass
class EvidenceRef(Serializable):
    """A stable pointer to a source used by a branch result."""

    evidence_id: str = ""
    source: str = ""
    locator: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    relation: str = "supports"


@dataclass
class ArtifactRef(Serializable):
    """A stable pointer to an experiment output or generated artifact."""

    artifact_id: str = ""
    uri: str = ""
    checksum: str = ""
    media_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchSpec(Serializable):
    """One hypothesis branch and its auditable priority inputs."""

    branch_id: str = ""
    parent_branch_id: str = ""
    objective_id: str = ""
    hypothesis: str = ""
    rationale: str = ""
    status: BranchStatus = BranchStatus.PROPOSED
    depth: int = 0
    information_gain: float = 0.0
    uncertainty_reduction: float = 0.0
    novelty: float = 0.0
    feasibility: float = 0.0
    cost: float = 0.0
    duplication: float = 0.0
    validity_risk: float = 0.0
    priority_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    budget: dict[str, int | float] = field(default_factory=dict)
    experiment: ExperimentSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _nested: ClassVar[dict[str, type[Serializable]]] = {"experiment": ExperimentSpec}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> BranchSpec:
        raw = dict(value or {})
        experiment = raw.get("experiment")
        if isinstance(experiment, Mapping):
            raw["experiment"] = ExperimentSpec.from_dict(experiment)
        status = raw.get("status")
        if isinstance(status, str):
            raw["status"] = BranchStatus(status)
        return cls(**raw)


@dataclass
class BranchOutcome(Serializable):
    """Worker output; the supervisor remains the authority that applies it."""

    branch_id: str = ""
    status: BranchStatus = BranchStatus.VERIFYING
    claims: list[str] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    proposed_children: list[BranchSpec] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, int | float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    _nested: ClassVar[dict[str, type[Serializable]]] = {
        "evidence": EvidenceRef,
        "artifacts": ArtifactRef,
        "proposed_children": BranchSpec,
    }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> BranchOutcome:
        raw = dict(value or {})
        raw["evidence"] = [
            EvidenceRef.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw.get("evidence", [])
        ]
        raw["artifacts"] = [
            ArtifactRef.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw.get("artifacts", [])
        ]
        raw["proposed_children"] = [
            BranchSpec.from_dict(item) if isinstance(item, Mapping) else item
            for item in raw.get("proposed_children", [])
        ]
        status = raw.get("status")
        if isinstance(status, str):
            raw["status"] = BranchStatus(status)
        return cls(**raw)


@dataclass
class ResearchFrontier(Serializable):
    """Deterministic best-first frontier with progressive widening.

    ``progressive_widening_alpha`` controls how many children a parent may
    expose as evidence accumulates.  A value of 0.5 gives roughly sqrt(n)
    available children and prevents an early hypothesis from flooding the run.
    """

    max_active: int = 2
    progressive_widening_alpha: float = 0.5
    initial_width: int = 1
    branches: dict[str, BranchSpec] = field(default_factory=dict)
    completed_evaluations: int = 0
    expansions: dict[str, int] = field(default_factory=dict)

    _nested: ClassVar[dict[str, type[Serializable]]] = {"branches": BranchSpec}
    _allowed_transitions: ClassVar[dict[BranchStatus, frozenset[BranchStatus]]] = {
        BranchStatus.PROPOSED: frozenset(
            {BranchStatus.SCREENED, BranchStatus.RUNNING, BranchStatus.PRUNED}
        ),
        BranchStatus.SCREENED: frozenset({BranchStatus.RUNNING, BranchStatus.PRUNED}),
        BranchStatus.RUNNING: frozenset(
            {
                BranchStatus.VERIFYING,
                BranchStatus.RETAINED,
                BranchStatus.PRUNED,
                BranchStatus.NEEDS_HUMAN,
            }
        ),
        BranchStatus.VERIFYING: frozenset(
            {BranchStatus.RETAINED, BranchStatus.PRUNED, BranchStatus.NEEDS_HUMAN}
        ),
        BranchStatus.NEEDS_HUMAN: frozenset(
            {BranchStatus.SCREENED, BranchStatus.RUNNING, BranchStatus.PRUNED}
        ),
        BranchStatus.RETAINED: frozenset(),
        BranchStatus.PRUNED: frozenset(),
    }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> ResearchFrontier:
        raw = dict(value or {})
        branches = raw.get("branches", {})
        if isinstance(branches, Mapping):
            raw["branches"] = {
                str(key): BranchSpec.from_dict(item) if isinstance(item, Mapping) else item
                for key, item in branches.items()
            }
        return cls(**raw)

    def __post_init__(self) -> None:
        self.max_active = max(1, int(self.max_active))
        self.initial_width = max(1, int(self.initial_width))
        self.progressive_widening_alpha = min(1.0, max(0.0, float(self.progressive_widening_alpha)))
        self.branches = {
            key: value if isinstance(value, BranchSpec) else BranchSpec.from_dict(value)
            for key, value in self.branches.items()
        }

    @staticmethod
    def score(branch: BranchSpec) -> float:
        """Compute and persist an auditable multi-objective priority score."""

        def bounded(value: float) -> float:
            return min(1.0, max(0.0, float(value)))

        positive = {
            "information_gain": bounded(branch.information_gain),
            "uncertainty_reduction": bounded(branch.uncertainty_reduction),
            "novelty": bounded(branch.novelty),
            "feasibility": bounded(branch.feasibility),
        }
        negative = {
            "cost": bounded(branch.cost),
            "duplication": bounded(branch.duplication),
            "validity_risk": bounded(branch.validity_risk),
        }
        branch.score_components = {**positive, **negative}
        branch.priority_score = sum(positive.values()) / 4 - sum(negative.values()) / 3
        return branch.priority_score

    def add(self, branch: BranchSpec) -> BranchSpec:
        """Add a branch, rejecting accidental replacement of a different spec."""
        if not branch.branch_id:
            raise ValueError("branch_id is required")
        self.score(branch)
        existing = self.branches.get(branch.branch_id)
        if existing is not None and existing.to_dict() != branch.to_dict():
            raise ValueError(f"branch {branch.branch_id!r} already exists with a different spec")
        self.branches[branch.branch_id] = branch
        return branch

    propose = add

    def transition(self, branch_id: str, status: BranchStatus | str) -> BranchSpec:
        """Apply one legal lifecycle transition and return the updated branch."""
        branch = self.branches.get(branch_id)
        if branch is None:
            raise KeyError(f"unknown branch {branch_id!r}")
        target = BranchStatus(status)
        current = BranchStatus(branch.status)
        if target != current and target not in self._allowed_transitions[current]:
            raise ValueError(f"illegal branch transition: {current.value} -> {target.value}")
        branch.status = target
        return branch

    def width_for(self, parent_branch_id: str = "") -> int:
        """Return the number of children currently allowed for a parent."""
        n = max(0, int(self.expansions.get(parent_branch_id, 0)))
        return max(self.initial_width, int(math.ceil((n + 1) ** self.progressive_widening_alpha)))

    allowed_children = width_for

    def can_expand(self, parent_branch_id: str = "") -> bool:
        children = sum(1 for b in self.branches.values() if b.parent_branch_id == parent_branch_id)
        return children < self.width_for(parent_branch_id)

    def select_next(self, *, limit: int | None = None) -> list[BranchSpec]:
        """Select proposed/screened branches by deterministic best-first order."""
        active = sum(
            b.status in {BranchStatus.RUNNING, BranchStatus.VERIFYING}
            for b in self.branches.values()
        )
        slots = max(0, self.max_active - active)
        if limit is not None:
            slots = min(slots, max(0, int(limit)))
        candidates = [
            b
            for b in self.branches.values()
            if b.status in {BranchStatus.PROPOSED, BranchStatus.SCREENED}
        ]
        candidates.sort(key=lambda b: (-self.score(b), b.depth, b.branch_id))
        selected = candidates[:slots]
        for branch in selected:
            self.transition(branch.branch_id, BranchStatus.RUNNING)
        return selected

    def apply_outcome(self, outcome: BranchOutcome) -> list[BranchSpec]:
        """Apply a worker result and admit children subject to widening."""
        branch = self.branches.get(outcome.branch_id)
        if branch is None:
            raise KeyError(f"unknown branch {outcome.branch_id!r}")
        if branch.status not in {BranchStatus.RUNNING, BranchStatus.VERIFYING}:
            raise ValueError(f"branch {outcome.branch_id!r} is not running")
        self.transition(outcome.branch_id, outcome.status)
        self.completed_evaluations += 1
        parent = branch.branch_id
        self.expansions[parent] = self.expansions.get(parent, 0) + 1
        admitted: list[BranchSpec] = []
        for child in outcome.proposed_children:
            if child.parent_branch_id != parent:
                child.parent_branch_id = parent
            if not self.can_expand(parent):
                break
            self.add(child)
            admitted.append(child)
        return admitted

    def pending(self) -> list[BranchSpec]:
        candidates = [
            b
            for b in self.branches.values()
            if b.status in {BranchStatus.PROPOSED, BranchStatus.SCREENED}
        ]
        candidates.sort(key=lambda b: (-self.score(b), b.depth, b.branch_id))
        return candidates

    next_batch = select_next
    next_branches = select_next
