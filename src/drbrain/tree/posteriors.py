"""Two-stage posterior protocol (plan T04; frozen by docs/unified-tree-algorithms.md).

Upstream RAPTOR computes a GMM posterior, thresholds it into labels, and then
fits independent local GMMs inside each global label.  The unified tree keeps
the *raw* posterior at both stages and applies the structural reweighting
separately in each stage:

    p~(k|i) = p(k|i) exp(lam A(i,k)) / sum_j p(j|i) exp(lam A(i,j))

Hard rules (do not weaken):

* ``lam == 0`` returns the input probabilities unchanged, so the ablation
  condition reproduces the same stage membership as the unmodified run.
* Global and local probabilities are never multiplied and then thresholded
  as one number; each stage thresholds its own reweighted posterior with the
  upstream strict ``>`` comparison.
* Local components are namespaced ``<global>.<local>`` because two local UMAP
  spaces are not comparable.
* A row may end with no membership above threshold; that is carried
  explicitly instead of silently dropping the node.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Legal structural-prior strength.  The dev-set choice is frozen later
#: (T57); 0.0 is the documented ablation condition.
LAMBDA_RANGE: tuple[float, float] = (0.0, 12.0)
DEFAULT_LAMBDA: float = 2.0

#: Upstream RAPTOR soft-membership threshold (strict ``>``).
DEFAULT_THRESHOLD: float = 0.1


def check_lambda(value: float) -> float:
    lo, hi = LAMBDA_RANGE
    value = float(value)
    if not (lo <= value <= hi):
        raise ValueError(f"lambda {value} outside frozen range [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class PosteriorStage:
    """One fitting stage's posterior over ``row_ids`` x components.

    ``affinity`` is the structural compatibility ``A(i,k)`` aligned with
    ``probs``.  It is optional: with ``lam == 0`` no affinity is consulted.
    """

    stage: str
    row_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    probs: tuple[tuple[float, ...], ...]
    threshold: float = DEFAULT_THRESHOLD
    lam: float = 0.0
    affinity: tuple[tuple[float, ...], ...] | None = None
    subset_of: str | None = None  # global component id for local stages

    def __post_init__(self) -> None:
        if self.stage not in {"global", "local"}:
            raise ValueError(f"stage must be global or local, got {self.stage!r}")
        if not self.row_ids:
            raise ValueError("posterior stage requires rows")
        if not self.component_ids:
            raise ValueError("posterior stage requires components")
        if len(self.row_ids) != len(self.probs):
            raise ValueError("row_ids and probs length mismatch")
        for row in self.probs:
            if len(row) != len(self.component_ids):
                raise ValueError("probability row length does not match components")
            total = sum(float(v) for v in row)
            if row and total <= 0:
                raise ValueError("probability row sums to zero")
        check_lambda(self.lam)
        if self.lam > 0.0 and self.affinity is None:
            raise ValueError("lam > 0 requires an affinity matrix")
        if self.affinity is not None:
            if len(self.affinity) != len(self.row_ids):
                raise ValueError("affinity row count mismatch")
            for row in self.affinity:
                if len(row) != len(self.component_ids):
                    raise ValueError("affinity row length mismatch")
                if any(not (0.0 <= float(v) <= 1.0) for v in row):
                    raise ValueError("affinity values must lie in [0, 1]")

    def reweighted(self, lam: float | None = None) -> PosteriorStage:
        """Apply the structural reweighting; ``lam == 0`` is a no-op."""
        effective = self.lam if lam is None else check_lambda(lam)
        if effective == 0.0:
            return self
        assert self.affinity is not None  # guaranteed by __post_init__
        new_rows: list[tuple[float, ...]] = []
        for probs, aff in zip(self.probs, self.affinity):
            weights = [math.exp(effective * float(a)) for a in aff]
            raw = [float(p) * w for p, w in zip(probs, weights)]
            total = sum(raw)
            if total <= 0:
                raise ValueError("reweighting produced an empty row")
            new_rows.append(tuple(value / total for value in raw))
        return PosteriorStage(
            stage=self.stage,
            row_ids=self.row_ids,
            component_ids=self.component_ids,
            probs=tuple(new_rows),
            threshold=self.threshold,
            lam=effective,
            affinity=self.affinity,
            subset_of=self.subset_of,
        )

    def membership(self) -> dict[str, tuple[tuple[str, float], ...]]:
        """Strict-``>`` thresholded membership; empty rows are kept."""
        result: dict[str, tuple[tuple[str, float], ...]] = {}
        for row_id, row in zip(self.row_ids, self.probs):
            result[row_id] = tuple(
                (component, float(prob))
                for component, prob in zip(self.component_ids, row)
                if float(prob) > self.threshold
            )
        return result

    def labels(self) -> dict[str, tuple[str, ...]]:
        return {
            row_id: tuple(component for component, _ in members)
            for row_id, members in self.membership().items()
        }

    def component_subsets(self) -> dict[str, tuple[str, ...]]:
        """Row ids per component under this stage's membership (order kept)."""
        subsets: dict[str, list[str]] = {component: [] for component in self.component_ids}
        for row_id, members in self.membership().items():
            for component, _ in members:
                subsets[component].append(row_id)
        return {component: tuple(rows) for component, rows in subsets.items()}


def local_component_id(global_component: str, local_index: int) -> str:
    """Namespace a local component under its global parent."""
    return f"{global_component}.{local_index}"


def stage_summary(stage: PosteriorStage) -> dict[str, Any]:
    """JSON-safe stage description for acceptance records."""
    return {
        "stage": stage.stage,
        "rows": len(stage.row_ids),
        "components": len(stage.component_ids),
        "threshold": stage.threshold,
        "lambda": stage.lam,
        "subset_of": stage.subset_of,
        "assigned": sum(1 for members in stage.membership().values() if members),
        "unassigned": sum(1 for members in stage.membership().values() if not members),
    }


def staged_membership_stub(
    row_ids: Sequence[str],
    probs: Sequence[Sequence[float]],
    components: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Mapping[str, Sequence[str]]:
    """Convenience wrapper used by fixtures: raw global posteriors only."""
    return PosteriorStage(
        stage="global",
        row_ids=tuple(row_ids),
        component_ids=tuple(components),
        probs=tuple(tuple(float(v) for v in row) for row in probs),
        threshold=threshold,
    ).labels()


def merge_memberships(
    first: Mapping[str, Iterable[str]], second: Mapping[str, Iterable[str]]
) -> dict[str, tuple[str, ...]]:
    """Union two membership maps without dropping unassigned rows."""
    keys = list(dict.fromkeys([*first.keys(), *second.keys()]))
    merged: dict[str, tuple[str, ...]] = {}
    for key in keys:
        merged[key] = tuple(dict.fromkeys([*(first.get(key) or ()), *(second.get(key) or ())]))
    return merged
