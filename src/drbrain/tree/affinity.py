"""Structural affinity A(i,k) for the structural-conditioned soft grouping.

Frozen protocol (plan T04, review constraints): a cluster's *picture* is built
only from the unmodified posterior with the evaluated node excluded, so the
prior can never be circularly redefined by the assignment it reweights.
Pictures aggregate by unique source ranges, weighted by token mass, so a node
reached through several soft parents does not multiply-count its document.

    A(i,k) = doc_share(i, k) * path_similarity(i, k)

``doc_share`` is the token-weighted fraction of cluster ``k``'s picture mass
that lives in node ``i``'s document; zero for a document with no picture mass
(cross-document affinity is 0.0 = neutral, it never hard-blocks).  The path
term compares heading paths of same-document picture spans; missing or empty
headings are neutral (0.0), never a fabricated signal.  The result lies in
``[0, 1]`` and the whole term is an explicit same-document preference: because
the reweighted row is renormalized, boosting a same-document component can
depress a cross-document component below the threshold.  That bias is part of
the design, not a bug; the dev-set lambda must be chosen with cross-document
multi-evidence questions in the loop (T57).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from drbrain.tree.posteriors import PosteriorStage


@dataclass(frozen=True)
class SourceSpan:
    """One unique source range with its structural anchor."""

    local_id: str
    revision: int
    block_id: str
    char_start: int
    char_end: int
    tokens: int
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise ValueError("source span must be non-empty")
        if self.tokens < 0:
            raise ValueError("tokens must be >= 0")

    @property
    def key(self) -> tuple[str, int, str, int, int]:
        return (self.local_id, self.revision, self.block_id, self.char_start, self.char_end)


def path_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalized common-prefix length in ``[0, 1]``; empty paths are neutral."""
    if not a or not b:
        return 0.0
    common = 0
    for left, right in zip(a, b):
        if left.strip().lower() != right.strip().lower():
            break
        common += 1
    return common / max(1, min(len(a), len(b)))


@dataclass
class ClusterPicture:
    """Fixed component profile derived from an unmodified posterior."""

    component_id: str
    spans: list[tuple[SourceSpan, float]] = field(default_factory=list)
    doc_mass: dict[str, float] = field(default_factory=dict)
    path_mass: dict[str, list[tuple[tuple[str, ...], float]]] = field(default_factory=dict)
    total_mass: float = 0.0

    def is_empty(self) -> bool:
        return self.total_mass <= 0.0 or not self.spans


def build_pictures(
    stage: PosteriorStage,
    spans: Mapping[str, SourceSpan],
    *,
    exclude_row: str | None = None,
) -> dict[str, ClusterPicture]:
    """Build one picture per component from unmodified stage probabilities.

    ``exclude_row`` drops the evaluated node's own contribution when building
    the picture it is scored against (T04/T29 self-exclusion rule).
    """
    pictures: dict[str, ClusterPicture] = {
        component: ClusterPicture(component_id=component) for component in stage.component_ids
    }
    for row_id, row in zip(stage.row_ids, stage.probs):
        if exclude_row is not None and row_id == exclude_row:
            continue
        span = spans.get(row_id)
        if span is None:
            continue
        for idx, component in enumerate(stage.component_ids):
            prob = float(row[idx])
            if prob <= 0.0:
                continue
            weight = prob * max(span.tokens, 1)
            picture = pictures[component]
            picture.spans.append((span, weight))
            picture.doc_mass[span.local_id] = picture.doc_mass.get(span.local_id, 0.0) + weight
            entries = picture.path_mass.setdefault(span.local_id, [])
            entries.append((span.heading_path, weight))
            picture.total_mass += weight
    return pictures


def structural_affinity(span: SourceSpan, picture: ClusterPicture) -> float:
    """A(i,k) for one evaluated span against a fixed cluster picture."""
    if picture.is_empty():
        return 0.0
    doc_mass = picture.doc_mass.get(span.local_id, 0.0)
    if doc_mass <= 0.0:
        return 0.0
    doc_share = doc_mass / picture.total_mass
    path_entries = picture.path_mass.get(span.local_id) or []
    if not path_entries:
        return 0.0
    total_weight = sum(weight for _, weight in path_entries)
    if total_weight <= 0.0:
        return 0.0
    similarity = 0.0
    for path, weight in path_entries:
        similarity += weight * path_similarity(span.heading_path, path)
    similarity /= total_weight
    return max(0.0, min(1.0, doc_share * similarity))


def affinity_matrix(
    stage: PosteriorStage,
    spans: Mapping[str, SourceSpan],
) -> tuple[tuple[float, ...], ...]:
    """A(i,k) for every row of ``stage`` with per-row self-exclusion."""
    matrix: list[tuple[float, ...]] = []
    for row_id in stage.row_ids:
        span = spans.get(row_id)
        if span is None:
            matrix.append(tuple(0.0 for _ in stage.component_ids))
            continue
        pictures = build_pictures(stage, spans, exclude_row=row_id)
        matrix.append(
            tuple(
                structural_affinity(span, pictures[component]) for component in stage.component_ids
            )
        )
    return tuple(matrix)
