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


@dataclass(frozen=True)
class SourceProfile:
    """A node's real source distribution: token mass per (document, heading path).

    Leaves carry a single part; upper-layer regions aggregate their
    descendants' parts (design §4.2: an upper region uses its true source
    distribution instead of being pinned to one document).  Duplicate soft
    paths must be collapsed by the caller so the same origin range is not
    counted twice.
    """

    parts: tuple[tuple[str, tuple[str, ...], int], ...]

    def __post_init__(self) -> None:
        if any(tokens < 0 for _local, _path, tokens in self.parts):
            raise ValueError("profile tokens must be >= 0")

    @classmethod
    def from_span(cls, span: SourceSpan) -> SourceProfile:
        return cls(parts=((span.local_id, span.heading_path, span.tokens),))

    def is_empty(self) -> bool:
        return not self.parts or self.token_total() <= 0

    def token_total(self) -> int:
        return sum(max(0, tokens) for _local, _path, tokens in self.parts)

    def documents(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(local_id for local_id, _path, _tokens in self.parts))

    def merged(self) -> SourceProfile:
        """Collapse duplicate (document, path) parts into one token mass."""
        aggregated: dict[tuple[str, tuple[str, ...]], int] = {}
        for local_id, path, tokens in self.parts:
            key = (local_id, tuple(path))
            aggregated[key] = aggregated.get(key, 0) + max(0, int(tokens))
        return SourceProfile(
            parts=tuple((local_id, path, tokens) for (local_id, path), tokens in aggregated.items())
        )


def _as_profile(source: SourceSpan | SourceProfile) -> SourceProfile:
    return SourceProfile.from_span(source) if isinstance(source, SourceSpan) else source


@dataclass
class ClusterPicture:
    """Fixed component profile derived from an unmodified posterior."""

    component_id: str
    spans: list[tuple[SourceSpan, float]] = field(default_factory=list)
    parts: list[tuple[str, tuple[str, ...], float]] = field(default_factory=list)
    doc_mass: dict[str, float] = field(default_factory=dict)
    path_mass: dict[str, list[tuple[tuple[str, ...], float]]] = field(default_factory=dict)
    total_mass: float = 0.0

    def is_empty(self) -> bool:
        return self.total_mass <= 0.0 or not self.parts


def build_pictures(
    stage: PosteriorStage,
    sources: Mapping[str, SourceSpan | SourceProfile],
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
        source = sources.get(row_id)
        if source is None:
            continue
        profile = _as_profile(source).merged()
        if profile.is_empty():
            continue
        for idx, component in enumerate(stage.component_ids):
            prob = float(row[idx])
            if prob <= 0.0:
                continue
            picture = pictures[component]
            for local_id, path, part_tokens in profile.parts:
                weight = prob * max(int(part_tokens), 1)
                picture.parts.append((local_id, tuple(path), weight))
                picture.doc_mass[local_id] = picture.doc_mass.get(local_id, 0.0) + weight
                picture.path_mass.setdefault(local_id, []).append((tuple(path), weight))
                picture.total_mass += weight
    return pictures


def _span_affinity(local_id: str, heading_path: tuple[str, ...], picture: ClusterPicture) -> float:
    if picture.is_empty():
        return 0.0
    doc_mass = picture.doc_mass.get(local_id, 0.0)
    if doc_mass <= 0.0:
        return 0.0
    doc_share = doc_mass / picture.total_mass
    path_entries = picture.path_mass.get(local_id) or []
    if not path_entries:
        return 0.0
    total_weight = sum(weight for _, weight in path_entries)
    if total_weight <= 0.0:
        return 0.0
    similarity = 0.0
    for path, weight in path_entries:
        similarity += weight * path_similarity(heading_path, path)
    similarity /= total_weight
    return max(0.0, min(1.0, doc_share * similarity))


def structural_affinity(span: SourceSpan, picture: ClusterPicture) -> float:
    """A(i,k) for one evaluated span against a fixed cluster picture."""
    return _span_affinity(span.local_id, span.heading_path, picture)


def profile_affinity(profile: SourceProfile, picture: ClusterPicture) -> float:
    """A(i,k) for a multi-source node: token-weighted mean over its parts."""
    parts = profile.merged().parts
    if not parts:
        return 0.0
    total = sum(max(0, tokens) for _local, _path, tokens in parts)
    if total <= 0:
        return 0.0
    value = 0.0
    for local_id, path, tokens in parts:
        value += max(0, tokens) * _span_affinity(local_id, tuple(path), picture)
    return max(0.0, min(1.0, value / total))


def _span_affinity_without(
    local_id: str,
    heading_path: tuple[str, ...],
    picture: ClusterPicture,
    *,
    own_total: float,
    own_doc: float,
    own_paths: Mapping[tuple[str, ...], float],
    path_sums: Mapping[tuple[str, ...], float],
) -> float:
    """``_span_affinity`` against ``picture`` minus the evaluating row's mass.

    Same formula as :func:`_span_affinity` with the row's own contribution
    (``own_*``) subtracted from the picture's aggregates.  ``path_sums`` holds
    the component's per-path weight totals for this document.
    """
    total = picture.total_mass - own_total
    if total <= 0.0:
        return 0.0
    doc_mass = picture.doc_mass.get(local_id, 0.0) - own_doc
    if doc_mass <= 0.0:
        return 0.0
    total_weight = 0.0
    similarity = 0.0
    for path, weight in path_sums.items():
        remaining = weight - own_paths.get(path, 0.0)
        if remaining <= 0.0:
            continue
        total_weight += remaining
        similarity += remaining * path_similarity(heading_path, path)
    if total_weight <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (doc_mass / total) * (similarity / total_weight)))


def _affinity_without(
    source: SourceSpan | SourceProfile,
    picture: ClusterPicture,
    own_parts: Sequence[tuple[str, tuple[str, ...], float]],
    path_sums: Mapping[tuple[str, Mapping[tuple[str, ...], float]]],
) -> float:
    """Affinity for one row against a picture that excludes the row itself."""
    own_total = 0.0
    own_by_doc: dict[str, float] = {}
    own_paths: dict[tuple[str, tuple[str, ...]], float] = {}
    for local_id, path, weight in own_parts:
        own_total += weight
        own_by_doc[local_id] = own_by_doc.get(local_id, 0.0) + weight
        own_paths[(local_id, path)] = own_paths.get((local_id, path), 0.0) + weight

    def _value(local_id: str, path: tuple[str, ...]) -> float:
        return _span_affinity_without(
            local_id,
            path,
            picture,
            own_total=own_total,
            own_doc=own_by_doc.get(local_id, 0.0),
            own_paths={
                entry_path: weight
                for (entry_local, entry_path), weight in own_paths.items()
                if entry_local == local_id
            },
            path_sums=path_sums.get(local_id, {}),
        )

    if isinstance(source, SourceSpan):
        return _value(source.local_id, source.heading_path)
    parts = source.merged().parts
    if not parts:
        return 0.0
    total = sum(max(0, tokens) for _local, _path, tokens in parts)
    if total <= 0:
        return 0.0
    value = 0.0
    for local_id, path, tokens in parts:
        value += max(0, tokens) * _value(local_id, tuple(path))
    return max(0.0, min(1.0, value / total))


def affinity_matrix(
    stage: PosteriorStage,
    sources: Mapping[str, SourceSpan | SourceProfile],
) -> tuple[tuple[float, ...], ...]:
    """A(i,k) for every row of ``stage`` with per-row self-exclusion.

    Rebuilding the self-excluded picture per row is quadratic in the stage
    size (the 20k frontier spent tens of minutes in this loop).  The pictures
    are built once and each row's own contribution — exactly the parts
    :func:`build_pictures` would have added for it — is subtracted per row
    instead; rows without mass in a component use the full picture unchanged.
    """
    components = stage.component_ids
    zero_row = tuple(0.0 for _ in components)
    full = build_pictures(stage, sources)
    path_sums: dict[str, dict[tuple[str, ...], float]] = {}
    own_rows: list[dict[str, list[tuple[str, tuple[str, ...], float]]]] = []
    for row_id, row in zip(stage.row_ids, stage.probs):
        source = sources.get(row_id)
        if source is None:
            own_rows.append({})
            continue
        profile = _as_profile(source).merged()
        if profile.is_empty():
            own_rows.append({})
            continue
        contribution: dict[str, list[tuple[str, tuple[str, ...], float]]] = {}
        for index, component in enumerate(components):
            prob = float(row[index])
            if prob <= 0.0:
                continue
            parts = [
                (local_id, tuple(path), prob * max(int(tokens), 1))
                for local_id, path, tokens in profile.parts
            ]
            contribution[component] = parts
            per_doc = path_sums.setdefault(component, {})
            for local_id, path, weight in parts:
                totals = per_doc.setdefault(local_id, {})
                totals[path] = totals.get(path, 0.0) + weight
        own_rows.append(contribution)

    matrix: list[tuple[float, ...]] = []
    for row_id, row_own in zip(stage.row_ids, own_rows):
        source = sources.get(row_id)
        if source is None:
            matrix.append(zero_row)
            continue
        values = []
        for component in components:
            picture = full[component]
            own_parts = row_own.get(component)
            if own_parts:
                values.append(
                    _affinity_without(source, picture, own_parts, path_sums.get(component, {}))
                )
            else:
                values.append(
                    structural_affinity(source, picture)
                    if isinstance(source, SourceSpan)
                    else profile_affinity(source, picture)
                )
        matrix.append(tuple(values))
    return tuple(matrix)
