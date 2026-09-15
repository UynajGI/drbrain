"""Structure-conditioned soft assignment over canonical nodes (plan T29/T30).

This module joins the storage layer to the frozen protocols:

* ``node_source_profile`` turns a canonical node into its real source
  distribution — a leaf is one span from ``content_blocks``, a region
  aggregates its descendants' unique leaves (soft multi-parent paths are
  collapsed, so the same origin range is never counted twice).
* ``stage_affinities`` builds the self-excluded pictures and computes the
  structural affinity matrix for a fitted stage.
* ``soft_assignment`` applies the frozen reweighting and strict threshold to
  produce multi-parent candidate groups; ``lam=0`` reproduces the stage's own
  membership exactly, and rows that end up with no component are still
  returned (with an empty member list) so the builder can keep them reachable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from drbrain.tree.affinity import (
    SourceProfile,
    SourceSpan,
    affinity_matrix,
    build_pictures,
    profile_affinity,
    structural_affinity,
)
from drbrain.tree.posteriors import PosteriorStage, check_lambda


class AssignmentError(RuntimeError):
    """The assignment step was given inconsistent inputs."""


@dataclass(frozen=True)
class AssignmentCandidate:
    """One candidate parent group produced by the grouping step."""

    component_id: str
    members: tuple[tuple[str, float], ...]
    origin: str = "semantic"
    empty: bool = False

    def member_ids(self) -> tuple[str, ...]:
        return tuple(node_id for node_id, _weight in self.members)


def _leaf_profile(row: Mapping, block: Mapping, count_tokens) -> SourceProfile:
    text = str(block.get("text") or "")
    char_start = int(row.get("char_start") or 0)
    char_end = int(row.get("char_end") or len(text))
    snippet = text[char_start:char_end]
    tokens = int(count_tokens(snippet)) if snippet else 0
    heading_raw = block.get("heading_path") or "[]"
    try:
        import json

        heading = (
            tuple(json.loads(heading_raw)) if isinstance(heading_raw, str) else tuple(heading_raw)
        )
    except (TypeError, ValueError):
        heading = ()
    return SourceProfile(parts=((str(row.get("local_id") or ""), heading, max(1, tokens)),))


def node_source_profile(db, node_id: str, *, count_tokens=None) -> SourceProfile:
    """Real source distribution of one canonical node (leaf or region)."""
    if count_tokens is None:
        from drbrain.services.tokens import count_tokens as _count

        count_tokens = _count
    row = db.get_tree_node(node_id)
    if row is None:
        raise AssignmentError(f"unknown tree node {node_id!r}")
    return _node_profile(db, row, count_tokens, seen=set())


def _node_profile(db, row: Mapping, count_tokens, *, seen: set[str]) -> SourceProfile:
    node_id = str(row["node_id"])
    if node_id in seen:
        raise AssignmentError(f"node graph contains a cycle at {node_id!r}")
    seen.add(node_id)
    if row["kind"] == "leaf":
        block_id = row.get("block_id")
        blocks = [
            block
            for block in db.get_content_blocks(row["local_id"], int(row["doc_revision"]))
            if block["block_id"] == block_id
        ]
        if not blocks:
            raise AssignmentError(f"leaf {node_id!r} references a missing block {block_id!r}")
        return _leaf_profile(row, blocks[0], count_tokens)
    parts: list[tuple[str, tuple[str, ...], int]] = []
    for child in db.get_tree_children(node_id):
        child_row = db.get_tree_node(child["child_id"])
        if child_row is None:
            raise AssignmentError(f"region {node_id!r} references missing child")
        parts.extend(_node_profile(db, child_row, count_tokens, seen=set(seen)).parts)
    return SourceProfile(parts=tuple(parts)).merged()


def node_source_profiles(
    db, node_ids: Iterable[str], *, count_tokens=None
) -> dict[str, SourceProfile]:
    return {
        node_id: node_source_profile(db, node_id, count_tokens=count_tokens) for node_id in node_ids
    }


#: One recursive lookup for every leaf reachable from the given nodes.  The
#: per-node helpers above refetch the whole document per leaf; batch callers
#: (the cost gates walk hundreds of proposals over thousands of members) use
#: this instead — one query replaces ``O(members)`` document reads.
_LEAF_ROWS_SQL = """
WITH RECURSIVE members(root_id, node_id, depth) AS (
    SELECT value, value, 0 FROM json_each(?)
    UNION ALL
    SELECT m.root_id, c.child_id, m.depth + 1
    FROM members m
    JOIN tree_node_children c ON c.parent_id = m.node_id
    WHERE m.depth < 32
)
SELECT m.root_id, n.node_id, n.local_id, n.doc_revision, n.block_id,
       n.char_start, n.char_end, b.text, b.heading_path
FROM members m
JOIN tree_nodes n ON n.node_id = m.node_id
LEFT JOIN content_blocks b
  ON b.local_id = n.local_id AND b.revision = n.doc_revision AND b.block_id = n.block_id
WHERE n.kind = 'leaf'
"""


def _leaf_rows_for_nodes(db, node_ids: Sequence[str]) -> list[dict]:
    ids = [str(node_id) for node_id in dict.fromkeys(node_ids)]
    if not ids:
        return []
    import json

    cursor = db.conn.execute(_LEAF_ROWS_SQL, (json.dumps(ids),))
    columns = [item[0] for item in cursor.description or ()]
    rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    for row in rows:
        if row.get("text") is None:
            raise AssignmentError(
                f"leaf {row['node_id']!r} references a missing block {row.get('block_id')!r}"
            )
    return rows


def _span_from_fields(
    *,
    local_id: str,
    revision: int,
    block_id: str,
    char_start: int,
    char_end: int,
    text: str,
    heading_raw: object,
    count_tokens,
) -> SourceSpan:
    snippet = text[char_start:char_end]
    return SourceSpan(
        local_id=str(local_id),
        revision=max(1, int(revision or 1)),
        block_id=str(block_id),
        char_start=char_start,
        char_end=max(char_start + 1, char_end),
        tokens=max(1, int(count_tokens(snippet))) if snippet else 0,
        heading_path=_heading_path_of(heading_raw),
    )


def _span_from_row(row: Mapping, block: Mapping, count_tokens) -> SourceSpan:
    text = str(block.get("text") or "")
    char_start = int(row.get("char_start") or 0)
    char_end = int(row.get("char_end") or len(text))
    return _span_from_fields(
        local_id=str(row.get("local_id") or ""),
        revision=int(row.get("doc_revision") or 1),
        block_id=str(row.get("block_id")),
        char_start=char_start,
        char_end=char_end,
        text=text,
        heading_raw=block.get("heading_path"),
        count_tokens=count_tokens,
    )


def _heading_path_of(raw: object) -> tuple[str, ...]:
    import json

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    return ()


def leaf_spans_of_nodes(
    db, node_ids: Sequence[str], *, count_tokens=None
) -> dict[str, list[SourceSpan]]:
    """Batch :func:`leaf_spans_of_node`: one query for every reachable leaf."""
    if count_tokens is None:
        from drbrain.services.tokens import count_tokens as _count

        count_tokens = _count
    out: dict[str, list[SourceSpan]] = {str(node_id): [] for node_id in node_ids}
    for row in _leaf_rows_for_nodes(db, node_ids):
        text = str(row.get("text") or "")
        char_start = int(row.get("char_start") or 0)
        char_end = int(row.get("char_end") or len(text))
        out.setdefault(str(row["root_id"]), []).append(
            _span_from_fields(
                local_id=str(row.get("local_id") or ""),
                revision=int(row.get("doc_revision") or 1),
                block_id=str(row.get("block_id")),
                char_start=char_start,
                char_end=char_end,
                text=text,
                heading_raw=row.get("heading_path"),
                count_tokens=count_tokens,
            )
        )
    return out


def source_profiles_for_nodes(
    db, node_ids: Sequence[str], *, count_tokens=None
) -> dict[str, SourceProfile]:
    """Batch :func:`node_source_profile`: one query for every descendant leaf."""
    if count_tokens is None:
        from drbrain.services.tokens import count_tokens as _count

        count_tokens = _count
    parts: dict[str, list[tuple[str, tuple[str, ...], int]]] = {
        str(node_id): [] for node_id in node_ids
    }
    for row in _leaf_rows_for_nodes(db, node_ids):
        text = str(row.get("text") or "")
        char_start = int(row.get("char_start") or 0)
        char_end = int(row.get("char_end") or len(text))
        snippet = text[char_start:char_end]
        parts.setdefault(str(row["root_id"]), []).append(
            (
                str(row.get("local_id") or ""),
                _heading_path_of(row.get("heading_path")),
                int(count_tokens(snippet)) if snippet else 0,
            )
        )
    return {node_id: SourceProfile(parts=tuple(items)).merged() for node_id, items in parts.items()}


def leaf_spans_of_node(db, node_id: str, *, count_tokens=None) -> list[SourceSpan]:
    """Every unique leaf span under a node (used for exact cost coverage)."""
    if count_tokens is None:
        from drbrain.services.tokens import count_tokens as _count

        count_tokens = _count
    row = db.get_tree_node(node_id)
    if row is None:
        raise AssignmentError(f"unknown tree node {node_id!r}")
    return _collect_leaf_spans(db, row, count_tokens, seen=set())


def _collect_leaf_spans(db, row: Mapping, count_tokens, *, seen: set[str]) -> list[SourceSpan]:
    node_id = str(row["node_id"])
    if node_id in seen:
        raise AssignmentError(f"node graph contains a cycle at {node_id!r}")
    seen.add(node_id)
    if row["kind"] == "leaf":
        block = _block_row(db, row)
        text = str(block.get("text") or "")
        char_start = int(row.get("char_start") or 0)
        char_end = int(row.get("char_end") or len(text))
        snippet = text[char_start:char_end]
        return [
            SourceSpan(
                local_id=str(row.get("local_id") or ""),
                revision=int(row.get("doc_revision") or 1),
                block_id=str(row.get("block_id")),
                char_start=char_start,
                char_end=max(char_start + 1, char_end),
                tokens=max(1, int(count_tokens(snippet))) if snippet else 0,
                heading_path=_heading_of(block),
            )
        ]
    spans: list[SourceSpan] = []
    for child in db.get_tree_children(node_id):
        child_row = db.get_tree_node(child["child_id"])
        if child_row is None:
            raise AssignmentError(f"region {node_id!r} references missing child")
        spans.extend(_collect_leaf_spans(db, child_row, count_tokens, seen=set(seen)))
    return spans


def _block_row(db, row: Mapping) -> Mapping:
    block_id = row.get("block_id")
    blocks = [
        block
        for block in db.get_content_blocks(row["local_id"], int(row["doc_revision"]))
        if block["block_id"] == block_id
    ]
    if not blocks:
        raise AssignmentError(f"leaf {row['node_id']!r} references a missing block {block_id!r}")
    return blocks[0]


def _heading_of(block: Mapping) -> tuple[str, ...]:
    import json

    raw = block.get("heading_path") or "[]"
    try:
        return tuple(json.loads(raw)) if isinstance(raw, str) else tuple(raw)
    except (TypeError, ValueError):
        return ()


def stage_affinities(
    stage: PosteriorStage,
    profiles: Mapping[str, SourceProfile | SourceSpan],
) -> tuple[tuple[float, ...], ...]:
    """A(i,k) matrix for a stage, with per-row self-exclusion."""
    return affinity_matrix(stage, profiles)


def picture_for(
    stage: PosteriorStage,
    profiles: Mapping[str, SourceProfile | SourceSpan],
    component_id: str,
    *,
    exclude_row: str | None = None,
):
    pictures = build_pictures(stage, profiles, exclude_row=exclude_row)
    if component_id not in pictures:
        raise AssignmentError(f"unknown component {component_id!r}")
    return pictures[component_id]


def row_affinity(
    profile: SourceProfile | SourceSpan,
    stage: PosteriorStage,
    profiles: Mapping[str, SourceProfile | SourceSpan],
    row_id: str,
) -> tuple[float, ...]:
    """Affinity row for one node: self-excluded pictures, one value per component."""
    pictures = build_pictures(stage, profiles, exclude_row=row_id)
    values = []
    for component in stage.component_ids:
        picture = pictures[component]
        values.append(
            structural_affinity(profile, picture)
            if isinstance(profile, SourceSpan)
            else profile_affinity(profile, picture)
        )
    return tuple(values)


def reweighted_stage(
    stage: PosteriorStage,
    profiles: Mapping[str, SourceProfile | SourceSpan],
    *,
    lam: float,
    with_affinity: bool = True,
) -> PosteriorStage:
    """Apply the frozen structural prior to one stage (T04/T29/T30).

    Every stage reweights its own raw posterior before the strict threshold, so
    the prior enters here and nowhere else — the global stage included, where
    the corrected membership decides which rows share a local fit.  ``lam=0``
    (or ``with_affinity=False``, the documented ablation) returns the stage
    unchanged, keeping the raw posterior measurable.
    """
    effective_lambda = check_lambda(lam)
    if not with_affinity or effective_lambda == 0.0:
        return stage
    matrix = stage_affinities(stage, profiles)
    return PosteriorStage(
        stage=stage.stage,
        row_ids=stage.row_ids,
        component_ids=stage.component_ids,
        probs=stage.probs,
        threshold=stage.threshold,
        lam=effective_lambda,
        affinity=matrix,
        subset_of=stage.subset_of,
    ).reweighted()


def soft_assignment(
    stage: PosteriorStage,
    profiles: Mapping[str, SourceProfile | SourceSpan],
    *,
    lam: float,
    with_affinity: bool = True,
) -> list[AssignmentCandidate]:
    """Reweight one stage with structural affinity and threshold it.

    ``with_affinity=False`` (or ``lam=0``) uses the raw posterior, which is the
    documented ablation condition; the returned candidates always cover every
    row of the stage (empty membership is explicit).
    """
    stage = reweighted_stage(stage, profiles, lam=lam, with_affinity=with_affinity)
    membership = stage.membership()
    by_component: dict[str, list[tuple[str, float]]] = {
        component: [] for component in stage.component_ids
    }
    for row_id in stage.row_ids:
        for component, weight in membership[row_id]:
            by_component[component].append((row_id, float(weight)))
    candidates: list[AssignmentCandidate] = []
    for component, members in by_component.items():
        ordered = tuple(sorted(members, key=lambda item: (-item[1], item[0])))
        candidates.append(
            AssignmentCandidate(
                component_id=component,
                members=ordered,
                empty=not ordered,
            )
        )
    return candidates


def assignment_coverage(candidates: Sequence[AssignmentCandidate]) -> dict:
    """Audit for one assignment pass (T30/T35 acceptance metrics)."""
    assigned: set[str] = set()
    multi_parent = 0
    rows_seen: dict[str, int] = {}
    for candidate in candidates:
        for node_id, _weight in candidate.members:
            assigned.add(node_id)
            rows_seen[node_id] = rows_seen.get(node_id, 0) + 1
    multi_parent = sum(1 for count in rows_seen.values() if count > 1)
    return {
        "candidates": len(candidates),
        "empty_candidates": sum(1 for candidate in candidates if candidate.empty),
        "assigned_rows": len(assigned),
        "multi_parent_rows": multi_parent,
    }
