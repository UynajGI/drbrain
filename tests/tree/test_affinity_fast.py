"""The subtractive affinity matrix must equal the per-row rebuild (T04/T29)."""

from __future__ import annotations

import random

import pytest

from drbrain.tree.affinity import (
    SourceProfile,
    SourceSpan,
    affinity_matrix,
    build_pictures,
    profile_affinity,
    structural_affinity,
)
from drbrain.tree.posteriors import PosteriorStage


def _oracle_matrix(stage: PosteriorStage, sources: dict) -> tuple[tuple[float, ...], ...]:
    """The original per-row rebuild; the fast path must reproduce it."""
    matrix = []
    for row_id in stage.row_ids:
        source = sources.get(row_id)
        if source is None:
            matrix.append(tuple(0.0 for _ in stage.component_ids))
            continue
        pictures = build_pictures(stage, sources, exclude_row=row_id)
        matrix.append(
            tuple(
                structural_affinity(source, pictures[component])
                if isinstance(source, SourceSpan)
                else profile_affinity(source, pictures[component])
                for component in stage.component_ids
            )
        )
    return tuple(matrix)


def _approx(matrix):
    return [[pytest.approx(value, rel=1e-9, abs=1e-12) for value in row] for row in matrix]


def _stage(probs) -> PosteriorStage:
    return PosteriorStage(
        stage="global",
        row_ids=tuple(f"r{index}" for index in range(len(probs))),
        component_ids=tuple(f"c{index}" for index in range(len(probs[0]))),
        probs=tuple(tuple(row) for row in probs),
    )


def test_subtractive_matrix_matches_the_rebuild() -> None:
    stage = _stage(
        [
            (0.70, 0.20, 0.10),
            (0.00, 0.40, 0.60),
            (0.50, 0.50, 0.00),
            (0.20, 0.20, 0.60),
            (0.30, 0.30, 0.40),  # no source entry: zero row
            (0.10, 0.90, 0.00),  # empty profile
        ]
    )
    sources = {
        "r0": SourceSpan("doc-a", 1, "b0", 0, 10, 12, ("Intro", "Methods")),
        "r1": SourceSpan("doc-a", 1, "b1", 0, 10, 8, ("Intro", "Results")),
        "r2": SourceProfile(
            parts=(
                ("doc-b", ("Intro",), 5),
                ("doc-b", ("Intro",), 7),  # duplicate part: collapsed by merged()
                ("doc-c", (), 3),
            )
        ),
        "r3": SourceSpan("doc-b", 1, "b2", 0, 10, 0, ("Discussion",)),  # zero tokens
        "r5": SourceProfile(parts=()),
    }
    fast = affinity_matrix(stage, sources)
    assert [list(row) for row in fast] == _approx(_oracle_matrix(stage, sources))


def test_subtractive_matrix_matches_on_a_randomized_stage() -> None:
    rng = random.Random(20260915)
    rows = 48
    cols = 6
    stage = _stage(
        [tuple(round(value, 4) for value in _normalized(rng, cols)) for _ in range(rows)]
    )
    sources: dict = {}
    for index in range(rows):
        if index % 11 == 7:
            continue  # missing source
        if index % 13 == 5:
            sources[f"r{index}"] = SourceProfile(parts=())
            continue
        parts = []
        for part in range(rng.randint(1, 3)):
            doc = f"doc-{rng.randint(0, 5)}"
            path = tuple(
                rng.choice(["Intro", "Methods", "Results", "Discussion"])
                for _ in range(rng.randint(0, 3))
            )
            parts.append((doc, path, rng.randint(1, 40)))
        if index % 3 == 0:
            sources[f"r{index}"] = SourceProfile(parts=tuple(parts))
        else:
            doc, path, tokens = parts[0]
            sources[f"r{index}"] = SourceSpan(doc, 1, f"b{index}", 0, 10, tokens, path)
    fast = affinity_matrix(stage, sources)
    assert [list(row) for row in fast] == _approx(_oracle_matrix(stage, sources))


def _normalized(rng: random.Random, cols: int) -> list[float]:
    values = [rng.random() for _ in range(cols)]
    if rng.random() < 0.25:
        values[rng.randrange(cols)] = 0.0
    total = sum(values) or 1.0
    return [value / total for value in values]
