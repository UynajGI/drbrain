"""Development-set ablations and parameter freezing (plan T57).

Each ablation changes exactly one declared mechanism -- the structural
affinity weight (λ), the structure hints, the cost gate, the entry view, the
summary expansion or the navigation budget -- so a dev-set comparison can
attribute a difference to that mechanism and nothing else.  Read-side
ablations need no rebuild; build-side ablations produce a variant generation
whose ANN is evaluated with the same scorer as the baselines.

Acceptance thresholds are frozen *before* the holdout is consulted:
:func:`freeze_thresholds` refuses to overwrite an existing record, so a later
result cannot silently move the bar.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drbrain.tree.builder import BuilderConfig


@dataclass(frozen=True)
class Ablation:
    name: str
    mechanism: str
    kind: str  # "build" | "read" | "reference"
    builder: dict[str, Any] = field(default_factory=dict)
    read: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registered ablations; each row changes exactly one mechanism.
ABLATIONS: dict[str, Ablation] = {
    "default": Ablation(
        "default",
        "none (production defaults)",
        "reference",
        rationale="λ=2.0, structure hints on, cost gate on, entry=all layers",
    ),
    "lambda0": Ablation(
        "lambda0",
        "structural affinity off (λ=0)",
        "build",
        {"lam": 0.0},
        rationale="documented ablation: the raw posterior without structural reweighting",
    ),
    "lambda12": Ablation(
        "lambda12",
        "structural affinity at the protocol ceiling (λ=12)",
        "build",
        {"lam": 12.0},
        rationale="the other end of LAMBDA_RANGE",
    ),
    "no_structure_hints": Ablation(
        "no_structure_hints",
        "structure hints off",
        "build",
        {"use_structure_hints": False},
        rationale="grouping proposals no longer follow the document structure",
    ),
    "cost_gate_open": Ablation(
        "cost_gate_open",
        "cost gate budgets opened (no budget-based rejection)",
        "build",
        {"cost": {"summary_input_budget": 1_000_000, "summary_output_budget": 1_000_000}},
        rationale="the membership floor (>=2) stays; only budget rejections disappear",
    ),
    "leaf_only_entry": Ablation(
        "leaf_only_entry",
        "entry restricted to leaves (no summary entry)",
        "read",
        read={"view": "leaf"},
        rationale="tests whether starting from all layers helps recall",
    ),
    "no_expansion": Ablation(
        "no_expansion",
        "summary hits are not expanded into leaves",
        "read",
        read={"expand_regions": False},
        rationale="the parent/neighbourhood jump in the navigator",
    ),
    "tight_budget": Ablation(
        "tight_budget",
        "navigation budget tightened (4 reads, 1 expansion)",
        "read",
        read={"max_expansions": 1, "budget": {"max_calls": 4}},
        rationale="cost/quality trade-off of the read budget",
    ),
}


def ablation(name: str) -> Ablation:
    key = str(name).strip().lower()
    if key not in ABLATIONS:
        raise ValueError(f"unknown ablation {name!r}; expected one of {sorted(ABLATIONS)}")
    return ABLATIONS[key]


def variant_builder_config(name: str, base: BuilderConfig | None = None) -> BuilderConfig:
    """Apply one build-side ablation to a builder config (others untouched)."""
    row = ablation(name)
    config = base or BuilderConfig()
    if not row.builder:
        return config
    overrides = dict(row.builder)
    cost_overrides = overrides.pop("cost", None)
    if cost_overrides:
        overrides["cost"] = dataclasses.replace(config.cost, **cost_overrides)
    return dataclasses.replace(config, **overrides)


def read_overrides(name: str) -> dict[str, Any]:
    """Read-side knobs for one ablation (``{}`` for build-side/reference rows)."""
    return dict(ablation(name).read)


def freeze_thresholds(path: str | Path, payload: dict[str, Any]) -> bool:
    """Record the frozen acceptance thresholds exactly once (T57).

    Returns ``True`` when the record was written, ``False`` when one already
    exists -- rewriting a frozen record would move the bar after seeing
    results, which is precisely what the protocol forbids.
    """
    target = Path(path)
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


__all__ = [
    "ABLATIONS",
    "Ablation",
    "ablation",
    "freeze_thresholds",
    "read_overrides",
    "variant_builder_config",
]
