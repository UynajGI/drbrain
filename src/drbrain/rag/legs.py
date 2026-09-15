"""The unified retrieval-leg registry (plan T43).

The outer layer registers exactly three production legs — ``bm25``,
``vector`` and ``tree``.  The legacy names ``pageindex`` and ``raptor`` used
to be separate candidate pools over overlapping content; they are now folded
into the single tree leg, and combining a legacy alias with the canonical
``tree`` name is a configuration conflict instead of a silent double
registration.  A tree leg contributes one RRF vote no matter how many layers
(leaves, regions) it returns.

``graph`` and ``claims`` remain explicit live extras outside the three
retrieval legs: they are KG sources, not document-recall legs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Production retrieval legs, in canonical order.
CANONICAL_LEGS: tuple[str, ...] = ("bm25", "vector", "tree")

#: Legacy names folded into the single tree leg.
LEG_ALIASES: dict[str, str] = {"pageindex": "tree", "raptor": "tree"}

#: Explicit live extras (KG sources), never part of the three legs.
EXTRA_LEGS: tuple[str, ...] = ("graph", "claims")


class LegConfigError(ValueError):
    """Raised for unknown names or conflicting legacy/new leg naming."""


@dataclass(frozen=True)
class NormalizedLegs:
    legs: tuple[str, ...]
    extras: tuple[str, ...]
    notes: tuple[str, ...]

    def as_list(self) -> list[str]:
        return [*self.legs, *self.extras]


def normalize_legs(wanted: Iterable[str] | None) -> NormalizedLegs:
    """Fold legacy retriever names into the canonical three-leg set.

    - ``pageindex`` and ``raptor`` merge into ``tree`` (noted, one entry).
    - Asking for a canonical leg *and* one of its legacy aliases raises
      :class:`LegConfigError`: the old config implied two candidate pools,
      which would double the RRF weight of overlapping content.
    - Unknown names raise :class:`LegConfigError`.
    """
    requested = [str(name).strip().lower() for name in (wanted or []) if str(name).strip()]
    if not requested:
        requested = ["bm25", "vector"]

    canonical: set[str] = set()
    alias_sources: dict[str, list[str]] = {}
    extras: list[str] = []
    notes: list[str] = []
    for name in requested:
        if name in LEG_ALIASES:
            alias_sources.setdefault(LEG_ALIASES[name], []).append(name)
            continue
        if name in CANONICAL_LEGS:
            canonical.add(name)
            continue
        if name in EXTRA_LEGS:
            if name not in extras:
                extras.append(name)
            continue
        raise LegConfigError(
            f"unknown retriever {name!r}; expected one of {list(CANONICAL_LEGS)} "
            f"(legacy aliases: {sorted(LEG_ALIASES)}), or {list(EXTRA_LEGS)}"
        )

    for leg, sources in alias_sources.items():
        if leg in canonical:
            raise LegConfigError(
                f"{sorted(sources)} and {leg!r} both request the unified {leg} leg; "
                f"keep only {leg!r} (the legacy names are aliases, not separate polls)"
            )
        notes.append(
            f"{'+'.join(sorted(sources))} -> {leg} "
            f"(legacy name folded into the single {leg} leg, one RRF vote)"
        )

    legs = tuple(leg for leg in CANONICAL_LEGS if leg in canonical or leg in alias_sources)
    return NormalizedLegs(legs=legs, extras=tuple(extras), notes=tuple(notes))


__all__ = [
    "CANONICAL_LEGS",
    "EXTRA_LEGS",
    "LEG_ALIASES",
    "LegConfigError",
    "NormalizedLegs",
    "normalize_legs",
]
