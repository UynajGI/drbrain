"""Research-loop schemas: shared state, evidence, hypotheses, and node events.

Every node exchanges a typed pydantic :class:`Event` (its input/output schema)
rather than free-form chat — this is the "非自由群聊 / Schema 交换" contract the
competition spec requires. The shared :class:`ResearchState` rides through
``Context.store`` so every node reads/writes the same structured state.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.workflow import Event
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A unit of provenance-backed evidence: paper → page → snippet → value."""

    paper_id: str = ""
    page: int | None = None
    snippet: str = ""
    value: Any = None
    unit: str = ""
    conditions: dict[str, Any] = Field(default_factory=dict)
    provenance: str = ""
    authority: str = ""


class Hypothesis(BaseModel):
    """A candidate hypothesis derived from a gap, tracked through the loop.

    ``status`` walks ``proposed → critiqued → confirmed | rejected``.
    """

    statement: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    status: str = "proposed"


class ResearchState(BaseModel):
    """Shared structured state carried through every node via ``Context.store``."""

    task: str = ""
    candidates: list[str] = Field(default_factory=list)
    parsed: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    report: str = ""


# ── node transition events ────────────────────────────────────────────────────


class TaskPlanned(Event):
    task: str = ""


class Retrieved(Event):
    candidates: list[str] = Field(default_factory=list)
    attempt: int = 1


class RetrieveAgain(Event):
    """Loop-back signal: filter asks retrieve to run again (bounded by attempts)."""

    attempt: int = 1
    reason: str = ""


class Filtered(Event):
    selected: list[str] = Field(default_factory=list)


class Parsed(Event):
    docs: list[str] = Field(default_factory=list)


class Extracted(Event):
    entities: list[str] = Field(default_factory=list)


class Normalized(Event):
    entities: list[str] = Field(default_factory=list)


class Fused(Event):
    entities: list[str] = Field(default_factory=list)


class GapsIdentified(Event):
    gaps: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class Critiqued(Event):
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class Verified(Event):
    verified: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)


class Settled(Event):
    verified: list[str] = Field(default_factory=list)
