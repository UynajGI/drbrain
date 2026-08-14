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

    AutoScientists semantics: a hypothesis is **falsifiable** — it carries a
    ``prediction`` (what observation would support it) and a ``falsification``
    (the bar at which it is abandoned). ``status`` walks
    ``proposed → critiqued → confirmed | falsified``.
    """

    statement: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    prediction: str = ""  # 可观察预测：什么证据会支持该假设
    falsification: str = ""  # 证伪标准：什么证据会放弃该假设
    score: float = 0.0
    status: str = "proposed"


class ResearchState(BaseModel):
    """Shared structured state carried through every node via ``Context.store``."""

    task: str = ""
    prior_context: str = ""  # champion + dead-ends from prior cycles (director injects)
    candidates: list[str] = Field(default_factory=list)
    retrieve_candidates: list[str] | None = None  # 检索中间产物（未筛选候选）
    parsed: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    scores: Any = None  # 假设互评分数（list 或 dict）
    verified: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)  # 证伪的假设（dead ends）
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
    falsified: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)


class Settled(Event):
    verified: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)
