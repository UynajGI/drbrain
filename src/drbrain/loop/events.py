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

    evidence_id: str = ""
    generation: str = ""
    document_locator: dict[str, Any] = Field(default_factory=dict)
    chunk_locator: dict[str, Any] = Field(default_factory=dict)
    content_checksum: str = ""
    excerpt_checksum: str = ""
    content_length: int | None = None
    excerpt_length: int | None = None
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    retriever: str = ""
    rank: int | None = None
    score: float | None = None
    tool_call_id: str = ""
    paper_id: str = ""
    page: int | None = None
    snippet: str = ""
    value: Any = None
    unit: str = ""
    conditions: dict[str, Any] = Field(default_factory=dict)
    provenance: str = ""
    authority: str = ""


class EvidenceBundle(BaseModel):
    """One tool-backed retrieval result, with immutable evidence references."""

    bundle_id: str = ""
    generation: str = ""
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    retriever: str = ""
    tool_call_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    records: list[Evidence] = Field(default_factory=list)


class Hypothesis(BaseModel):
    """A candidate hypothesis derived from a gap, tracked through the loop.

    AutoScientists semantics: a hypothesis is **falsifiable** — it carries a
    ``prediction`` (what observation would support it) and a ``falsification``
    (the bar at which it is abandoned). ``status`` walks
    ``proposed → critiqued → confirmed | falsified`` (plus ``discarded`` for
    hypotheses the critic filtered out at T5 — they never reach verification).
    """

    claim_id: str = ""
    proposal_id: str = ""
    queue_item_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    statement: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    prediction: str = ""  # 可观察预测：什么证据会支持该假设
    falsification: str = ""  # 证伪标准：什么证据会放弃该假设
    score: float = 0.0
    status: str = "proposed"


class Verification(BaseModel):
    """Structured per-hypothesis verification result (T3).

    The verifier agent reports evidence counts (``supports`` / ``refutes`` /
    ``orthogonal``) plus optional numeric evidence; downstream code derives the
    verdict from the counts — never from an LLM sentence. ``status`` is
    code-derived: ``verified`` | ``falsified`` | ``prediction``.

    T4 compute gate: ``computed`` / ``value`` are **human-readable summaries
    only** — the code-level evidence that a real computation ran is ``job_id``,
    produced by the dedicated **compute node** (``run_python(mode="async")``,
    AutoScientists ROLE-GPU) — never by the verifier, which only counts evidence.
    The corresponding on-disk job artifacts (``<job_id>.json`` + ``<job_id>.log``
    with a parseable number) must exist for the entry to be ``verified``.
    """

    claim_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    statement: str
    supports: int = 0  # 支持 prediction 的证据条数
    refutes: int = 0  # 反驳的证据条数
    orthogonal: int = 0  # 无关 / 无法判定的证据条数
    evidence: str = ""  # 证据摘要（来源 / 要点）
    computed: str = ""  # 实际计算结果（给人看的摘要，来自 compute 节点）；无实算则为空
    value: float | None = None  # 结构化数值（可选，摘要用）
    unit: str = ""  # 数值单位（可选）
    job_id: str = ""  # T4: compute 节点 run_python(mode=async) 返回的后台作业 id（证据是落盘文件）
    status: str = "prediction"  # code-derived: verified | falsified | prediction


class ResearchState(BaseModel):
    """Shared structured state carried through every node via ``Context.store``."""

    task: str = ""
    prior_context: str = ""  # champion + dead-ends from prior cycles (director injects)
    prior_champion: list[str] = Field(default_factory=list)  # T6: structured champion list
    prior_rejected: list[str] = Field(default_factory=list)  # T6: structured dead-ends list
    candidates: list[str] = Field(default_factory=list)
    retrieve_candidates: list[str] | None = None  # 检索中间产物（未筛选候选）
    parsed: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    scores: Any = None  # 假设互评分数（list 或 dict）
    verifications: list[Verification] = Field(default_factory=list)  # T3: 三角验证计数
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


class Computed(Event):
    """Compute-node results: per-hypothesis async job ids (T4 evidence).

    The compute node (AutoScientists ROLE-GPU: run experiments, record results)
    runs a real computation for each critiqued hypothesis BEFORE verification.
    ``job_ids`` maps each statement to the ``run_python(mode="async")`` job id
    whose on-disk artifacts carry the numeric result — the only evidence the
    verify node's T4 gate trusts. ``summaries`` are optional human-readable
    compute summaries. ``hypotheses`` carries the critiqued hypotheses through
    so verify keeps operating on the same event payload it always did.
    """

    hypotheses: list[Hypothesis] = Field(default_factory=list)  # critiqued (KEEP) hypotheses
    job_ids: dict[str, str] = Field(default_factory=dict)  # statement → run_python(async) job_id
    summaries: dict[str, str] = Field(default_factory=dict)  # statement → 实算摘要（可选）
    experiment_ids: dict[str, str] = Field(default_factory=dict)  # statement → durable experiment


class Verified(Event):
    verified: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)  # T3 三角验证计数
    experiment_ids: dict[str, str] = Field(default_factory=dict)  # statement → durable experiment


class Settled(Event):
    verified: list[str] = Field(default_factory=list)
    falsified: list[str] = Field(default_factory=list)
