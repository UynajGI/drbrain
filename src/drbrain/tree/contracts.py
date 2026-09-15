"""Frozen contracts for the unified tree RAG (plan T03/T06).

Coordinate and identity conventions are normative; later modules must not
invent parallel notions of a block, a node, or a stage state.

Conventions
-----------
* ``ContentBlock`` char offsets are half-open ``[char_start, char_end)`` into
  the canonical document text.  Blocks are contiguous and ordered: block
  ``ordinal`` 0 starts at 0, each next block starts where the previous ends,
  and the last block ends at ``len(canonical)``.  Concatenating ``text`` in
  ``ordinal`` order reproduces the canonical text verbatim, so separators,
  whitespace and residual material live inside exactly one block.
* PDF pages are 1-based inclusive; MD/TeX lines are 1-based inclusive.  A
  material exposes one locator family, never a fabricated mix.  PDF blocks
  may still carry line offsets when the parser provides them (additive, not
  a substitute for pages).
* A leaf's sub-range is half-open ``[char_start, char_end)`` inside its
  block text and must be non-empty.  An unresolved end (``None``) means
  "to the end of the block" and is resolved before persistence.
* Node identity includes content and member revisions.  ``content_hash``
  covers only the text (safe to reuse embeddings across provenance);
  ``fingerprint`` covers identity, content and members (change detection).
  Identical text in different papers keeps different node ids.
* Region identity covers the member set *and* the generation contract, so a
  different member set or a different summary contract is a different node
  rather than a silent overwrite.  Cross-paper regions carry no page range.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

CONTRACT_SCHEMA = "tree-contract-v1"

NODE_KINDS: tuple[str, ...] = ("leaf", "region")

#: Persisted node lifecycle.  ``staging`` nodes are never visible to readers.
NODE_STATES: tuple[str, ...] = ("staging", "ready", "failed", "stale")

#: Per-paper content revision lifecycle.
DOCUMENT_STATES: tuple[str, ...] = ("ready", "stale", "failed")

#: Audit-only provenance for how a candidate group was proposed.
CHILD_ORIGINS: tuple[str, ...] = ("structure", "semantic", "mixed", "manual")

#: Reasons a candidate parent group can be rejected (T05 protocol).
REJECTION_REASONS: tuple[str, ...] = (
    "invalid_members",
    "single_member",
    "duplicate_group",
    "input_over_budget",
    "no_compression_gain",
    "empty_summary",
    "summary_truncated",
    "summary_over_budget",
    "coverage_incomplete",
    "cycle_risk",
    "model_failure",
    "budget_exhausted",
)


class StageState(StrEnum):
    """Stage-level state for ingest/prepare/ask reporting (T06)."""

    ABSENT = "absent"
    RUNNING = "running"
    PARTIAL = "partial"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    STALE = "stale"


#: Severity ranking used when a composite stage summarizes parts.  A stage
#: containing a failed part is failed; a stale part outranks progress.
_STAGE_SEVERITY: dict[StageState, int] = {
    StageState.READY: 0,
    StageState.ABSENT: 1,
    StageState.DEGRADED: 2,
    StageState.PARTIAL: 3,
    StageState.RUNNING: 4,
    StageState.STALE: 5,
    StageState.FAILED: 6,
}

STAGE_STATES: tuple[str, ...] = tuple(state.value for state in StageState)

#: Stages a reader can serve from.  ``degraded`` is complete but lossy.
_QUERYABLE = {StageState.READY, StageState.DEGRADED}


def combine_stage_states(states: Iterable[StageState | str]) -> StageState:
    """Worst-wins aggregation; empty input is ``ABSENT``."""
    worst = StageState.ABSENT
    seen = False
    for raw in states:
        state = raw if isinstance(raw, StageState) else StageState(str(raw))
        if not seen:
            worst = state
            seen = True
            continue
        if _STAGE_SEVERITY[state] > _STAGE_SEVERITY[worst]:
            worst = state
    return worst


def is_queryable(state: StageState | str) -> bool:
    return (state if isinstance(state, StageState) else StageState(str(state))) in _QUERYABLE


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def content_block_id(local_id: str, revision: int, ordinal: int) -> str:
    """Stable block id: provenance plus position, never text-only."""
    _require(bool(local_id) and local_id == local_id.strip(), "local_id must be nonempty")
    _require(isinstance(revision, int) and revision >= 1, "revision must be a positive int")
    _require(isinstance(ordinal, int) and ordinal >= 0, "ordinal must be a nonnegative int")
    return "cb-" + _sha256_hex(f"{CONTRACT_SCHEMA}|block|{local_id}|{revision}|{ordinal}")[:24]


def leaf_node_id(ref: LeafRef) -> str:
    payload = (
        f"{CONTRACT_SCHEMA}|leaf|{ref.local_id}|{ref.revision}|{ref.block_id}"
        f"|{ref.char_start}|{ref.char_end}"
    )
    return "nl-" + _sha256_hex(payload)[:24]


def contract_digest(contract: Mapping[str, Any]) -> str:
    """Digest a region's generation contract (prompt/model/tokenizer/budget)."""
    return _sha256_hex(f"{CONTRACT_SCHEMA}|contract|{_canonical_json(dict(contract))}")[:24]


def members_key(children: Sequence[ChildRef]) -> str:
    """Canonical member key: sorted unique ``child_id@child_revision``."""
    keys = {f"{child.child_id}@{child.child_revision}" for child in children}
    _require(bool(keys), "region requires at least one member")
    _require(len(keys) == len(children), "duplicate member in region children")
    return "|".join(sorted(keys))


def region_node_id(children: Sequence[ChildRef], contract: Mapping[str, Any]) -> str:
    payload = f"{CONTRACT_SCHEMA}|region|{members_key(children)}|{contract_digest(contract)}"
    return "nr-" + _sha256_hex(payload)[:24]


def node_fingerprint(record: NodeRecord) -> str:
    """Identity+content+member change detector for one node snapshot."""
    if record.kind == "leaf":
        assert record.leaf is not None
        payload = (
            f"{CONTRACT_SCHEMA}|fp|leaf|{record.leaf.local_id}|{record.leaf.revision}"
            f"|{record.leaf.block_id}|{record.leaf.char_start}|{record.leaf.char_end}"
            f"|{record.content_hash}"
        )
    else:
        payload = (
            f"{CONTRACT_SCHEMA}|fp|region|{members_key(record.children)}"
            f"|{contract_digest(record.contract)}|{record.content_hash}"
        )
    return _sha256_hex(payload)


@dataclass(frozen=True)
class DocumentRevision:
    """One normalized version of one source material."""

    local_id: str
    revision: int
    source_hash: str
    canonical_hash: str
    backend: str
    media_type: str
    parser_revision: str = ""
    state: str = "ready"

    def __post_init__(self) -> None:
        _require(bool(self.local_id) and self.local_id == self.local_id.strip(), "local_id")
        _require(self.revision >= 1, "revision must be >= 1")
        _require(bool(self.source_hash), "source_hash required")
        _require(bool(self.canonical_hash), "canonical_hash required")
        _require(self.media_type in {"pdf", "tex", "md"}, f"media_type {self.media_type!r}")
        _require(self.state in DOCUMENT_STATES, f"document state {self.state!r}")


@dataclass(frozen=True)
class ContentBlock:
    """A boundary-faithful slice of the canonical document text."""

    block_id: str
    local_id: str
    revision: int
    ordinal: int
    text: str
    text_hash: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    heading_path: tuple[str, ...] = ()
    anchor: str = ""
    kind: str = "paragraph"
    parser: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.char_start >= 0, "char_start must be >= 0")
        _require(self.char_end > self.char_start, "block must be non-empty")
        _require(
            self.char_end - self.char_start == len(self.text),
            "block text length must equal the declared char span",
        )
        _require(self.text_hash == _sha256_hex(self.text), "text_hash mismatch")
        _require(self.ordinal >= 0, "ordinal must be >= 0")
        for page in (self.page_start, self.page_end):
            _require(page is None or page >= 1, "PDF pages are 1-based")
        if self.page_start is not None and self.page_end is not None:
            _require(self.page_end >= self.page_start, "page_end must be >= page_start")
        for line in (self.line_start, self.line_end):
            _require(line is None or line >= 1, "lines are 1-based")
        if self.line_start is not None and self.line_end is not None:
            _require(self.line_end >= self.line_start, "line_end must be >= line_start")

    @property
    def text_len(self) -> int:
        return self.char_end - self.char_start

    def page_span(self) -> tuple[int, int] | None:
        if self.page_start is None or self.page_end is None:
            return None
        return (self.page_start, self.page_end)

    def line_span(self) -> tuple[int, int] | None:
        if self.line_start is None or self.line_end is None:
            return None
        return (self.line_start, self.line_end)


@dataclass(frozen=True)
class LeafRef:
    """A leaf node's reference into one canonical block."""

    local_id: str
    revision: int
    block_id: str
    char_start: int = 0
    char_end: int | None = None

    def __post_init__(self) -> None:
        _require(bool(self.local_id), "local_id")
        _require(self.revision >= 1, "revision must be >= 1")
        _require(bool(self.block_id), "block_id")
        _require(self.char_start >= 0, "char_start must be >= 0")
        _require(
            self.char_end is None or self.char_end > self.char_start,
            "leaf sub-range must be non-empty",
        )

    def resolved_char_end(self, block_len: int) -> int:
        end = block_len if self.char_end is None else self.char_end
        _require(end > self.char_start and end <= block_len, "leaf range exceeds block")
        return end

    def resolve(self, block_len: int) -> LeafRef:
        """Return a copy with the end fixed, for persistence."""
        return LeafRef(
            local_id=self.local_id,
            revision=self.revision,
            block_id=self.block_id,
            char_start=self.char_start,
            char_end=self.resolved_char_end(block_len),
        )


@dataclass(frozen=True)
class ChildRef:
    """One child membership edge of a region."""

    child_id: str
    child_revision: int
    ordinal: int = 0
    weight: float | None = None

    def __post_init__(self) -> None:
        _require(bool(self.child_id), "child_id")
        _require(self.child_revision >= 1, "child_revision must be >= 1")
        _require(self.ordinal >= 0, "ordinal must be >= 0")
        _require(
            self.weight is None or (0.0 < float(self.weight) <= 1.0),
            "membership weight must be in (0, 1]",
        )


def child_ref_to_json(child: ChildRef) -> dict[str, Any]:
    return {
        "child_id": child.child_id,
        "child_revision": child.child_revision,
        "ordinal": child.ordinal,
        "weight": child.weight,
    }


def child_ref_from_json(payload: Mapping[str, Any]) -> ChildRef:
    return ChildRef(
        child_id=str(payload["child_id"]),
        child_revision=int(payload["child_revision"]),
        ordinal=int(payload.get("ordinal", 0)),
        weight=None if payload.get("weight") is None else float(payload["weight"]),
    )


@dataclass(frozen=True)
class NodeRecord:
    """One node snapshot: leaf (source) or region (generated)."""

    node_id: str
    revision: int
    kind: str
    state: str
    layer: int
    content_hash: str
    fingerprint: str = ""
    title: str = ""
    summary: str = ""
    leaf: LeafRef | None = None
    children: tuple[ChildRef, ...] = ()
    contract: dict[str, Any] = field(default_factory=dict)
    origin: str = ""
    heading_path: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.kind in NODE_KINDS, f"node kind {self.kind!r}")
        _require(self.state in NODE_STATES, f"node state {self.state!r}")
        _require(self.revision >= 1, "revision must be >= 1")
        _require(self.layer >= 0, "layer must be >= 0")
        _require(bool(self.content_hash), "content_hash required")
        if self.kind == "leaf":
            _require(self.leaf is not None, "leaf node requires a leaf reference")
            _require(not self.children, "leaf node cannot have children")
            _require(not self.summary, "leaf node stores no generated summary")
            leaf = self.leaf
            if leaf is None:  # _require already rejected this; narrow for the checker
                raise ValueError("leaf node requires a leaf reference")
            expected = leaf_node_id(leaf)
            _require(self.node_id == expected, "leaf node_id does not match its reference")
        else:
            _require(self.leaf is None, "region node cannot reference a block")
            _require(bool(self.children), "region node requires children")
            _require(bool(self.summary.strip()), "region node requires a non-empty summary")
            _require(self.layer >= 1, "region layer must be >= 1")
            expected = region_node_id(self.children, self.contract)
            _require(self.node_id == expected, "region node_id does not match members/contract")
        expected_fp = node_fingerprint(self)
        if self.fingerprint:
            _require(self.fingerprint == expected_fp, "fingerprint does not match content")
        else:
            object.__setattr__(self, "fingerprint", expected_fp)

    def with_state(self, state: str) -> NodeRecord:
        return NodeRecord(
            node_id=self.node_id,
            revision=self.revision,
            kind=self.kind,
            state=state,
            layer=self.layer,
            content_hash=self.content_hash,
            fingerprint=self.fingerprint,
            title=self.title,
            summary=self.summary,
            leaf=self.leaf,
            children=self.children,
            contract=self.contract,
            origin=self.origin,
            heading_path=self.heading_path,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class ReadReceipt:
    """Proof that a tool actually read one exact source range (T38/T41)."""

    request_id: str
    tool: str
    node_id: str
    node_revision: int
    local_id: str
    block_id: str
    char_start: int
    char_end: int
    content_hash: str
    tokens: int
    truncated: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.request_id), "request_id")
        _require(self.tool in {"read", "read_scope", "expand"}, f"tool {self.tool!r}")
        _require(self.char_end > self.char_start, "read range must be non-empty")
        _require(self.tokens >= 0, "tokens must be >= 0")
        _require(bool(self.content_hash), "content_hash required")

    @property
    def span_key(self) -> tuple[str, str, int, int]:
        return (self.local_id, self.block_id, self.char_start, self.char_end)


@dataclass
class StageReport:
    """One stage's ledger entry for ingest/prepare/ask JSON output (T06)."""

    stage: str
    state: StageState | str
    counts: dict[str, int] = field(default_factory=dict)
    reused: int = 0
    created: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    messages: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        state = self.state if isinstance(self.state, StageState) else StageState(str(self.state))
        return {
            "stage": self.stage,
            "state": state.value,
            "counts": dict(self.counts),
            "reused": self.reused,
            "created": self.created,
            "failed": self.failed,
            "duration_ms": round(float(self.duration_ms), 3),
            "messages": list(self.messages),
            "details": dict(self.details),
        }
