"""Structured dialogue state for multi-turn RAG (Memory ≠ State).

The retrieval/synthesis layer keeps *memory* — facts, chunks, node embeddings
— durable across turns. This module keeps the *state* of a single
conversation explicit and structured: what the user is currently talking
about, decoupled from the raw chat history. Instead of making the LLM guess
the antecedent of "那负责人呢?" from the whole transcript, follow-ups resolve
against a small, per-turn-replaceable ``RAGState``.

State is deliberately small, versionable, and per-conversation; it is not a
memory model and should not accumulate facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class RAGState:
    """Current focus of a multi-turn RAG conversation.

    Every field is optional on purpose: the state fills in turn by turn, so an
    empty state is a valid starting point and partial dicts must round-trip.
    """

    entity_ids: list[str] = field(default_factory=list)
    # String, not an enum: the intent taxonomy is open and tenant-specific, so
    # a closed enumeration would churn on every new verb instead of evolving.
    intent: str = ""
    attribute: str = ""
    period: str = ""
    comparison_period: str = ""
    tenant_id: str = ""
    user_id: str = ""
    snapshot_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields; the two mutable ones are copied so callers
        can't mutate this state through the returned dict."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["entity_ids"] = list(data["entity_ids"])
        data["extra"] = dict(data["extra"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RAGState:
        """Rebuild from a possibly partial dict.

        Missing fields fall back to their defaults; unknown keys are absorbed
        into ``extra`` so forward-compatible callers don't silently drop data.
        """
        raw = dict(data or {})
        field_names = {f.name for f in fields(cls)}
        values: dict[str, Any] = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        extra_raw = raw.get("extra") or {}
        extra = dict(extra_raw) if isinstance(extra_raw, dict) else {}
        for key, val in raw.items():
            if key == "extra":
                continue
            if key in field_names:
                values[key] = val
            else:
                extra[key] = val
        values["extra"] = extra
        return cls(**values)

    def update(self, **kwargs: Any) -> RAGState:
        """Return a new state with ``kwargs`` merged in, leaving ``self``
        untouched.

        Immutable updates make each turn's state a value that can be kept for
        versioning/audit without aliasing hazards. Unknown keys land in
        ``extra``; an explicit ``extra=`` replaces the extension bag wholesale.
        """
        field_names = {f.name for f in fields(self)}
        values: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "entity_ids":
                v = list(v)
            elif f.name == "extra":
                v = dict(v)
            values[f.name] = v
        extra = values["extra"]
        for key, val in kwargs.items():
            if key == "extra":
                extra = dict(val)
            elif key in field_names:
                values[key] = val
            else:
                extra[key] = val
        values["extra"] = extra
        return type(self)(**values)

    def resolve_reference(self, prompt: str) -> str:
        """Rewrite an anaphoric follow-up ("那负责人呢?") with current context.

        This is a deterministic fallback for the common elliptical case; a
        production system should call an LLM to resolve full coreference
        against the dialogue history and fall back here only when that fails.
        Non-referential prompts are returned unchanged.
        """
        text = prompt.strip()
        if not self._is_reference(text):
            return prompt
        clause = self._context_clause()
        if not clause:
            return prompt
        return f"{clause}{text}"

    @staticmethod
    def _is_reference(text: str) -> bool:
        if not text:
            return False
        # "X呢?" / "X呢" is the canonical elliptical follow-up ("what about X?").
        if text.endswith(("呢?", "呢？", "呢")):
            return True
        bare = text.rstrip("?？").strip()
        return bare in {"它", "那", "这个", "那个", "其", "该", "他", "她", "它们", "他们"}

    def _context_clause(self) -> str:
        bits: list[str] = []
        if self.entity_ids:
            bits.append("实体=" + "、".join(self.entity_ids))
        if self.attribute:
            bits.append(f"属性={self.attribute}")
        if self.intent:
            bits.append(f"意图={self.intent}")
        if self.period:
            bits.append(f"时间窗={self.period}")
        if self.comparison_period:
            bits.append(f"对比窗={self.comparison_period}")
        if not bits:
            return ""
        return "【承接上文：" + "，".join(bits) + "】"

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.entity_ids:
            parts.append(f"entities={self.entity_ids}")
        if self.intent:
            parts.append(f"intent={self.intent!r}")
        if self.attribute:
            parts.append(f"attr={self.attribute!r}")
        if self.period:
            parts.append(f"period={self.period!r}")
        if self.comparison_period:
            parts.append(f"vs={self.comparison_period!r}")
        if self.tenant_id:
            parts.append(f"tenant={self.tenant_id!r}")
        if self.user_id:
            parts.append(f"user={self.user_id!r}")
        if self.snapshot_id:
            parts.append(f"snapshot={self.snapshot_id!r}")
        if self.extra:
            parts.append(f"extra={self.extra}")
        return f"RAGState({', '.join(parts)})"
