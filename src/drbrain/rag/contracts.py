"""Backend-independent retrieval requests, diagnostics, and scope checks.

No model or workflow dependency belongs here. Legacy APIs may return
``RetrievalRows`` (a JSON-compatible list); new callers can inspect its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from drbrain.rag.status import RetrievalError, RetrievalStatus, classify_failure

CONTENT_FILTERS = frozenset({"paper_ids", "categories"})


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for key, value in (filters or {}).items():
        if key not in CONTENT_FILTERS:
            raise ValueError(f"unsupported content filter: {key}")
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise ValueError(f"filter {key} requires a string or list of nonempty strings")
        result[key] = list(dict.fromkeys(item.strip() for item in values))
    return result


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    generation: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    acl_filter: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if self.generation is not None and not self.generation.strip():
            raise ValueError("generation must be nonempty when supplied")
        object.__setattr__(self, "filters", normalize_filters(self.filters))
        object.__setattr__(self, "acl_filter", dict(self.acl_filter))


@dataclass
class LegResult:
    source: str
    status: str
    count: int = 0
    duration_ms: float = 0.0
    reason: str = ""


@dataclass
class RetrievalResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    status: str = "empty"
    generation: str | None = None
    legs: list[LegResult] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)


class RetrievalRows(list):
    """List compatibility without losing diagnostics on empty/degraded results."""

    def __init__(self, result: RetrievalResult):
        super().__init__(result.records)
        self.result = result


def finish_retrieval(
    records: list[dict[str, Any]],
    *,
    generation: str | None,
    legs: list[LegResult],
    capabilities: dict[str, Any] | None = None,
) -> RetrievalRows:
    failed = [leg for leg in legs if leg.status not in {"ok", "empty"}]
    if failed and len(failed) == len(legs):
        raise RetrievalError(
            "all retrieval legs unavailable",
            failures=[(leg.source, RetrievalStatus(leg.reason)) for leg in failed],
        )
    status = "degraded" if failed else "ok" if records else "empty"
    return RetrievalRows(RetrievalResult(records, status, generation, legs, capabilities or {}))


def failure_leg(source: str, exc: Exception, duration_ms: float = 0.0) -> LegResult:
    return LegResult(source, "unavailable", duration_ms=duration_ms, reason=classify_failure(exc))


def matches_scope(
    metadata: dict[str, Any],
    filters: dict[str, list[str]],
    acl_filter: dict[str, str] | None = None,
) -> bool:
    """Missing constrained metadata is denied, including an empty allow-list."""
    if "paper_ids" in filters and metadata.get("paper_id") not in filters["paper_ids"]:
        return False
    if "categories" in filters:
        categories = metadata.get("categories", [])
        if isinstance(categories, str):
            categories = categories.split()
        if not any(
            str(category).lower() == expected.lower()
            or str(category).lower().startswith(expected.lower() + ".")
            for category in categories or []
            for expected in filters["categories"]
        ):
            return False
    return all(
        key in metadata and (expected == "*" or metadata[key] == expected)
        for key, expected in (acl_filter or {}).items()
    )
