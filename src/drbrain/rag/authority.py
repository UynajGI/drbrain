"""Source-authority conflict resolution for epistemic claims.

A real knowledge base contains conflicting statements about the same fact
(Doc A says the deadline is Aug 10, Doc B says Aug 15, an email says Aug 20).
When several candidate claims share one ``label`` but disagree, this module
resolves them with a deterministic precedence order — authority tier first,
then freshness (``valid_from``), then extraction confidence — instead of
leaving the choice to the LLM (which would average or guess).

The resolver never silently throws information away:

* ``stale`` marks a fact that *exists* but whose validity window has lapsed
  (``valid_to < now``). It is reported, not discarded, because a user may still
  ask "what was the value last year?".
* ``no_evidence`` is the *separate* situation where a label has no candidate
  claims at all. Absence of evidence is not evidence of absence: "there is a
  fact, it expired" (``stale``) and "we found nothing" (``no_evidence``) must
  not collapse into the same answer.

``resolve_claims`` returns one :class:`ResolvedClaim` per label found in the
input; an empty input returns an empty list, and the caller is responsible for
translating that into ``no_evidence`` for a queried label.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# ── Authority hierarchy ──
# Ordered high → low. Why these tiers: in an academic corpus the most binding
# sources are external, hard-to-fabricate records (a database of record, a
# signed document), then institutionally vetted text (approved policy,
# peer-reviewed paper body), then internal/unvetted prose (wiki, email, chat).
# Peer-reviewed ranks *below* official databases and policy because a single
# paper's claim can later be retracted or superseded, whereas a signed/policy
# record is a commitment of record. Gaps between tiers are deliberate so that a
# high tier always beats *any* number of lower tiers (authority is ordinal, not
# countable — two emails don't outweigh one contract).
AUTHORITY_RANK: dict[str, int] = {
    "official_db": 100,  # database of record (CrossRef, publisher, funding registry)
    "signed_contract": 90,  # signed / legally binding document
    "approved_policy": 80,  # institutionally approved policy or mandate
    "peer_reviewed": 70,  # peer-reviewed paper body (primary scholarly evidence)
    "internal_wiki": 50,  # internal team wiki (documented but unvetted)
    "email": 30,  # informal single-author correspondence
    "chat": 10,  # ephemeral, least vetted
    "": 0,  # unlabeled — treated as weakest, not as "unknown"
}


def authority_rank(authority: str) -> int:
    """Return the numeric rank of *authority* (unknown tiers fall back to 0).

    Unknown values rank equal to the unlabeled ``""`` tier rather than raising,
    so an unrecognized tier can never accidentally outrank a known one.
    """
    return AUTHORITY_RANK.get(authority, 0)


@dataclass
class ResolvedClaim:
    """A single fact resolved from one or more candidate claims.

    ``resolution`` is one of:

    * ``single`` — exactly one candidate, or several candidates that agree
      (corroboration merges into a single fact).
    * ``authoritative`` — several candidates, won by the single highest tier.
    * ``conflict`` — several candidates tied at the top tier with divergent
      values; not silently resolved, reported for upstream presentation.
    * ``stale`` — the fact exists but every candidate's validity window has
      lapsed; ``value`` is the best historical value.
    * ``no_evidence`` — reserved for the caller's "label has no candidates"
      case; ``resolve_claims`` returns ``[]`` instead, leaving the mapping to
      the caller so absence of evidence is kept distinct from a stale fact.

    For a ``conflict``, ``value`` carries the best-guess representative (the
    freshest, then most confident tied candidate) and ``reason`` enumerates the
    full divergence so the caller can present every side rather than one.
    """

    label: str
    value: str
    authority: str
    provenance: str
    confidence: float
    valid_from: int | None
    valid_to: int | None
    resolution: str  # "single" | "authoritative" | "conflict" | "stale" | "no_evidence"
    reason: str  # human-readable account of how the resolution was reached


def is_stale(claim: dict, now: int | None = None) -> bool:
    """True if *claim*'s validity window has lapsed (``valid_to < now``).

    A missing ``valid_to`` means "still valid" (open-ended), never stale. The
    comparison is strictly ``<`` so a claim expiring exactly at *now* is still
    current.
    """
    valid_to = claim.get("valid_to")
    if valid_to is None:
        return False
    ts = int(time.time()) if now is None else now
    return valid_to < ts


def resolve_claims(claims: list[dict], *, now: int | None = None) -> list[ResolvedClaim]:
    """Resolve multiple candidate claims into one :class:`ResolvedClaim` per label.

    Groups the input by ``label``, then within each group:

    1. Split lapsed claims (``valid_to < now``) from still-valid ones. Lapsed
       claims are excluded from the *current* value competition but reported —
       never dropped — so a historical query still has an answer.
    2. Among still-valid claims, rank by :func:`authority_rank` (highest wins),
       then freshness (larger ``valid_from``), then ``confidence``.
    3. If several claims tie at the highest tier with *different* values, mark
       ``resolution="conflict"`` and report the divergence instead of silently
       picking a side.

    Missing keys are tolerated (``authority``/``provenance`` default to ``""``,
    ``confidence`` to 0.0, ``valid_from``/``valid_to`` to ``None``).

    Returns one entry per label, in first-appearance order. Empty input returns
    an empty list — the caller maps that to ``no_evidence`` for the queried
    label rather than fabricating a result.
    """
    ts = int(time.time()) if now is None else now

    groups: dict[str, list[dict]] = {}
    for claim in claims:
        label = claim.get("label")
        if label is None:
            continue  # malformed claim with no label cannot be grouped
        groups.setdefault(label, []).append(claim)

    return [_resolve_group(label, group, ts) for label, group in groups.items()]


# ── Internal helpers ──


def _resolve_group(label: str, group: list[dict], now: int) -> ResolvedClaim:
    fresh = [c for c in group if not is_stale(c, now)]
    stale = [c for c in group if is_stale(c, now)]

    # Every candidate has lapsed: there is a fact, it just expired. Report the
    # best historical value rather than nothing — the caller decides whether a
    # past value answers the question or the fact needs re-verification.
    if not fresh:
        best = _best(stale)
        return _build(label, best, "stale", _stale_reason(stale, now))

    if len(fresh) == 1:
        note = f"; {len(stale)} lapsed candidate(s) excluded" if stale else ""
        return _build(label, fresh[0], "single", "single source for this fact" + note)

    # Multiple still-valid candidates: rank by authority tier.
    max_rank = max(authority_rank(c.get("authority", "")) for c in fresh)
    top = [c for c in fresh if authority_rank(c.get("authority", "")) == max_rank]

    if len(top) == 1:
        # A single source holds the top tier and wins outright; lower tiers are
        # overruled regardless of their freshness or confidence.
        chosen = top[0]
        reason = (
            f"highest authority ({chosen.get('authority') or 'unlabeled'}) beats "
            f"{len(fresh) - 1} lower-tier candidate(s)"
        )
        return _build(label, chosen, "authoritative", reason)

    # Several sources tie at the top tier. Distinguish agreement from genuine
    # disagreement: corroboration merges into one fact; a tie with different
    # values is a real conflict that must be surfaced, not resolved by flipping
    # a coin on the freshest one.
    values = {c.get("value") for c in top}
    if len(values) == 1:
        chosen = _best(top)
        reason = (
            f"{len(top)} sources at the same authority agree on the value "
            f"(corroborated); {len(fresh) - len(top)} lower-tier candidate(s) ignored"
        )
        return _build(label, chosen, "single", reason)

    chosen = _best(top)
    return _build(label, chosen, "conflict", _conflict_reason(top))


def _build(
    label: str,
    chosen: dict,
    resolution: str,
    reason: str,
) -> ResolvedClaim:
    return ResolvedClaim(
        label=label,
        value=chosen.get("value", ""),
        authority=chosen.get("authority", ""),
        provenance=chosen.get("provenance", ""),
        confidence=float(chosen.get("confidence") or 0.0),
        valid_from=chosen.get("valid_from"),
        valid_to=chosen.get("valid_to"),
        resolution=resolution,
        reason=reason,
    )


def _best(candidates: list[dict]) -> dict:
    """Pick the winning claim: authority rank, then freshest, then confidence."""
    return max(
        candidates,
        key=lambda c: (
            authority_rank(c.get("authority", "")),
            c.get("valid_from") if c.get("valid_from") is not None else -1,
            c.get("confidence") if c.get("confidence") is not None else 0.0,
        ),
    )


def _stale_reason(stale: list[dict], now: int) -> str:
    best = _best(stale)
    return (
        f"fact exists but lapsed: {len(stale)} candidate(s) expired "
        f"(valid_to={best.get('valid_to')}, now={now}); reported as historical value"
    )


def _conflict_reason(top: list[dict]) -> str:
    parts = []
    for c in top:
        parts.append(
            f"{c.get('value')!r} (authority={c.get('authority') or 'unlabeled'}, "
            f"provenance={c.get('provenance') or 'unknown'})"
        )
    return "equal-authority conflict: " + "; ".join(parts)
