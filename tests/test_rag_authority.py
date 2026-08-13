"""Tests for source-authority conflict resolution (src/drbrain/rag/authority.py)."""

from __future__ import annotations

from drbrain.rag.authority import (
    authority_rank,
    is_stale,
    resolve_claims,
)


def _claim(
    label: str,
    value: str,
    *,
    authority: str = "",
    provenance: str = "SOURCE",
    confidence: float = 0.9,
    valid_from: int | None = None,
    valid_to: int | None = None,
) -> dict:
    return {
        "label": label,
        "value": value,
        "authority": authority,
        "provenance": provenance,
        "confidence": confidence,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


# ── 1. authority_rank ──


def test_authority_rank_known():
    assert authority_rank("official_db") == 100
    assert authority_rank("signed_contract") == 90
    assert authority_rank("chat") == 10
    assert authority_rank("") == 0


def test_authority_rank_unknown_falls_back_to_zero():
    assert authority_rank("blog_post") == 0
    assert authority_rank("made_up_tier") == 0


# ── 2. single candidate → single ──


def test_single_candidate_is_single():
    claims = [_claim("deadline", "Aug 10", authority="email")]
    out = resolve_claims(claims, now=1_000_000)

    assert len(out) == 1
    assert out[0].resolution == "single"
    assert out[0].value == "Aug 10"
    assert out[0].authority == "email"


# ── 3. multiple candidates, different authority → authoritative ──


def test_highest_authority_wins_regardless_of_freshness_or_confidence():
    claims = [
        _claim("deadline", "Aug 10", authority="chat", confidence=0.99, valid_from=100),
        _claim("deadline", "Aug 15", authority="email", confidence=0.5, valid_from=50),
        _claim("deadline", "Aug 20", authority="official_db", confidence=0.4, valid_from=10),
    ]
    out = resolve_claims(claims, now=1_000_000)

    assert len(out) == 1
    r = out[0]
    assert r.resolution == "authoritative"
    # Highest authority wins even though it is least fresh and least confident.
    assert r.value == "Aug 20"
    assert r.authority == "official_db"


# ── 4. equal authority tie, different values → conflict ──


def test_equal_authority_tie_different_values_is_conflict():
    claims = [
        _claim("deadline", "Aug 10", authority="email", provenance="email-a"),
        _claim("deadline", "Aug 15", authority="email", provenance="email-b"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.resolution == "conflict"
    # Divergence is reported, not silently dropped to one side.
    assert "Aug 10" in r.reason
    assert "Aug 15" in r.reason


# ── 5. stale claim → stale (not dropped) ──


def test_stale_claim_is_reported_not_dropped():
    now = 1_000_000
    claims = [
        _claim("grant", "$10k", authority="official_db", valid_from=100, valid_to=999),
    ]
    out = resolve_claims(claims, now=now)

    r = out[0]
    assert r.resolution == "stale"
    assert r.value == "$10k"  # historical value preserved for "last year" queries
    assert "lapsed" in r.reason


def test_is_stale_helper():
    assert is_stale({"valid_to": 100}, now=200)
    assert not is_stale({"valid_to": 200}, now=200)  # strict `<`
    assert not is_stale({"valid_to": None}, now=200)  # open-ended = still valid
    assert not is_stale({}, now=200)  # no valid_to key


# ── 6. empty input → empty list ──


def test_empty_input_returns_empty_list():
    assert resolve_claims([], now=1_000_000) == []


# ── 7. equal authority, same value → merged single ──


def test_equal_authority_same_value_merges_to_single():
    claims = [
        _claim("method", "X", authority="peer_reviewed", provenance="paper-a"),
        _claim("method", "X", authority="peer_reviewed", provenance="paper-b"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.resolution == "single"
    assert r.value == "X"


# ── metadata propagation + tie-breaking ─────────────────────────────────────


def test_confidence_breaks_tie_within_same_authority_and_value():
    # Two corroborating claims (same authority, same value) differ only in
    # confidence: the higher-confidence claim is the representative and its
    # confidence is carried into the resolved claim.
    claims = [
        _claim("method", "X", authority="peer_reviewed", confidence=0.5, provenance="paper-a"),
        _claim("method", "X", authority="peer_reviewed", confidence=0.95, provenance="paper-b"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.value == "X"
    assert r.confidence == 0.95  # higher confidence wins the tie


def test_provenance_propagates_on_authoritative_resolution():
    # The single top-tier claim wins; its provenance is preserved.
    claims = [
        _claim("deadline", "Aug 10", authority="chat", provenance="chat-log"),
        _claim("deadline", "Aug 20", authority="official_db", provenance="registry"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.resolution == "authoritative"
    assert r.value == "Aug 20"
    assert r.provenance == "registry"


def test_provenance_propagates_on_conflict_resolution():
    # Among tied conflicting claims the freshest/most-confident is chosen as
    # the representative; its provenance is the one surfaced.
    claims = [
        _claim("deadline", "Aug 10", authority="email", provenance="email-a", confidence=0.6),
        _claim("deadline", "Aug 15", authority="email", provenance="email-b", confidence=0.9),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.resolution == "conflict"
    assert r.provenance == "email-b"


def test_conflict_reason_enumerates_values_in_input_order():
    # The conflict reason is deterministic: it lists each divergent value in
    # input order, each with its authority and provenance.
    claims = [
        _claim("deadline", "Aug 10", authority="email", provenance="email-a"),
        _claim("deadline", "Aug 15", authority="email", provenance="email-b"),
        _claim("deadline", "Aug 20", authority="email", provenance="email-c"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    r = out[0]
    assert r.resolution == "conflict"
    assert r.reason.startswith("equal-authority conflict: ")
    # values appear in deterministic (input) order
    assert r.reason.index("'Aug 10'") < r.reason.index("'Aug 15'") < r.reason.index("'Aug 20'")
    # each value is annotated with its authority and provenance
    assert "authority=email" in r.reason
    assert "provenance=email-a" in r.reason
    assert "provenance=email-b" in r.reason
    assert "provenance=email-c" in r.reason


# ── extras: stale excluded when fresh exists; multiple labels ──


def test_stale_excluded_when_fresh_candidate_exists():
    now = 1_000_000
    claims = [
        _claim("deadline", "Aug 10", authority="email", valid_to=500),
        _claim("deadline", "Aug 15", authority="official_db", valid_from=900),
    ]
    out = resolve_claims(claims, now=now)

    r = out[0]
    assert r.resolution == "single"
    assert r.value == "Aug 15"
    assert "lapsed" in r.reason  # the stale candidate was noted, not dropped


def test_multiple_labels_resolved_independently():
    claims = [
        _claim("deadline", "Aug 10", authority="email"),
        _claim("venue", "Hall A", authority="official_db"),
    ]
    out = resolve_claims(claims, now=1_000_000)

    by_label = {r.label: r for r in out}
    assert set(by_label) == {"deadline", "venue"}
    assert by_label["deadline"].resolution == "single"
    assert by_label["venue"].resolution == "single"
