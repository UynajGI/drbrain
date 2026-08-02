"""Concept co-occurrence graph builder.

Turns per-paper concept sets into a timestamped co-occurrence multigraph (each
paper contributes a clique over its concepts, edges stamped with publication
year), then filters/aggregates nodes by document frequency and word count —
mirroring the concept-graph construction of Marwitz et al. (2026, NMI).

Concept sources (no-fulltext-first):
  * ``"terms"``    — source-provided keywords/topics (``paper_terms``), zero LLM
  * ``"concepts"`` — labels from the existing typed KG (``concepts.label``)
  * ``"abstract"`` — lightweight LLM extraction from title+abstract (optional)
"""

from __future__ import annotations

import re
from itertools import combinations

from loguru import logger

from drbrain.storage.database import Database

_FILL_WORDS = {"of", "the", "a", "an", "and", "or", "for", "in", "on", "to", "with"}
_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_concept(text: str) -> str:
    """Normalize a concept label (lowercase, de-fill, singularize, tidy).

    Implements the light linguistic normalization illustrated in the paper's
    Table 1: lowercasing, removal of the fill word ``of``, plural→singular, and
    whitespace/punctuation tidy-up. Returns ``""`` for labels that collapse to
    nothing.

    Args:
        text: Raw concept string.

    Returns:
        The normalized label (may be empty).
    """
    if not text:
        return ""
    label = text.lower().strip()
    label = _WS_RE.sub(" ", label)
    tokens = [t for t in label.split(" ") if t and t not in _FILL_WORDS]
    tokens = [_singularize(t) for t in tokens]
    label = " ".join(tokens)
    label = _EDGE_PUNCT_RE.sub("", label).strip()
    return _WS_RE.sub(" ", label)


def _singularize(token: str) -> str:
    """Apply a conservative plural→singular heuristic to a single token."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def concepts_for_paper(db: Database, local_id: str, source: str = "terms") -> list[str]:
    """Return the normalized, deduplicated concept labels for a paper.

    Args:
        db: Database handle.
        local_id: Paper local_id.
        source: One of ``"terms"`` / ``"concepts"`` / ``"abstract"``.

    Returns:
        A list of unique normalized concept labels.
    """
    raw: list[str] = []
    if source == "terms":
        raw = [term for term, _kind in db.get_paper_terms(local_id)]
    elif source == "concepts":
        raw = [
            r[0]
            for r in db.conn.execute(
                "SELECT label FROM concepts WHERE local_id = ?", (local_id,)
            ).fetchall()
        ]
    elif source == "abstract":
        raw = _concepts_from_abstract(db, local_id)
    else:
        raise ValueError(f"Unknown concept source '{source}'")

    seen: set[str] = set()
    out: list[str] = []
    for label in raw:
        norm = normalize_concept(label)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _concepts_from_abstract(db: Database, local_id: str) -> list[str]:
    """Extract concept labels from a paper's title+abstract via the LLM chain."""
    import asyncio

    from drbrain.config import load_config
    from drbrain.extractor.concept.types import extract_concepts

    row = db.conn.execute(
        "SELECT title, abstract FROM papers WHERE local_id = ?", (local_id,)
    ).fetchone()
    if not row:
        return []
    title, abstract = row
    text = f"{title}\n\n{abstract}".strip()
    if not text:
        return []
    models = load_config().llm.models
    result = asyncio.run(extract_concepts(text, models))
    if result is None:
        return []
    labels: list[str] = []
    for category in ("problems", "methods", "conclusions", "debates", "gaps", "actors"):
        for item in getattr(result, category, []):
            label = item.get("label", "")
            if label:
                labels.append(label)
    return labels


def build_cliques(
    db: Database,
    *,
    source: str = "terms",
    paper_ids: list[str] | None = None,
    commit_every: int = 200,
    rebuild: bool = True,
) -> int:
    """Build co-occurrence clique edges for each paper's concept set.

    For every paper, all unordered concept pairs become co-occurrence edges
    timestamped with the paper's publication year.

    Args:
        db: Database handle.
        source: Concept source (see :func:`concepts_for_paper`).
        paper_ids: Restrict to these papers (default: all with a year).
        commit_every: Commit cadence.
        rebuild: When doing a full build (``paper_ids is None``), clear existing
            co-occurrence edges first so reruns are idempotent (weights do not
            accumulate). Ignored for incremental ``paper_ids`` builds.

    Returns:
        The number of edge upserts performed.
    """
    if paper_ids is None:
        if rebuild:
            db.conn.execute("DELETE FROM concept_cooccurrence")
        rows = db.conn.execute("SELECT local_id, year FROM papers").fetchall()
    else:
        placeholders = ",".join("?" * len(paper_ids))
        rows = db.conn.execute(
            f"SELECT local_id, year FROM papers WHERE local_id IN ({placeholders})",
            paper_ids,
        ).fetchall()

    edge_count = 0
    processed = 0
    for local_id, year in rows:
        concepts = concepts_for_paper(db, local_id, source=source)
        if len(concepts) < 2:
            continue
        for a, b in combinations(sorted(concepts), 2):
            db.insert_cooccurrence(a, b, year, local_id)
            edge_count += 1
        processed += 1
        if processed % commit_every == 0:
            db.conn.commit()

    db.conn.commit()
    logger.info("[cg.build] cliques: {} papers -> {} edges", processed, edge_count)
    return edge_count


def apply_filter(db: Database, *, min_freq: int = 3, min_words: int = 2) -> dict:
    """Aggregate node statistics and populate ``concept_nodes`` for kept concepts.

    A concept is kept when its document frequency (distinct papers) is at least
    ``min_freq`` and its label has at least ``min_words`` words. Kept concepts are
    upserted into ``concept_nodes`` with ``doc_freq`` / ``first_year`` /
    ``last_year``. Co-occurrence edges are left intact; downstream stages join
    against ``concept_nodes`` to restrict to the filtered node set.

    Args:
        db: Database handle.
        min_freq: Minimum document frequency to keep a concept.
        min_words: Minimum number of words in the label.

    Returns:
        Stats dict with ``total_concepts`` and ``kept`` counts.
    """
    # Accurate doc_freq: count distinct papers each concept appears in, across
    # BOTH endpoint roles. A UNION ALL of (label, paper_id) from src and dst sides
    # followed by COUNT(DISTINCT paper_id) yields the true union of paper sets
    # (a plain max() of per-role counts would undercount).
    freq_rows = db.conn.execute(
        "SELECT label, COUNT(DISTINCT paper_id) FROM ("
        "  SELECT src_label AS label, paper_id FROM concept_cooccurrence"
        "  UNION ALL"
        "  SELECT dst_label AS label, paper_id FROM concept_cooccurrence"
        ") GROUP BY label"
    ).fetchall()
    doc_freq: dict[str, int] = {label: cnt for label, cnt in freq_rows}

    year_rows = db.conn.execute(
        "SELECT src_label, MIN(year), MAX(year) FROM concept_cooccurrence GROUP BY src_label"
    ).fetchall()
    dst_year = db.conn.execute(
        "SELECT dst_label, MIN(year), MAX(year) FROM concept_cooccurrence GROUP BY dst_label"
    ).fetchall()
    first_year: dict[str, int] = {}
    last_year: dict[str, int] = {}
    for label, mn, mx in list(year_rows) + list(dst_year):
        if mn is not None:
            first_year[label] = min(first_year.get(label, mn), mn)
        if mx is not None:
            last_year[label] = max(last_year.get(label, mx), mx)

    # Recompute the filtered node set from scratch so repeated `cg build` runs are
    # idempotent (stale nodes from a previous run are dropped).
    db.conn.execute("DELETE FROM concept_nodes")
    kept = 0
    for label, freq in doc_freq.items():
        word_count = len(label.split())
        if freq >= min_freq and word_count >= min_words:
            db.upsert_concept_node(
                label,
                doc_freq=freq,
                word_count=word_count,
                first_year=first_year.get(label),
                last_year=last_year.get(label),
            )
            kept += 1
    db.conn.commit()
    logger.info("[cg.build] filter: {} concepts -> {} kept", len(doc_freq), kept)
    return {"total_concepts": len(doc_freq), "kept": kept}
