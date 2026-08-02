"""Research-direction recommendation from the concept graph.

Given a researcher (identified by name match on ``papers.authors``), builds their
concept profile ``C_own`` and suggests novel concept combinations:

* ``S_own×other`` — own concepts paired with new, semantically-related concepts;
* ``S_(manyown)×other`` — concepts connecting strongly to many own concepts;
* optional LLM curation that elaborates promising combinations.

Mirrors the report-generation approach of Marwitz et al. (2026, NMI).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from drbrain.concept_graph.builder import normalize_concept
from drbrain.storage.database import Database


def own_concepts(db: Database, author: str) -> set[str]:
    """Return the researcher's concept profile ``C_own``.

    ``C_own`` = (concepts appearing in the author's papers) ∩ (known concept
    nodes). Concepts are gathered from ``paper_terms`` and ``concepts.label``.

    Args:
        db: Database handle.
        author: Author name (case-insensitive substring match on ``papers.authors``).

    Returns:
        A set of normalized concept labels.
    """
    rows = db.conn.execute(
        "SELECT local_id FROM papers WHERE LOWER(authors) LIKE ?", (f"%{author.lower()}%",)
    ).fetchall()
    paper_ids = [r[0] for r in rows]
    if not paper_ids:
        return set()

    known = {r[0] for r in db.conn.execute("SELECT label FROM concept_nodes").fetchall()}
    placeholders = ",".join("?" * len(paper_ids))
    terms = db.conn.execute(
        f"SELECT term FROM paper_terms WHERE local_id IN ({placeholders})", paper_ids
    ).fetchall()
    labels = db.conn.execute(
        f"SELECT label FROM concepts WHERE local_id IN ({placeholders})", paper_ids
    ).fetchall()

    candidates = {normalize_concept(t[0]) for t in terms} | {
        normalize_concept(lab[0]) for lab in labels
    }
    candidates.discard("")
    return candidates & known


def _embeddings_or_empty(db: Database) -> dict[str, np.ndarray]:
    from drbrain.concept_graph.embeddings import load_concept_embeddings

    return load_concept_embeddings(db)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _known_neighbors(db: Database, c_own: set[str]) -> set[str]:
    """Concepts already co-occurring with any own concept (to exclude as known)."""
    if not c_own:
        return set()
    placeholders = ",".join("?" * len(c_own))
    own_list = list(c_own)
    rows = db.conn.execute(
        f"SELECT src_label, dst_label FROM concept_cooccurrence "
        f"WHERE src_label IN ({placeholders}) OR dst_label IN ({placeholders})",
        own_list + own_list,
    ).fetchall()
    neighbors: set[str] = set()
    for src, dst in rows:
        neighbors.add(src)
        neighbors.add(dst)
    return neighbors


def recommend_combinations(
    db: Database,
    author: str,
    *,
    top_k: int = 25,
    sim_min: float = 0.15,
    sim_max: float = 0.95,
    max_hub_freq: int | None = None,
) -> dict:
    """Generate filtered concept-combination suggestions for a researcher.

    Scoring uses semantic similarity (mean cosine to ``C_own``) when embeddings
    are available; otherwise falls back to co-occurrence breadth. Heuristics
    exclude already-known combinations, generic hubs, and pairs that are too
    similar or too unrelated.

    Args:
        db: Database handle.
        author: Author name.
        top_k: Maximum suggestions per section.
        sim_min: Minimum similarity to keep (avoid unrelated).
        sim_max: Maximum similarity to keep (avoid near-duplicates).
        max_hub_freq: Exclude concepts with doc_freq above this (avoid generic).

    Returns:
        Dict with ``c_own``, ``own_x_other`` and ``many_own_x_other`` lists.
    """
    c_own = own_concepts(db, author)
    embeddings = _embeddings_or_empty(db)
    known = _known_neighbors(db, c_own)

    nodes = db.conn.execute("SELECT label, doc_freq FROM concept_nodes").fetchall()
    own_vecs = [embeddings[c] for c in c_own if c in embeddings]

    scored: list[tuple[str, float, int]] = []
    many_own: list[tuple[str, int]] = []
    for label, doc_freq in nodes:
        if label in c_own or label in known:
            continue
        if max_hub_freq is not None and doc_freq > max_hub_freq:
            continue

        if own_vecs and label in embeddings:
            sims = [_cosine(embeddings[label], ov) for ov in own_vecs]
            score = float(np.mean(sims))
            n_related = sum(1 for s in sims if s >= sim_min)
            if not (sim_min <= score <= sim_max):
                continue
            scored.append((label, score, n_related))
            many_own.append((label, n_related))
        else:
            # No embeddings: rank by document frequency as a weak proxy.
            scored.append((label, float(doc_freq), 0))

    scored.sort(key=lambda t: t[1], reverse=True)
    own_x_other = [{"concept": lab, "score": round(s, 4)} for lab, s, _ in scored[:top_k]]

    many_own.sort(key=lambda t: t[1], reverse=True)
    many_section = [{"concept": lab, "related_own_count": n} for lab, n in many_own if n >= 2][:top_k]

    logger.info(
        "[cg.recommend] author='{}' c_own={} suggestions={}", author, len(c_own), len(own_x_other)
    )
    return {"c_own": sorted(c_own), "own_x_other": own_x_other, "many_own_x_other": many_section}


def llm_curation(suggestions: list[dict], models: list[dict], *, top: int = 5) -> str:
    """Ask an LLM to select and elaborate the most promising combinations.

    Returns an empty string when no models are configured or the call fails, so
    recommendation degrades gracefully without an LLM.

    Args:
        suggestions: Candidate combinations (``{"concept", "score"}``).
        models: LLM model configs (see ``drbrain.extractor.llm_client``).
        top: Number of suggestions to present to the LLM.

    Returns:
        A markdown paragraph of curated research directions.
    """
    if not models or not suggestions:
        return ""
    import asyncio

    from drbrain.extractor.llm_client import acall_with_fallback

    top_items = suggestions[:top]
    listing = "\n".join(f"- {s['concept']} (score {s.get('score', 'n/a')})" for s in top_items)
    prompt = (
        "You are a research-ideation assistant. From these candidate concept "
        "combinations, select the most promising novel research directions and "
        "write a short paragraph for each explaining how the concepts could be "
        "combined and why the combination is promising.\n\nCandidates:\n" + listing
    )
    try:
        result = asyncio.run(acall_with_fallback(prompt=prompt, models=models))
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        logger.warning("[cg.recommend] LLM curation failed: {}", exc)
        return ""
    if isinstance(result, dict):
        return str(result.get("text", result))
    return str(result) if result else ""
