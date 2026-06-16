"""Transfer opportunity detection and history."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

from loguru import logger

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database

if TYPE_CHECKING:
    pass

from drbrain.graph.genealogy.landscape import _get_concepts_by_type
from drbrain.graph.genealogy.paradigm import (
    _format_provenance,
    _get_concept_provenance,
)


def find_transfer_opportunities(
    db: Database,
    graph: GraphEngine,
    source_paper_ids: list[str] | None = None,
    target_paper_ids: list[str] | None = None,
    min_confidence: float = 0.3,
) -> list[dict]:
    """Find Method->Problem transfer opportunities between domains.

    Explicit mode: user provides source (where Methods live) and target
    (where Problems live) paper ID lists.
    """
    if not source_paper_ids or not target_paper_ids:
        return []

    # Get Method concepts from source papers
    src_methods = _get_concepts_by_type(db, source_paper_ids, "Method")
    # Get Problem concepts from target papers
    tgt_problems = _get_concepts_by_type(db, target_paper_ids, "Problem")

    if not src_methods or not tgt_problems:
        return []

    transfers = _score_transfer_pairs(graph, src_methods, tgt_problems, min_confidence)
    for t in transfers:
        section, node_id, paper_id = _get_concept_provenance(db, t["source_method"], "Method")
        t["source_section"] = section
        t["source_node_id"] = node_id
        t["source_paper_id"] = paper_id
        t["source_provenance"] = _format_provenance(section, node_id, paper_id)
    return transfers


def find_transfer_opportunities_auto(
    db: Database,
    graph: GraphEngine,
    min_confidence: float = 0.3,
    cluster_similarity: float = 0.4,
) -> list[dict]:
    """Auto-detect domains by clustering concepts, then find transfer opportunities.

    Groups Method concepts by label similarity, same for Problems.
    Cross-pairs clusters and finds transfer candidates.
    """
    # Get all Methods and Problems
    all_methods = db.conn.execute(
        "SELECT DISTINCT label FROM concepts WHERE type = 'Method'"
    ).fetchall()
    all_problems = db.conn.execute(
        "SELECT DISTINCT label FROM concepts WHERE type = 'Problem'"
    ).fetchall()

    if not all_methods or not all_problems:
        return []

    # Cluster Methods by label similarity
    method_labels = [r[0] for r in all_methods]
    method_clusters = _cluster_by_similarity(method_labels, threshold=cluster_similarity)

    # Cluster Problems by label similarity
    problem_labels = [r[0] for r in all_problems]
    problem_clusters = _cluster_by_similarity(problem_labels, threshold=cluster_similarity)

    # Cross-pair clusters and find transfers
    results: list[dict] = []
    for m_cluster in method_clusters:
        for p_cluster in problem_clusters:
            # Skip if clusters overlap (same domain)
            overlap = set(m_cluster) & set(p_cluster)
            if overlap:
                continue

            transfers = _score_transfer_pairs(graph, m_cluster, p_cluster, min_confidence)
            for t in transfers:
                section, node_id, paper_id = _get_concept_provenance(
                    db, t["source_method"], "Method"
                )
                t["source_section"] = section
                t["source_node_id"] = node_id
                t["source_paper_id"] = paper_id
                t["source_provenance"] = _format_provenance(section, node_id, paper_id)
            results.extend(transfers)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:20]


def _score_transfer_pairs(
    graph: GraphEngine,
    source_methods: list[str],
    target_problems: list[str],
    min_confidence: float,
) -> list[dict]:
    """Score Method->Problem pairs using isomorphism + label similarity."""
    from drbrain.extractor.concept import _label_similarity
    from drbrain.extractor.isomorphism import _relation_signature

    _t0_genealogy = _time.monotonic()
    results: list[dict] = []

    for method in source_methods:
        if method not in graph.graph:
            continue
        method_sig = _relation_signature(graph, method)
        if not method_sig:
            continue

        for problem in target_problems:
            if problem not in graph.graph:
                continue
            prob_sig = _relation_signature(graph, problem)
            if not prob_sig:
                continue

            # Jaccard on signatures
            m_set = set(method_sig.items())
            p_set = set(prob_sig.items())
            union = m_set | p_set
            intersection = m_set & p_set
            jaccard = len(intersection) / len(union) if union else 0.0

            # Label similarity
            label_sim = _label_similarity(method, problem)

            # Combined: signature match + label match
            confidence = jaccard * 0.5 + label_sim * 0.5

            if confidence >= min_confidence:
                results.append(
                    {
                        "source_method": method,
                        "target_problem": problem,
                        "confidence": round(confidence, 4),
                    }
                )

    results.sort(key=lambda x: x["confidence"], reverse=True)
    _t_done = _time.monotonic() - _t0_genealogy  # noqa: F821
    logger.info("[genealogy] paradigm shifts done in %.1fs — %d shifts", _t_done, len(results))
    return results


def _cluster_by_similarity(labels: list[str], threshold: float = 0.4) -> list[list[str]]:
    """Simple single-pass clustering: each label joins the first matching cluster or starts a new one."""
    from drbrain.extractor.concept import _label_similarity

    clusters: list[list[str]] = []
    for label in labels:
        matched = False
        for cluster in clusters:
            if any(_label_similarity(label, member) >= threshold for member in cluster):
                cluster.append(label)
                matched = True
                break
        if not matched:
            clusters.append([label])
    return clusters


def find_transfer_history(db: Database, graph: GraphEngine) -> list[dict]:
    """Return all historical applies edges as transfer timeline, ordered by year."""
    edges = db.conn.execute(
        "SELECT src_id, dst_id, weight FROM edges WHERE relation = 'applies'"
    ).fetchall()

    if not edges:
        return []

    # Batch-preload concept label → (title, year) to eliminate N+1.
    # Previously: 3 SQL queries per edge (year + src_title + tgt_title).
    # Now: a single JOIN query builds the lookup before the loop.
    label_info: dict[str, tuple[str | None, int | None]] = {}
    info_rows = db.conn.execute(
        "SELECT c.label, "
        "(SELECT p.title FROM papers p WHERE p.local_id = c.local_id LIMIT 1) AS title, "
        "(SELECT p.year FROM papers p WHERE p.local_id = c.local_id AND p.year IS NOT NULL "
        "ORDER BY p.year LIMIT 1) AS year "
        "FROM concepts c GROUP BY c.label"
    ).fetchall()
    for label, title, year in info_rows:
        label_info[label] = (title, year)

    results: list[dict] = []
    for src_id, dst_id, conf in edges:
        src_title_src, _ = label_info.get(src_id, (None, None))
        tgt_title, year = label_info.get(dst_id, (None, None))
        src_title = (src_title_src or src_id)[:80]
        tgt_title = (tgt_title or dst_id)[:80]

        results.append(
            {
                "source_concept": src_id,
                "source_title": src_title,
                "target_concept": dst_id,
                "target_title": tgt_title,
                "relation": "applies",
                "confidence": conf,
                "year": year,
            }
        )

    results.sort(key=lambda x: (x["year"] or 0, -x["confidence"]))
    return results
