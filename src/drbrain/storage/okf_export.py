"""Open Knowledge Format (OKF v0.1) export.

Exports the DrBrain knowledge graph as a directory tree of markdown files
conforming to the OKF specification: YAML frontmatter + markdown body, with
cross-concept relationships expressed as standard markdown links.

Bundle layout::

    bundle/
    ├── index.md                      # root directory listing
    ├── concepts/
    │   ├── index.md
    │   ├── problem/
    │   │   └── <slug>.md
    │   └── method/
    │       └── <slug>.md
    └── papers/
        └── <local_id>.md

See docs/ or https://okf.spec for the full OKF spec. This module implements
the conformance requirements of OKF §9 (every .md has parseable YAML
frontmatter with a non-empty ``type`` field).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

# Relation vocabulary → human-readable prose verb. Unknown relations fall
# back to "related to". Keys cover the TBOX concept-level relations plus the
# closure-derived and citation-level ones.
_RELATION_PROSE: dict[str, str] = {
    "challenges": "challenges",
    "supports": "supports",
    "extends": "extends",
    "limits": "limits",
    "solves": "solves",
    "proposes": "proposes",
    "replaces": "replaces",
    "constrains": "constrains",
    "addresses": "addresses",
    "leaves_open": "leaves open",
    "points_to": "points to",
    "creates_debate": "creates debate with",
    "contains": "contains",
    "refines": "refines",
    "applies": "applies",
    "cites": "cites",
    "cited_by": "is cited by",
    "indirect_evolution": "evolves into",
    "shared_actor": "shares actor with",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(label: str, seen: set[str]) -> str:
    """Convert a concept label to a filesystem-safe, deduplicated slug.

    Non-``[a-z0-9]`` runs collapse to ``-``; truncated to 120 chars. If the
    resulting slug is already in ``seen``, append ``-2``, ``-3``, ... until
    unique. The final slug is added to ``seen`` before returning.
    """
    raw = (label or "").strip().lower()
    slug = _SLUG_RE.sub("-", raw).strip("-")[:120].strip("-") or "concept"
    if slug in seen:
        n = 2
        while f"{slug}-{n}" in seen:
            n += 1
        slug = f"{slug}-{n}"
    seen.add(slug)
    return slug


def _relation_to_prose(relation: str) -> str:
    """Map a DrBrain relation to OKF body prose."""
    return _RELATION_PROSE.get(relation, "related to")


def _build_label_index(db: Any) -> tuple[dict[str, dict], dict[str, str]]:
    """Load all concepts into a label→metadata map and a label→slug map.

    Returns (label_to_concept, label_to_slug) where label_to_concept has keys
    {type, label, first_seen, last_seen, updated_at} and label_to_slug maps
    each concept label to its deduplicated slug.
    """
    rows = db.conn.execute(
        "SELECT DISTINCT label, type, first_seen, last_seen, updated_at FROM concepts"
    ).fetchall()
    # Dedup by label (keep first occurrence); assign slugs.
    seen_slugs: set[str] = set()
    label_to_concept: dict[str, dict] = {}
    label_to_slug: dict[str, str] = {}
    for label, ctype, first_seen, last_seen, updated_at in rows:
        if label in label_to_concept:
            continue
        label_to_concept[label] = {
            "type": ctype,
            "label": label,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "updated_at": updated_at,
        }
        label_to_slug[label] = _slugify(label, seen_slugs)
    return label_to_concept, label_to_slug


def _concept_path(label: str, label_to_concept: dict, label_to_slug: dict) -> str:
    """Return the OKF bundle-relative path for a concept (no .md suffix).

    Returns the empty string if the label is unknown (broken link target —
    OKF §5.3 tolerates these).
    """
    if label not in label_to_concept:
        return ""
    ctype = label_to_concept[label]["type"].lower()
    slug = label_to_slug[label]
    return f"concepts/{ctype}/{slug}"


def _render_concept_md(
    concept: dict,
    edges_out: list[tuple[str, str]],
    edges_in: list[tuple[str, str]],
    arguments: list[dict],
    label_to_concept: dict,
    label_to_slug: dict,
) -> str:
    """Render a single concept as an OKF markdown document."""
    label = concept["label"]
    ctype = concept["type"]

    # Frontmatter
    ts = concept.get("updated_at") or (
        str(concept["last_seen"]) if concept.get("last_seen") else None
    )
    tags = sorted(
        {ctype}
        | {rel for _, rel in edges_out}
        | {rel for _, rel in edges_in}
    )
    fm_lines = [
        "---",
        f"type: {ctype}",
        f"title: {label!r}",
    ]
    description = (arguments[0].get("mechanism") if arguments else "") or label
    fm_lines.append(f"description: {description[:200]!r}")
    if tags:
        fm_lines.append(f"tags: [{', '.join(tags)}]")
    if ts:
        fm_lines.append(f"timestamp: {ts}")
    fm_lines.append("---\n")

    # Body
    body = [f"# {label}\n"]
    body.append(f"A **{ctype}** concept in the knowledge graph.\n")

    # Relationships
    rel_lines: list[str] = []
    for target, rel in edges_out:
        path = _concept_path(target, label_to_concept, label_to_slug)
        prose = _relation_to_prose(rel)
        if path:
            rel_lines.append(f"- **{prose}** [{target}](/{path}.md)")
        else:
            rel_lines.append(f"- **{prose}** {target}")  # broken link tolerated
    for source, rel in edges_in:
        path = _concept_path(source, label_to_concept, label_to_slug)
        prose = _relation_to_prose(rel)
        # Incoming edge phrasing: "<source> <prose> this"
        if path:
            rel_lines.append(f"- [{source}](/{path}.md) **{prose}** this concept")
        else:
            rel_lines.append(f"- {source} **{prose}** this concept")
    if rel_lines:
        body.append("## Relationships\n")
        body.extend(rel_lines)
        body.append("")

    # Arguments
    if arguments:
        body.append("## Arguments\n")
        for arg in arguments:
            claim_type = arg.get("claim_type", "proposes")
            claim = arg.get("claim", "")
            conf = arg.get("confidence")
            ev = arg.get("evidence_type")
            mech = arg.get("mechanism")
            conf_str = f" (confidence {conf})" if conf is not None else ""
            block = f"> **{claim_type}**{conf_str}: {claim}"
            if ev:
                block += f"\n>\n> evidence: {ev}"
            if mech:
                block += f"\n>\n> mechanism: {mech}"
            body.append(block)
            body.append("")

    return "\n".join(fm_lines + body).rstrip() + "\n"


def _render_paper_md(paper: dict, concept_labels: list[str]) -> str:
    """Render a paper as an OKF markdown document."""
    fm = ["---", "type: Paper", f"title: {paper.get('title', '')!r}"]
    if paper.get("year"):
        fm.append(f"description: {paper.get('title', '')} ({paper['year']})")
    if paper.get("doi"):
        fm.append(f"resource: https://doi.org/{paper['doi']}")
    tags = []
    if paper.get("doi"):
        tags.append("doi")
    if paper.get("arxiv"):
        tags.append("arxiv")
    if tags:
        fm.append(f"tags: [{', '.join(tags)}]")
    if paper.get("updated_at"):
        fm.append(f"timestamp: {paper['updated_at']}")
    fm.append("---\n")

    body = [f"# {paper.get('title', paper['local_id'])}\n"]
    if paper.get("abstract"):
        body.append("## Abstract\n")
        body.append(paper["abstract"] + "\n")
    meta_bits = []
    for k in ("journal", "authors", "volume", "pages"):
        v = paper.get(k)
        if v:
            meta_bits.append(f"**{k}**: {v}")
    if meta_bits:
        body.append("## Metadata\n")
        body.append(" · ".join(meta_bits) + "\n")
    if concept_labels:
        body.append("## Concepts\n")
        for lbl in sorted(set(concept_labels)):
            body.append(f"- {lbl}")
        body.append("")
    return "\n".join(fm + body).rstrip() + "\n"


def export_okf(
    graph: Any,
    db: Any,
    out_dir: str | Path,
    *,
    paper_ids: list[str] | None = None,
) -> dict:
    """Export the knowledge graph as an OKF v0.1 bundle at ``out_dir``.

    Args:
        graph: GraphEngine with edges loaded (used for cross-link discovery).
        db: Open Database connection.
        out_dir: Destination directory (created if missing; overwritten if present).
        paper_ids: Optional allowlist of paper local_ids. When set, only
            concepts/edges/arguments from these papers are exported.

    Returns:
        Stats dict: {"concepts": N, "papers": M, "edges": K, "arguments": A}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load concepts + build slug maps ---
    label_to_concept, label_to_slug = _build_label_index(db)

    # Optional: restrict to concepts belonging to the given papers.
    allowed_labels: set[str] | None = None
    if paper_ids is not None:
        ph = ",".join("?" * len(paper_ids))
        rows = db.conn.execute(
            f"SELECT DISTINCT label FROM concepts WHERE local_id IN ({ph})", paper_ids
        ).fetchall()
        allowed_labels = {r[0] for r in rows}

    # --- Gather edges per concept from the in-memory graph ---
    # edges_out[label] = [(target_label, relation), ...]
    # edges_in[label]  = [(source_label, relation), ...]
    edges_out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    edges_in: dict[str, list[tuple[str, str]]] = defaultdict(list)
    edge_count = 0
    for u, v, data in graph.graph.edges(data=True):
        rel = data.get("relation", "related_to")
        edges_out[u].append((v, rel))
        edges_in[v].append((u, rel))
        edge_count += 1

    # --- Gather arguments per target_label ---
    args_rows = db.conn.execute(
        "SELECT target_label, claim, claim_type, evidence_type, mechanism, confidence "
        "FROM arguments"
    ).fetchall()
    args_by_target: dict[str, list[dict]] = defaultdict(list)
    arg_count = 0
    for target, claim, ct, ev, mech, conf in args_rows:
        args_by_target[target].append(
            {
                "claim": claim,
                "claim_type": ct,
                "evidence_type": ev,
                "mechanism": mech,
                "confidence": conf,
            }
        )
        arg_count += 1

    # --- Write concept files ---
    concepts_dir = out / "concepts"
    concepts_written = 0
    index_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)  # type -> [(label, path)]
    for label, concept in label_to_concept.items():
        if allowed_labels is not None and label not in allowed_labels:
            continue
        ctype_lower = concept["type"].lower()
        slug = label_to_slug[label]
        type_dir = concepts_dir / ctype_lower
        type_dir.mkdir(parents=True, exist_ok=True)
        md = _render_concept_md(
            concept,
            edges_out.get(label, []),
            edges_in.get(label, []),
            args_by_target.get(label, []),
            label_to_concept,
            label_to_slug,
        )
        (type_dir / f"{slug}.md").write_text(md, encoding="utf-8")
        index_by_type[ctype_lower].append((label, f"concepts/{ctype_lower}/{slug}"))
        concepts_written += 1

    # --- Write paper files ---
    papers_dir = out / "papers"
    papers_written = 0
    papers = db.get_all_papers()
    if paper_ids is not None:
        id_set = set(paper_ids)
        papers = [p for p in papers if p["local_id"] in id_set]
    for paper in papers:
        # Concepts for this paper
        crows = db.conn.execute(
            "SELECT DISTINCT label FROM concepts WHERE local_id = ?", (paper["local_id"],)
        ).fetchall()
        clabels = [r[0] for r in crows]
        md = _render_paper_md(paper, clabels)
        papers_dir.mkdir(parents=True, exist_ok=True)
        (papers_dir / f"{paper['local_id']}.md").write_text(md, encoding="utf-8")
        papers_written += 1

    # --- Root index.md ---
    index_lines = ["# Knowledge Graph Bundle\n"]
    index_lines.append(
        "Exported from DrBrain. Each concept is a markdown file; relationships "
        "are expressed as markdown links.\n"
    )
    index_lines.append(f"- **Concepts**: {concepts_written}")
    index_lines.append(f"- **Papers**: {papers_written}")
    index_lines.append(f"- **Edges**: {edge_count}")
    index_lines.append(f"- **Arguments**: {arg_count}\n")
    if index_by_type:
        index_lines.append("## Concepts by Type\n")
        for ctype in sorted(index_by_type):
            items = index_by_type[ctype]
            index_lines.append(f"\n### {ctype.title()} ({len(items)})\n")
            for label, path in sorted(items)[:200]:  # cap to keep index readable
                index_lines.append(f"- [{label}](/{path}.md)")
            if len(items) > 200:
                index_lines.append(f"- ... and {len(items) - 200} more")
        index_lines.append("")
    if papers_written:
        index_lines.append("## Papers\n")
        for paper in papers[:200]:
            index_lines.append(f"- [{paper['title']}](/papers/{paper['local_id']}.md)")
        if len(papers) > 200:
            index_lines.append(f"- ... and {len(papers) - 200} more")
    (out / "index.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")

    stats = {
        "concepts": concepts_written,
        "papers": papers_written,
        "edges": edge_count,
        "arguments": arg_count,
    }
    logger.info(
        "[export:okf] wrote {} concepts, {} papers, {} edges, {} arguments → {}",
        concepts_written,
        papers_written,
        edge_count,
        arg_count,
        out,
    )
    return stats
