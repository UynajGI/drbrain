"""One-shot literature survey report generator (markdown).

``generate_survey`` produces a three-section markdown 综述 that reuses existing
knowledge-graph assets instead of re-deriving them:

  1. Gap 清单 — research gaps / debates / seeds (``detect_research_seeds``,
     ``concepts`` type ``Gap``/``Debate``) plus genealogy frontier signals
     (``detect_paradigm_shifts``).
  2. 文献交叉引用 — who-cites-whom (``paper_citations``) + shared references,
     with per-paper citation counts via ``citation_graph``.
  3. 证据链 — every Gap/结论 traced back to its source (concept provenance /
     authority / validity window + originating paper title/year + section).

Deterministic and LLM-free: everything comes from the SQLite graph, so the
command works offline and is trivially testable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from drbrain.storage.database import Database

# ── Stable section headings (referenced by tests and the CLI) ──────────────
H_GAPS = "## 一、Gap 清单"
H_CROSS = "## 二、文献交叉引用"
H_EVIDENCE = "## 三、证据链"


def _norm(s: object) -> str:
    return (str(s or "")).strip().lower()


def _matches(topic: str | None, *fields: object) -> bool:
    t = _norm(topic)
    if not t:
        return True
    return any(t in _norm(f) for f in fields)


def _paper_title_map(db: Database) -> dict[str, tuple[str, int | None]]:
    """local_id -> (title, year) for every paper."""
    rows = db.conn.execute("SELECT local_id, title, year FROM papers").fetchall()
    return {r[0]: (r[1] or "", r[2]) for r in rows}


def _load_graph(db: Database):
    """Build an in-memory GraphEngine from the database's edges."""
    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)
    return graph


def _relevant_paper_ids(db: Database, topic: str | None) -> set[str] | None:
    """Resolve the paper scope for a topic, or None for the whole library."""
    if not topic:
        return None

    ids: set[str] = set()
    t = _norm(topic)

    # Papers whose title or abstract mention the topic.
    for p in db.get_all_papers():
        if t in _norm(p.get("title")) or t in _norm(p.get("abstract")):
            ids.add(p["local_id"])

    # Papers whose concepts mention the topic.
    like = f"%{t}%"
    for (lid,) in db.conn.execute(
        "SELECT DISTINCT local_id FROM concepts WHERE LOWER(label) LIKE ?", (like,)
    ).fetchall():
        ids.add(lid)

    return ids


def _seed_filter(seeds: list[dict], topic: str | None) -> list[dict]:
    """Drop seeds whose concept label doesn't match the topic."""
    if not topic:
        return seeds
    t = _norm(topic)
    return [s for s in seeds if t in _norm(s.get("concept")) or t in _norm(s.get("description"))]


def _collect_gaps(db: Database, graph, topic: str | None, top_n: int) -> dict:
    """Gap list from seeds + Gap/Debate concepts + paradigm shifts."""
    seeds = _seed_filter(graph.detect_research_seeds(db), topic)

    unaddressed = [s for s in seeds if s.get("type") == "unaddressed_gap"]
    debates = [s for s in seeds if s.get("type") == "debate_zone"]
    other = [s for s in seeds if s.get("type") not in ("unaddressed_gap", "debate_zone")]

    # Direct Gap/Debate concepts (authoritative source, even if seed detection
    # doesn't flag them).
    gap_concepts: list[dict] = []
    debate_concepts: list[dict] = []
    like = f"%{_norm(topic)}%" if topic else "%"
    rows = db.conn.execute(
        "SELECT label, type, section, node_id, local_id, provenance, authority, "
        "       valid_from, valid_to, confidence "
        "FROM concepts WHERE type IN ('Gap', 'Debate') AND LOWER(label) LIKE ? "
        "ORDER BY confidence DESC",
        (like,),
    ).fetchall()
    for label, ctype, section, node_id, paper_id, prov, auth, vf, vt, conf in rows:
        entry = {
            "label": label,
            "type": ctype,
            "section": section or "",
            "node_id": node_id or "",
            "paper_id": paper_id or "",
            "provenance": prov or "",
            "authority": auth or "",
            "valid_from": vf,
            "valid_to": vt,
            "confidence": conf,
        }
        if ctype == "Gap":
            gap_concepts.append(entry)
        else:
            debate_concepts.append(entry)

    # Frontier: paradigm shifts (replacement / explosion / cross_domain).
    shifts: list[dict] = []
    try:
        from drbrain.graph.genealogy import detect_paradigm_shifts

        for s in detect_paradigm_shifts(graph, db):
            if not _matches(topic, s.get("description"), s.get("concept")):
                continue
            shifts.append(
                {
                    "type": s.get("type", ""),
                    "description": s.get("description", ""),
                }
            )
    except Exception:  # noqa: BLE001 — frontier is best-effort
        shifts = []

    return {
        "unaddressed_gaps": unaddressed[:top_n],
        "gap_concepts": gap_concepts[:top_n],
        "debates": debates[:top_n],
        "debate_concepts": debate_concepts[:top_n],
        "other_seeds": other[:top_n],
        "paradigm_shifts": shifts[:top_n],
        "counts": {
            "seeds": len(seeds),
            "gap_concepts": len(gap_concepts),
            "debate_concepts": len(debate_concepts),
        },
    }


def _collect_cross_refs(db: Database, paper_ids: set[str] | None, top_n: int) -> dict:
    """Who-cites-whom + shared references, via paper_citations and citation_graph."""
    title_map = _paper_title_map(db)

    # A topic with zero matches yields an empty (not None) paper scope — there
    # is nothing to cross-reference in that case.
    if paper_ids is not None and not paper_ids:
        return {
            "direct_citations": [],
            "shared_references": [],
            "paper_overview": [],
            "counts": {"direct_citations": 0, "shared_references": 0},
        }

    # 1. Direct citations (who cites whom), scoped to the topic when present.
    direct: list[dict] = []
    if paper_ids is not None:
        ph = ",".join("?" for _ in paper_ids)
        params = tuple(paper_ids) * 2
        rows = db.conn.execute(
            f"SELECT citing_id, cited_id, year FROM paper_citations "
            f"WHERE citing_id IN ({ph}) OR cited_id IN ({ph}) "
            f"ORDER BY year DESC",
            params,
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT citing_id, cited_id, year FROM paper_citations ORDER BY year DESC"
        ).fetchall()
    for citing, cited, year in rows:
        c_title, _ = title_map.get(citing, (citing, None))
        d_title, _ = title_map.get(cited, (cited, None))
        direct.append(
            {
                "citing_id": citing,
                "citing_title": c_title,
                "cited_id": cited,
                "cited_title": d_title,
                "year": year,
            }
        )
    direct = direct[:top_n]

    # 2. Shared references: a paper cited by >=2 distinct papers in scope.
    shared: list[dict] = []
    if paper_ids is not None:
        ph = ",".join("?" for _ in paper_ids)
        rows = db.conn.execute(
            f"SELECT cited_id, COUNT(DISTINCT citing_id) AS n "
            f"FROM paper_citations WHERE citing_id IN ({ph}) "
            f"GROUP BY cited_id HAVING n >= 2 ORDER BY n DESC",
            tuple(paper_ids),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT cited_id, COUNT(DISTINCT citing_id) AS n "
            "FROM paper_citations GROUP BY cited_id HAVING n >= 2 ORDER BY n DESC"
        ).fetchall()
    for cited_id, n in rows:
        citers = db.conn.execute(
            "SELECT DISTINCT citing_id FROM paper_citations WHERE cited_id = ?",
            (cited_id,),
        ).fetchall()
        citing_titles = [title_map.get(c[0], (c[0], None))[0] for c in citers]
        shared.append(
            {
                "cited_id": cited_id,
                "cited_title": title_map.get(cited_id, (cited_id, None))[0],
                "count": n,
                "citing_titles": citing_titles,
            }
        )
    shared = shared[:top_n]

    # 3. Per-paper citation counts via citation_graph (citation_cache based).
    from drbrain.storage.citation_graph import query_citation_graph

    overview: list[dict] = []
    papers = (
        [p for p in db.get_all_papers() if p["local_id"] in paper_ids]
        if paper_ids is not None
        else db.get_all_papers()
    )
    for p in papers[:top_n]:
        try:
            cg = query_citation_graph(p["local_id"], db.conn, ctype="all")
            counts = cg.get("counts", {})
            overview.append(
                {
                    "local_id": p["local_id"],
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "references": counts.get("references", 0),
                    "citing": counts.get("citing", 0),
                    "shared_refs": len(cg.get("shared_refs", [])),
                }
            )
        except Exception:  # noqa: BLE001 — citation overview is best-effort
            continue

    return {
        "direct_citations": direct,
        "shared_references": shared,
        "paper_overview": overview,
        "counts": {
            "direct_citations": len(direct),
            "shared_references": len(shared),
        },
    }


def _collect_evidence(db: Database, topic: str | None, top_n: int) -> dict:
    """Evidence chain: every Gap/Debate/Conclusion traced to its source."""
    title_map = _paper_title_map(db)
    like = f"%{_norm(topic)}%" if topic else "%"

    gap_evidence: list[dict] = []
    rows = db.conn.execute(
        "SELECT label, type, section, node_id, local_id, provenance, authority, "
        "       valid_from, valid_to, confidence "
        "FROM concepts WHERE type IN ('Gap', 'Debate', 'Conclusion') "
        "AND LOWER(label) LIKE ? ORDER BY type, confidence DESC",
        (like,),
    ).fetchall()
    for label, ctype, section, node_id, paper_id, prov, auth, vf, vt, conf in rows:
        title, year = title_map.get(paper_id, ("", None))
        gap_evidence.append(
            {
                "label": label,
                "type": ctype,
                "paper_id": paper_id or "",
                "paper_title": title,
                "paper_year": year,
                "section": section or "",
                "node_id": node_id or "",
                "provenance": prov or "",
                "authority": auth or "",
                "valid_from": vf,
                "valid_to": vt,
                "confidence": conf,
            }
        )

    # Answers bound to their evidence (answer_records, v13).
    answers: list[dict] = []
    try:
        arows = db.conn.execute(
            "SELECT question, answer, evidence_ids, provenance, model_version "
            "FROM answer_records ORDER BY answer_id DESC LIMIT ?",
            (top_n,),
        ).fetchall()
        for question, answer, evidence_ids, prov, model in arows:
            if not _matches(topic, question, answer):
                continue
            answers.append(
                {
                    "question": question,
                    "answer": answer,
                    "evidence_ids": evidence_ids or "",
                    "provenance": prov or "",
                    "model_version": model or "",
                }
            )
    except Exception:  # noqa: BLE001 — answer_records is optional
        answers = []

    return {
        "gap_evidence": gap_evidence[:top_n],
        "answer_records": answers[:top_n],
        "counts": {"gap_evidence": len(gap_evidence), "answer_records": len(answers)},
    }


def generate_survey_data(
    db: Database,
    topic: str | None = None,
    *,
    graph=None,
    top_n: int = 10,
) -> dict:
    """Collect structured survey data (gaps / cross-refs / evidence)."""
    if graph is None:
        graph = _load_graph(db)

    paper_ids = _relevant_paper_ids(db, topic)
    n_papers = len(paper_ids) if paper_ids is not None else len(db.get_all_papers())

    gaps = _collect_gaps(db, graph, topic, top_n)
    cross = _collect_cross_refs(db, paper_ids, top_n)
    evidence = _collect_evidence(db, topic, top_n)

    return {
        "topic": topic or "",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            "paper_count": n_papers,
            "seeds": gaps["counts"]["seeds"],
            "gap_concepts": gaps["counts"]["gap_concepts"],
            "debate_concepts": gaps["counts"]["debate_concepts"],
            "direct_citations": cross["counts"]["direct_citations"],
            "shared_references": cross["counts"]["shared_references"],
            "evidence_items": evidence["counts"]["gap_evidence"],
        },
        "gaps": gaps,
        "cross_references": cross,
        "evidence_chain": evidence,
    }


# ── Markdown rendering ─────────────────────────────────────────────────────


def _md_join(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line != "")


def _render_gaps(gaps: dict) -> list[str]:
    out: list[str] = [H_GAPS, ""]

    unaddressed = gaps["unaddressed_gaps"]
    gap_concepts = gaps["gap_concepts"]
    seed_by_label = {s.get("concept"): s for s in unaddressed}
    if unaddressed or gap_concepts:
        out.append("### 1.1 研究空白（未解决 Gap）")
        out.append("")
        seen: set[str] = set()
        for g in gap_concepts:
            seen.add(g["label"])
            out.append(f"- **{g['label']}** — 类型 `Gap`（置信度 {g['confidence']:.2f}）")
            seed = seed_by_label.get(g["label"])
            if seed:
                out.append(f"  - 依据：{seed.get('description')}")
            prov = _format_source(g)
            if prov:
                out.append(f"  - 来源：{prov}")
        for s in unaddressed:
            if s.get("concept") in seen:
                continue
            out.append(f"- **{s.get('concept')}** — {s.get('description')}")
            out.append(f"  - 置信度 {s.get('confidence', 0):.2f}")
        out.append("")
    else:
        out.append("### 1.1 研究空白（未解决 Gap）")
        out.append("")
        out.append("_未检测到研究空白。_")
        out.append("")

    debates = gaps["debates"]
    debate_concepts = gaps["debate_concepts"]
    dseed_by_label = {s.get("concept"): s for s in debates}
    out.append("### 1.2 活跃争论（Debate）")
    out.append("")
    if debates or debate_concepts:
        seen_d: set[str] = set()
        for d in debate_concepts:
            seen_d.add(d["label"])
            out.append(f"- **{d['label']}** — 类型 `Debate`（置信度 {d['confidence']:.2f}）")
            seed = dseed_by_label.get(d["label"])
            if seed:
                out.append(f"  - 依据：{seed.get('description')}")
            prov = _format_source(d)
            if prov:
                out.append(f"  - 来源：{prov}")
        for s in debates:
            if s.get("concept") in seen_d:
                continue
            out.append(f"- **{s.get('concept')}** — {s.get('description')}")
            out.append(f"  - 置信度 {s.get('confidence', 0):.2f}")
    else:
        out.append("_未检测到活跃争论。_")
    out.append("")

    other = gaps["other_seeds"]
    if other:
        out.append("### 1.3 其他研究信号")
        out.append("")
        for s in other:
            label = s.get("concept") or s.get("description")
            out.append(
                f"- **[{s.get('type')}]** {label} — {s.get('description')} "
                f"（置信度 {s.get('confidence', 0):.2f}）"
            )
        out.append("")

    shifts = gaps["paradigm_shifts"]
    if shifts:
        out.append("### 1.4 知识前沿（范式迁移）")
        out.append("")
        for s in shifts:
            out.append(f"- **[{s.get('type')}]** {s.get('description')}")
        out.append("")

    return out


def _format_source(entry: dict) -> str:
    """Compact source description for a gap/debate concept."""
    parts: list[str] = []
    if entry.get("paper_title"):
        year = entry.get("paper_year")
        parts.append(f"论文《{entry['paper_title']}》({year or '年份不详'})")
    elif entry.get("paper_id"):
        parts.append(f"论文 {entry['paper_id']}")
    if entry.get("section"):
        parts.append(f"章节「{entry['section']}」")
    if entry.get("authority"):
        parts.append(f"权威性 {entry['authority']}")
    if entry.get("provenance"):
        parts.append(f"来源 {entry['provenance']}")
    return "，".join(parts) if parts else ""


def _render_cross(cross: dict) -> list[str]:
    out: list[str] = [H_CROSS, ""]

    out.append("### 2.1 直接引用关系（谁引谁）")
    out.append("")
    direct = cross["direct_citations"]
    if direct:
        out.append("| 引用方 | 被引方 | 年份 |")
        out.append("| --- | --- | --- |")
        for d in direct:
            out.append(f"| {d['citing_title']} | {d['cited_title']} | {d['year'] or '—'} |")
    else:
        out.append("_未发现论文间的直接引用关系。_")
    out.append("")

    out.append("### 2.2 共享引用（多篇论文共同引用）")
    out.append("")
    shared = cross["shared_references"]
    if shared:
        for s in shared:
            citers = "、".join(s["citing_titles"])
            out.append(f"- **{s['cited_title']}** — 被 {s['count']} 篇论文共同引用（{citers}）")
    else:
        out.append("_未发现共享引用。_")
    out.append("")

    overview = cross["paper_overview"]
    if overview:
        out.append("### 2.3 单篇引用概览")
        out.append("")
        for p in overview:
            year = p.get("year")
            out.append(
                f"- **{p['title']}** ({year or '年份不详'}) — "
                f"参考文献 {p['references']} 篇 / 被引 {p['citing']} 次 / "
                f"共享引用 {p['shared_refs']} 项"
            )
        out.append("")

    return out


def _render_evidence(evidence: dict) -> list[str]:
    out: list[str] = [H_EVIDENCE, ""]

    out.append("### 3.1 Gap / 结论证据链")
    out.append("")
    gap_evidence = evidence["gap_evidence"]
    if gap_evidence:
        for g in gap_evidence:
            out.append(f"#### {g['label']}")
            out.append("")
            out.append(f"- **类型**：{g['type']}")
            if g["paper_title"]:
                out.append(f"- **来源论文**：{g['paper_title']} ({g['paper_year'] or '年份不详'})")
            elif g["paper_id"]:
                out.append(f"- **来源论文**：{g['paper_id']}")
            if g["section"]:
                out.append(f"- **原文位置**：章节「{g['section']}」")
            if g["node_id"]:
                out.append(f"- **节点**：{g['node_id']}")
            if g["provenance"]:
                out.append(f"- **provenance**：{g['provenance']}")
            if g["authority"]:
                out.append(f"- **权威性 (authority)**：{g['authority']}")
            if g["valid_from"] is not None or g["valid_to"] is not None:
                vf = g["valid_from"] if g["valid_from"] is not None else "?"
                vt = g["valid_to"] if g["valid_to"] is not None else "至今"
                out.append(f"- **有效性窗口**：{vf}–{vt}")
            out.append(f"- **置信度**：{g['confidence']:.2f}")
            out.append("")
    else:
        out.append("_无 Gap / 结论可供追溯。_")
        out.append("")

    answers = evidence["answer_records"]
    if answers:
        out.append("### 3.2 回答记录（answer_records）")
        out.append("")
        for a in answers:
            out.append(f"- **Q**: {a['question']}")
            out.append(f"  - **A**: {_truncate(a['answer'], 200)}")
            if a["evidence_ids"]:
                out.append(f"  - **证据 ID**：{a['evidence_ids']}")
            if a["provenance"]:
                out.append(f"  - **provenance**：{a['provenance']}")
            if a["model_version"]:
                out.append(f"  - **模型**：{a['model_version']}")
        out.append("")

    return out


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def generate_survey(
    db: Database,
    topic: str | None = None,
    *,
    graph=None,
    top_n: int = 10,
) -> str:
    """Generate a markdown literature survey report for a topic (or whole library).

    Args:
        db: Open ``Database`` instance.
        topic: Optional topic filter (matched against concept labels / paper
            titles / abstracts). When ``None``, the whole library is surveyed.
        graph: Optional pre-built ``GraphEngine`` (avoids re-loading edges).
        top_n: Max items per section.

    Returns:
        Markdown text with three sections: Gap 清单 / 文献交叉引用 / 证据链.
    """
    data = generate_survey_data(db, topic, graph=graph, top_n=top_n)
    summary = data["summary"]

    title = f"# 文献调研综述：{topic}" if topic else "# 文献调研综述（全库）"
    subtitle = (
        f"> 生成时间 {data['generated_at']} · 覆盖 {summary['paper_count']} 篇论文 · "
        f"{summary['seeds']} 个研究种子 · {summary['gap_concepts']} 个 Gap · "
        f"{summary['debate_concepts']} 个 Debate · "
        f"{summary['direct_citations']} 条直接引用 · {summary['shared_references']} 组共享引用"
    )

    blocks: list[str] = [title, "", subtitle, ""]
    blocks.extend(_render_gaps(data["gaps"]))
    blocks.extend(_render_cross(data["cross_references"]))
    blocks.extend(_render_evidence(data["evidence_chain"]))

    return _md_join(blocks).strip() + "\n"
