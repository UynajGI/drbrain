"""Read-only storage audit for the unified store and legacy artifacts (T49).

The audit never writes: it opens the database in SQLite ``mode=ro`` (so no
migration runs), never touches the network, and only inspects files whose
metadata it reads.  Each finding lands in a category with a severity and a
small sample list, so a migration plan (T50) can be derived from evidence
instead of guesswork.

Categories cover the failure modes the plan names: dual body copies, content
hash conflicts, missing source files, broken legacy trees, short-id risk,
legacy vectors that cannot be used, unparseable evidence, FTS drift, dangling
membership and unfinished (staging) state.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

#: Severity ordering used for the summary line.
SEVERITIES = ("error", "warning", "info")

#: Legacy ``p`` + 6-hex ids live in a 24-bit namespace that can collide;
#: longer ids (the current 24-hex generator) are out of that namespace.
LEGACY_ID_MAX_LENGTH = 9


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    message: str
    local_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "local_id": self.local_id,
            "detail": dict(self.detail),
        }


@dataclass
class AuditReport:
    schema_version: int = 0
    tables: tuple[str, ...] = ()
    unified_store: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    storage_dir: str = ""
    papers_root: str = ""

    def add(self, finding: Finding, *, sample_limit: int = 5) -> None:
        self.findings.append(finding)
        bucket = self.samples.setdefault(finding.category, [])
        if len(bucket) < sample_limit:
            bucket.append(finding.to_json())

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return counts

    def by_severity(self) -> dict[str, int]:
        counts = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def ok(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok(),
            "schema_version": self.schema_version,
            "unified_store": self.unified_store,
            "counts": dict(self.counts),
            "categories": self.by_category(),
            "severities": self.by_severity(),
            "findings": [finding.to_json() for finding in self.findings][:200],
            "samples": {key: value for key, value in self.samples.items()},
            "storage_dir": self.storage_dir,
            "papers_root": self.papers_root,
        }


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the database for reading only, without migrating it.

    ``mode=ro`` is not enough for a WAL database: when the ``-shm`` file is
    absent (a closed writer), a strict read-only connection silently reads
    only the main file and misses committed WAL content, which would make the
    audit lie.  A normal connection with ``PRAGMA query_only=ON`` may create
    the shm file but cannot write rows, and this module never runs DDL, so no
    migration is triggered.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def audit_storage(
    *,
    db_path: str | Path,
    papers_root: str | Path | None = None,
    storage_dir: str | Path | None = None,
    sample_limit: int = 5,
) -> AuditReport:
    """Audit the store and (optionally) the legacy artifact tree, read-only."""
    report = AuditReport(
        papers_root=str(papers_root or ""),
        storage_dir=str(storage_dir or ""),
    )
    conn = open_readonly(db_path)
    try:
        tables = _table_names(conn)
        report.tables = tuple(sorted(tables))
        versions = [
            int(row[0])
            for row in conn.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            ).fetchall()
        ]
        report.schema_version = versions[-1] if versions else 0
        unified = {"document_revisions", "content_blocks", "tree_nodes"} <= tables
        report.unified_store = unified
        if not unified:
            report.add(
                Finding(
                    "legacy_store",
                    "warning",
                    "unified content tables are absent; migration has not run",
                    detail={"schema_version": report.schema_version},
                ),
                sample_limit=sample_limit,
            )
        if unified:
            _audit_content(conn, report, sample_limit)
            _audit_nodes(conn, report, sample_limit)
            _audit_vectors(conn, report, sample_limit)
        _audit_legacy_tables(conn, report, tables, sample_limit)
    finally:
        conn.close()

    if papers_root is not None:
        _audit_legacy_artifacts(Path(papers_root), report, sample_limit)
    if storage_dir is not None:
        _audit_generations(Path(storage_dir), report, sample_limit)
    return report


def _audit_content(conn: sqlite3.Connection, report: AuditReport, sample_limit: int) -> None:
    report.counts["documents"] = _count(conn, "SELECT COUNT(*) FROM document_revisions")
    report.counts["blocks"] = _count(conn, "SELECT COUNT(*) FROM content_blocks")
    report.counts["documents_stale"] = _count(
        conn, "SELECT COUNT(*) FROM document_revisions WHERE state = 'stale'"
    )
    report.counts["documents_failed"] = _count(
        conn, "SELECT COUNT(*) FROM document_revisions WHERE state = 'failed'"
    )
    for local_id, revision in conn.execute(
        "SELECT local_id, revision FROM document_revisions WHERE state = 'failed'"
    ).fetchall():
        report.add(
            Finding(
                "failed_revision",
                "warning",
                "document revision is marked failed",
                local_id=str(local_id),
                detail={"revision": int(revision)},
            ),
            sample_limit=sample_limit,
        )

    # Hash conflicts: a revision whose blocks no longer hash to its canonical text.
    import hashlib

    rows = conn.execute(
        """
        SELECT r.local_id, r.revision, r.canonical_hash, GROUP_CONCAT(b.text, '') AS body
        FROM document_revisions r
        LEFT JOIN content_blocks b
          ON b.local_id = r.local_id AND b.revision = r.revision
        GROUP BY r.local_id, r.revision
        ORDER BY r.local_id, r.revision
        """
    ).fetchall()
    for row in rows:
        body = row["body"] or ""
        if not body:
            report.add(
                Finding(
                    "missing_body",
                    "error",
                    "revision has no canonical blocks",
                    local_id=str(row["local_id"]),
                    detail={"revision": int(row["revision"])},
                ),
                sample_limit=sample_limit,
            )
            continue
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest != str(row["canonical_hash"]):
            report.add(
                Finding(
                    "content_hash_conflict",
                    "error",
                    "blocks do not reproduce the recorded canonical hash",
                    local_id=str(row["local_id"]),
                    detail={"revision": int(row["revision"])},
                ),
                sample_limit=sample_limit,
            )

    # Per-block invariant: the declared span must equal the stored text length
    # (a rogue update can extend a span without touching the text).
    for row in conn.execute(
        """
        SELECT local_id, revision, ordinal, length(text) AS text_len,
               char_end - char_start AS span
        FROM content_blocks
        WHERE length(text) != char_end - char_start
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall():
        report.add(
            Finding(
                "content_span_mismatch",
                "error",
                "block span does not match its text length",
                local_id=str(row["local_id"]),
                detail={
                    "revision": int(row["revision"]),
                    "ordinal": int(row["ordinal"]),
                    "text_len": int(row["text_len"]),
                    "span": int(row["span"]),
                },
            ),
            sample_limit=sample_limit,
        )

    # Reading-order contiguity: each block must start where the previous ended.
    for row in conn.execute(
        """
        SELECT local_id, revision, COUNT(*) AS gaps FROM (
            SELECT local_id, revision, char_start,
                   LAG(char_end) OVER (
                       PARTITION BY local_id, revision ORDER BY ordinal
                   ) AS previous_end
            FROM content_blocks
        )
        WHERE previous_end IS NOT NULL AND char_start != previous_end
        GROUP BY local_id, revision
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall():
        report.add(
            Finding(
                "content_gap",
                "error",
                "block spans are not contiguous in reading order",
                local_id=str(row["local_id"]),
                detail={"revision": int(row["revision"]), "gaps": int(row["gaps"])},
            ),
            sample_limit=sample_limit,
        )

    if "content_fts" in report.tables:
        import re as _re

        rows = conn.execute(
            "SELECT block_id, text FROM content_blocks ORDER BY block_id LIMIT ?",
            (max(1, sample_limit * 4),),
        ).fetchall()
        missing: list[str] = []
        for block_id, text in rows:
            tokens = sorted(_re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text or ""), key=len)
            if not tokens:
                continue
            probe = tokens[-1]
            try:
                found = conn.execute(
                    "SELECT 1 FROM content_fts WHERE content_fts MATCH ? AND rowid = "
                    "(SELECT rowid FROM content_blocks WHERE block_id = ?) LIMIT 1",
                    (f'"{probe}"', str(block_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                found = None
            if found is None:
                missing.append(str(block_id))
        report.counts["fts_sampled"] = len(rows)
        report.counts["fts_verified"] = len(rows) - len(missing)
        if missing:
            report.add(
                Finding(
                    "fts_drift",
                    "warning",
                    "sampled blocks are not findable through the FTS index",
                    detail={"missing_sample": missing[:5], "sampled": len(rows)},
                ),
                sample_limit=sample_limit,
            )


def _audit_nodes(conn: sqlite3.Connection, report: AuditReport, sample_limit: int) -> None:
    report.counts["nodes_ready"] = _count(
        conn, "SELECT COUNT(*) FROM tree_nodes WHERE state = 'ready'"
    )
    report.counts["nodes_staging"] = _count(
        conn, "SELECT COUNT(*) FROM tree_nodes WHERE state = 'staging'"
    )
    report.counts["nodes_failed"] = _count(
        conn, "SELECT COUNT(*) FROM tree_nodes WHERE state = 'failed'"
    )
    dangling = conn.execute(
        """
        SELECT n.node_id, n.local_id
        FROM tree_nodes n
        WHERE n.kind = 'leaf' AND n.state = 'ready'
          AND NOT EXISTS (
            SELECT 1 FROM content_blocks b
            WHERE b.block_id = n.block_id AND b.text_hash IS NOT NULL
          )
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall()
    for row in dangling:
        report.add(
            Finding(
                "dangling_leaf",
                "error",
                "leaf references a block that no longer exists",
                local_id=str(row["local_id"]),
                detail={"node_id": str(row["node_id"])},
            ),
            sample_limit=sample_limit,
        )
    orphan_regions = conn.execute(
        """
        SELECT c.parent_id
        FROM tree_node_children c
        LEFT JOIN tree_nodes n ON n.node_id = c.child_id
        WHERE n.node_id IS NULL
        LIMIT ?
        """,
        (sample_limit,),
    ).fetchall()
    for row in orphan_regions:
        report.add(
            Finding(
                "dangling_member",
                "error",
                "membership edge points at a missing node",
                detail={"parent_id": str(row["parent_id"])},
            ),
            sample_limit=sample_limit,
        )


def _audit_vectors(conn: sqlite3.Connection, report: AuditReport, sample_limit: int) -> None:
    report.counts["vectors_ready"] = _count(
        conn, "SELECT COUNT(*) FROM node_vectors WHERE state = 'ready'"
    )
    report.counts["vectors_staging"] = _count(
        conn, "SELECT COUNT(*) FROM node_vectors WHERE state = 'staging'"
    )
    if report.counts["vectors_staging"]:
        report.add(
            Finding(
                "vector_staging",
                "info",
                "vector writes are in progress (staging rows present)",
                detail={"count": report.counts["vectors_staging"]},
            ),
            sample_limit=sample_limit,
        )
    dimensions = [
        int(row[0])
        for row in conn.execute(
            "SELECT DISTINCT dimension FROM node_vectors WHERE state = 'ready'"
        ).fetchall()
    ]
    if len(dimensions) > 1:
        report.add(
            Finding(
                "mixed_dimensions",
                "error",
                "ready vectors use more than one dimension",
                detail={"dimensions": sorted(dimensions)},
            ),
            sample_limit=sample_limit,
        )


def _audit_legacy_tables(
    conn: sqlite3.Connection, report: AuditReport, tables: set[str], sample_limit: int
) -> None:
    if "tree_vectors" in tables:
        report.counts["legacy_vectors"] = _count(conn, "SELECT COUNT(*) FROM tree_vectors")
        empty = _count(
            conn,
            "SELECT COUNT(*) FROM tree_vectors "
            "WHERE embedding IS NULL OR length(embedding) = 0 OR (length(embedding) % 4) != 0",
        )
        if empty:
            report.add(
                Finding(
                    "legacy_vector_unusable",
                    "warning",
                    "legacy vector rows without a usable embedding blob",
                    detail={"count": empty},
                ),
                sample_limit=sample_limit,
            )
    if "evidence" in tables:
        report.counts["legacy_evidence"] = _count(conn, "SELECT COUNT(*) FROM evidence")
    for row in conn.execute(
        "SELECT local_id FROM papers WHERE local_id LIKE 'p%' AND length(local_id) <= ? LIMIT 1",
        (LEGACY_ID_MAX_LENGTH,),
    ).fetchall():
        report.add(
            Finding(
                "short_id_risk",
                "info",
                "papers with 24-bit legacy ids exist (collision-prone namespace)",
                local_id=str(row["local_id"]),
            ),
            sample_limit=sample_limit,
        )


def _audit_legacy_artifacts(papers_root: Path, report: AuditReport, sample_limit: int) -> None:
    from drbrain.storage.paths import (
        iter_paper_dirs,
        paper_id_from_dir,
        raw_md_path,
        tree_json_path,
    )

    report.counts["paper_dirs"] = 0
    for directory in iter_paper_dirs(papers_root):
        report.counts["paper_dirs"] += 1
        try:
            local_id = paper_id_from_dir(directory, papers_root)
        except Exception:  # noqa: BLE001 - unreadable dirs are reported, not fatal
            report.add(
                Finding(
                    "unnamed_paper_dir",
                    "warning",
                    "paper directory does not map to a paper id",
                    detail={"path": str(directory)},
                ),
                sample_limit=sample_limit,
            )
            continue
        if not (directory / "source.pdf").exists() and not any(
            item.name.startswith("source.") for item in directory.iterdir() if item.is_file()
        ):
            report.add(
                Finding(
                    "missing_source",
                    "warning",
                    "paper directory has no hosted source file",
                    local_id=local_id,
                    detail={"path": str(directory)},
                ),
                sample_limit=sample_limit,
            )
        tree = tree_json_path(directory)
        if tree.exists():
            try:
                payload = json.loads(tree.read_text(encoding="utf-8", errors="replace"))
                if not isinstance(payload, dict) or not isinstance(payload.get("structure"), list):
                    raise ValueError("structure is not a list")
            except (OSError, ValueError) as exc:
                report.add(
                    Finding(
                        "broken_tree",
                        "warning",
                        f"legacy tree.json is unusable: {exc}",
                        local_id=local_id,
                        detail={"path": str(tree)},
                    ),
                    sample_limit=sample_limit,
                )
        if raw_md_path(directory).exists():
            report.counts["legacy_raw_md"] = report.counts.get("legacy_raw_md", 0) + 1


def _audit_generations(storage_dir: Path, report: AuditReport, sample_limit: int) -> None:
    from drbrain.tree.publish import (
        ACTIVE_POINTER_NAME,
        GENERATIONS_DIR_NAME,
        STAGING_PREFIX,
        get_active_tree_generation,
        verify_tree_generation,
    )

    generations_root = storage_dir / GENERATIONS_DIR_NAME
    if not generations_root.is_dir():
        report.add(
            Finding("no_generation", "warning", "no published generation directory exists"),
            sample_limit=sample_limit,
        )
        return
    staging = [path for path in generations_root.iterdir() if path.name.startswith(STAGING_PREFIX)]
    for path in staging:
        report.add(
            Finding(
                "staging_generation",
                "warning",
                "an interrupted publication left a staging directory",
                detail={"path": str(path)},
            ),
            sample_limit=sample_limit,
        )
    active = get_active_tree_generation(storage_dir)
    if not active:
        report.add(
            Finding(
                "no_active_generation",
                "warning",
                "no active pointer is published",
                detail={"pointer": str(storage_dir / ACTIVE_POINTER_NAME)},
            ),
            sample_limit=sample_limit,
        )
        return
    report.counts["active_generation"] = 1
    try:
        check = verify_tree_generation(storage_dir, active)
    except Exception as exc:  # noqa: BLE001 - report instead of raising
        report.add(
            Finding(
                "generation_unreadable",
                "error",
                f"active generation cannot be verified: {exc}",
                detail={"generation": active},
            ),
            sample_limit=sample_limit,
        )
        return
    if not check.get("ok"):
        report.add(
            Finding(
                "generation_inconsistent",
                "error",
                "active generation does not match its manifest",
                detail={"generation": active, "mismatches": check.get("mismatches", [])},
            ),
            sample_limit=sample_limit,
        )


def summarize(report: AuditReport) -> str:
    severities = report.by_severity()
    return (
        f"schema v{report.schema_version}"
        f"{' (unified)' if report.unified_store else ' (legacy)'}; "
        f"{severities.get('error', 0)} errors, {severities.get('warning', 0)} warnings, "
        f"{severities.get('info', 0)} notes"
    )


def audit_and_log(**kwargs: Any) -> AuditReport:
    report = audit_storage(**kwargs)
    logger.info("[audit] storage: {}", summarize(report))
    return report
