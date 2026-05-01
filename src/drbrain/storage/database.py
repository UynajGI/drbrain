"""SQLite backend with schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    local_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT DEFAULT '',
    year INTEGER,
    paper_type TEXT NOT NULL DEFAULT 'paper'
        CHECK(paper_type IN ('paper','review','thesis','preprint','book','document')),
    status TEXT NOT NULL DEFAULT 'placeholder' CHECK(status IN ('uploaded', 'placeholder', 'merged')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_ids (
    local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    doi TEXT UNIQUE,
    arxiv TEXT UNIQUE,
    s2_id TEXT UNIQUE,
    openalex_id TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_id TEXT NOT NULL REFERENCES papers(local_id),
    type TEXT NOT NULL CHECK(type IN ('Problem', 'Method', 'Conclusion', 'Debate', 'Gap', 'Actor')),
    label TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    section TEXT DEFAULT '',
    first_seen INTEGER,
    last_seen INTEGER
);

CREATE TABLE IF NOT EXISTS arguments (
    arg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL REFERENCES papers(local_id),
    claim TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK(claim_type IN ('supports', 'challenges', 'extends', 'limits', 'solves', 'proposes')),
    target_label TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK(target_type IN ('Method', 'Problem', 'Conclusion', 'Gap', 'Debate', 'Argument')),
    evidence_type TEXT CHECK(evidence_type IN ('empirical', 'theoretical', 'case_study', 'survey')),
    evidence_detail TEXT,
    mechanism TEXT DEFAULT '',
    section TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, relation, source_paper)
);

CREATE TABLE IF NOT EXISTS aliases (
    variant TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS confidence_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL,
    item_type TEXT NOT NULL CHECK(item_type IN ('concept', 'alias', 'relation')),
    item_data TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type);
CREATE INDEX IF NOT EXISTS idx_concepts_label ON concepts(label);
CREATE INDEX IF NOT EXISTS idx_concepts_first_seen ON concepts(first_seen);
CREATE INDEX IF NOT EXISTS idx_arguments_source ON arguments(source_paper);
CREATE INDEX IF NOT EXISTS idx_arguments_target ON arguments(target_label);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON confidence_queue(status);

CREATE TABLE IF NOT EXISTS research_seeds (
    seed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS citation_cache (
    source_paper TEXT NOT NULL,
    target_title TEXT NOT NULL,
    target_year INTEGER,
    relation TEXT NOT NULL CHECK(relation IN ('references','citing')),
    target_doi TEXT,
    target_s2_id TEXT,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_paper, target_title)
);
"""


class Database:
    """Thin SQLite wrapper with schema auto-init."""

    def __init__(self, db_path: str | Path = "data/db/drbrain.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self._migrate_add_paper_type()
        self.conn.commit()

    def _migrate_add_paper_type(self) -> None:
        """Add paper_type column if missing (pre-v2 DBs)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "paper_type" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN paper_type TEXT NOT NULL DEFAULT 'paper'"
            )

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- Paper queries --

    def get_paper_by_external_id(self, id_type: str, value: str) -> str | None:
        """Look up local_id by external identifier."""
        col = {"doi": "doi", "arxiv": "arxiv", "s2_id": "s2_id", "openalex_id": "openalex_id"}[
            id_type
        ]
        row = self.conn.execute(
            f"SELECT local_id FROM paper_ids WHERE {col} = ?", (value,)
        ).fetchone()
        return row[0] if row else None

    def fuzzy_match_title_year(self, title: str, year: int) -> str | None:
        """Simple exact title+year match. Upgrade to SimHash later."""
        row = self.conn.execute(
            "SELECT local_id FROM papers WHERE title = ? AND year = ?",
            (title, year),
        ).fetchone()
        return row[0] if row else None

    def insert_paper(
        self,
        local_id: str,
        title: str,
        year: int | None,
        status: str,
        paper_type: str = "paper",
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO papers (local_id, title, year, status, paper_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (local_id, title, year, status, paper_type),
        )

    def insert_paper_ids(
        self, local_id: str, doi=None, arxiv=None, s2_id=None, openalex_id=None
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_ids (local_id, doi, arxiv, s2_id, openalex_id) VALUES (?, ?, ?, ?, ?)",
            (local_id, doi, arxiv, s2_id, openalex_id),
        )

    def set_paper_abstract(self, local_id: str, abstract: str) -> None:
        self.conn.execute(
            "UPDATE papers SET abstract = ? WHERE local_id = ?",
            (abstract, local_id),
        )

    def upgrade_placeholder(self, local_id: str) -> None:
        self.conn.execute(
            "UPDATE papers SET status = 'uploaded' WHERE local_id = ? AND status = 'placeholder'",
            (local_id,),
        )

    # -- Concept/edge/alias/seed inserts --

    def insert_concept(
        self,
        local_id: str,
        ctype: str,
        label: str,
        confidence: float = 1.0,
        year: int | None = None,
        section: str = "",
    ) -> int:
        """Insert a concept with temporal tracking. Returns concept_id."""
        cur = self.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (local_id, ctype, label, confidence, section, year, year),
        )
        return cur.lastrowid

    def insert_edge(
        self, src_id: str, dst_id: str, relation: str, source_paper: str, weight: float = 1.0
    ) -> None:
        """Insert an edge between concepts."""
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
            (src_id, dst_id, relation, source_paper, weight),
        )

    def insert_alias(self, variant: str, canonical_id: str) -> None:
        """Insert an alias mapping."""
        self.conn.execute(
            "INSERT OR IGNORE INTO aliases (variant, canonical_id) VALUES (?, ?)",
            (variant, canonical_id),
        )

    def insert_seed(self, pattern_type: str, description: str, confidence: float = 0.0) -> int:
        """Insert a research seed and return its seed_id."""
        cur = self.conn.execute(
            "INSERT INTO research_seeds (pattern_type, description, confidence) VALUES (?, ?, ?)",
            (pattern_type, description, confidence),
        )
        return cur.lastrowid

    # -- Query helpers --

    def get_all_papers(self) -> list[dict]:
        """Return all papers as list of dicts."""
        rows = self.conn.execute(
            "SELECT p.local_id, p.title, p.abstract, p.year, p.paper_type, p.status, p.created_at, "
            "pi.doi, pi.arxiv, pi.s2_id, pi.openalex_id "
            "FROM papers p LEFT JOIN paper_ids pi ON p.local_id = pi.local_id"
        ).fetchall()
        cols = [
            "local_id",
            "title",
            "abstract",
            "year",
            "paper_type",
            "status",
            "created_at",
            "doi",
            "arxiv",
            "s2_id",
            "openalex_id",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_paper(self, local_id: str) -> dict | None:
        """Get a single paper by local_id."""
        row = self.conn.execute(
            "SELECT p.local_id, p.title, p.abstract, p.year, p.paper_type, p.status, "
            "pi.doi, pi.arxiv, pi.s2_id, pi.openalex_id "
            "FROM papers p LEFT JOIN paper_ids pi ON p.local_id = pi.local_id "
            "WHERE p.local_id = ?",
            (local_id,),
        ).fetchone()
        if not row:
            return None
        cols = [
            "local_id",
            "title",
            "abstract",
            "year",
            "paper_type",
            "status",
            "doi",
            "arxiv",
            "s2_id",
            "openalex_id",
        ]
        return dict(zip(cols, row))

    def get_concepts_by_paper(self, local_id: str) -> list[dict]:
        """Get all concepts for a paper."""
        rows = self.conn.execute(
            "SELECT concept_id, type, label, confidence FROM concepts WHERE local_id = ?",
            (local_id,),
        ).fetchall()
        return [dict(zip(["concept_id", "type", "label", "confidence"], row)) for row in rows]

    def get_all_seeds(self) -> list[dict]:
        """Return all research seeds."""
        rows = self.conn.execute(
            "SELECT seed_id, pattern_type, description, confidence, created_at FROM research_seeds"
        ).fetchall()
        cols = ["seed_id", "pattern_type", "description", "confidence", "created_at"]
        return [dict(zip(cols, row)) for row in rows]

    def delete_seed(self, seed_id: int) -> None:
        """Delete a research seed."""
        self.conn.execute("DELETE FROM research_seeds WHERE seed_id = ?", (seed_id,))

    def insert_argument(
        self,
        source_paper: str,
        claim: str,
        claim_type: str,
        target_label: str,
        target_type: str,
        evidence_type: str | None = None,
        evidence_detail: str | None = None,
        mechanism: str = "",
        confidence: float = 1.0,
        section: str = "",
    ) -> int:
        """Insert an argument unit. Returns arg_id."""
        cur = self.conn.execute(
            "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, "
            "evidence_type, evidence_detail, mechanism, section, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_paper,
                claim,
                claim_type,
                target_label,
                target_type,
                evidence_type,
                evidence_detail,
                mechanism,
                section,
                confidence,
            ),
        )
        return cur.lastrowid

    def insert_queue_item(
        self, source_paper: str, item_type: str, item_data: str, confidence: float
    ) -> int:
        """Insert a confidence queue item. Returns queue_id."""
        cur = self.conn.execute(
            "INSERT INTO confidence_queue (source_paper, item_type, item_data, confidence, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (source_paper, item_type, item_data, confidence),
        )
        return cur.lastrowid

    def accept_queue_item(self, queue_id: int) -> None:
        """Mark queue item as accepted."""
        self.conn.execute(
            "UPDATE confidence_queue SET status = 'accepted' WHERE queue_id = ?", (queue_id,)
        )

    def reject_queue_item(self, queue_id: int) -> None:
        """Mark queue item as rejected."""
        self.conn.execute(
            "UPDATE confidence_queue SET status = 'rejected' WHERE queue_id = ?", (queue_id,)
        )

    def get_queue_pending(self) -> list[dict]:
        """Return all pending queue items."""
        rows = self.conn.execute(
            "SELECT queue_id, source_paper, item_type, item_data, confidence, created_at "
            "FROM confidence_queue WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
        cols = ["queue_id", "source_paper", "item_type", "item_data", "confidence", "created_at"]
        return [dict(zip(cols, row)) for row in rows]

    def get_arguments_by_paper(self, local_id: str) -> list[dict]:
        """Get all arguments for a paper."""
        rows = self.conn.execute(
            "SELECT arg_id, claim, claim_type, target_label, target_type, "
            "evidence_type, evidence_detail, mechanism, confidence "
            "FROM arguments WHERE source_paper = ?",
            (local_id,),
        ).fetchall()
        cols = [
            "arg_id",
            "claim",
            "claim_type",
            "target_label",
            "target_type",
            "evidence_type",
            "evidence_detail",
            "mechanism",
            "confidence",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def delete_paper(self, local_id: str) -> dict:
        """Delete a paper and all associated data. Returns counts of deleted items."""
        concept_count = self.conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE local_id = ?", (local_id,)
        ).fetchone()[0]
        arg_count = self.conn.execute(
            "SELECT COUNT(*) FROM arguments WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]
        edge_count = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?", (local_id, local_id)
        ).fetchone()[0]
        queue_count = self.conn.execute(
            "SELECT COUNT(*) FROM confidence_queue WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]

        self.conn.execute("DELETE FROM concepts WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM arguments WHERE source_paper = ?", (local_id,))
        self.conn.execute("DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (local_id, local_id))
        self.conn.execute("DELETE FROM paper_ids WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM confidence_queue WHERE source_paper = ?", (local_id,))
        self.conn.execute("DELETE FROM papers WHERE local_id = ?", (local_id,))
        self.commit()

        return {
            "concepts": concept_count,
            "arguments": arg_count,
            "edges": edge_count,
            "queue_items": queue_count,
        }

    def detect_evolution_signals(self) -> list[dict]:
        """Detect evolution signals across all concepts.

        Signals per Spec §15:
        - emerging: first_seen in last 2 years, paper_count growing (year-over-year increase)
        - established: paper_count > 10, avg_confidence > 0.8
        - declining: last_seen > 3 years ago, paper_count plateau (no growth in final period)
        - contested: avg_confidence < 0.7, paper_count > 5
        - resurging: dormant > 3 years (gap in timeline), then new papers in last 2 years
        """
        from datetime import datetime

        current_year = datetime.now().year

        rows = self.conn.execute(
            "SELECT c.label, c.type, MIN(p.year) as first_seen, MAX(p.year) as last_seen, "
            "COUNT(DISTINCT c.local_id) as paper_count, AVG(c.confidence) as avg_conf "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE p.year IS NOT NULL "
            "GROUP BY c.label, c.type"
        ).fetchall()

        signals = []
        for label, ctype, first_seen, last_seen, paper_count, avg_conf in rows:
            signal = self._classify_signal(
                label,
                ctype,
                first_seen,
                last_seen,
                paper_count,
                avg_conf,
                current_year,
            )
            signals.append(
                {
                    "label": label,
                    "type": ctype,
                    "signal": signal,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "paper_count": paper_count,
                    "avg_confidence": round(avg_conf, 3),
                }
            )
        return signals

    def _classify_signal(
        self,
        label: str,
        ctype: str,
        first_seen: int,
        last_seen: int,
        paper_count: int,
        avg_conf: float,
        current_year: int,
    ) -> str:
        """Classify a single concept's evolution signal."""
        # Check contested first (overrides established for high-count low-conf)
        if paper_count > 5 and avg_conf < 0.7:
            return "contested"

        # Check resurging: dormant > 3 years then recent activity
        if self._has_resurgence(label, current_year):
            return "resurging"

        # Check emerging: recent first appearance with growing trend
        if first_seen >= current_year - 2 and self._is_growing(label, current_year):
            return "emerging"

        # Check declining: last_seen > 3 years ago (strictly more than 3 year gap)
        if last_seen < current_year - 3:
            return "declining"

        # Check established
        if paper_count > 10 and avg_conf > 0.8:
            return "established"

        return "unknown"

    def _has_resurgence(self, label: str, current_year: int) -> bool:
        """Check if concept has a gap > 3 years followed by recent activity."""
        rows = self.conn.execute(
            "SELECT DISTINCT p.year FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL ORDER BY p.year",
            (label,),
        ).fetchall()
        years = sorted([r[0] for r in rows])
        if len(years) < 2:
            return False

        # Check for gap > 3 years
        has_gap = False
        for i in range(1, len(years)):
            if years[i] - years[i - 1] > 3:
                has_gap = True
                break
        if not has_gap:
            return False

        # Must have recent activity (last 2 years)
        return years[-1] >= current_year - 1

    def _is_growing(self, label: str, current_year: int) -> bool:
        """Check if paper count for concept is growing (recent > early)."""
        rows = self.conn.execute(
            "SELECT p.year, COUNT(*) as cnt FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL GROUP BY p.year ORDER BY p.year",
            (label,),
        ).fetchall()
        if len(rows) < 2:
            return False  # Need at least 2 years to determine growth trend

        mid = len(rows) // 2
        early_avg = sum(r[1] for r in rows[:mid]) / mid
        late_avg = sum(r[1] for r in rows[mid:]) / (len(rows) - mid)
        return late_avg > early_avg

    def get_concept_signal(self, label: str) -> dict | None:
        """Detect evolution signal for a specific concept."""
        from datetime import datetime

        current_year = datetime.now().year

        row = self.conn.execute(
            "SELECT c.label, c.type, MIN(p.year), MAX(p.year), "
            "COUNT(DISTINCT c.local_id), AVG(c.confidence) "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL "
            "GROUP BY c.label, c.type",
            (label,),
        ).fetchone()
        if row is None:
            return None

        lbl, ctype, first_seen, last_seen, paper_count, avg_conf = row
        signal = self._classify_signal(
            lbl,
            ctype,
            first_seen,
            last_seen,
            paper_count,
            avg_conf,
            current_year,
        )
        return {
            "label": lbl,
            "type": ctype,
            "signal": signal,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "paper_count": paper_count,
            "avg_confidence": round(avg_conf, 3),
        }

    def get_concept_evolution(self, label: str) -> list[dict]:
        """Get year-by-year usage stats for a concept label with trend annotation."""
        rows = self.conn.execute(
            "SELECT p.year, COUNT(*) as count, AVG(c.confidence) as avg_conf "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL "
            "GROUP BY p.year ORDER BY p.year",
            (label,),
        ).fetchall()
        result = []
        prev_count = None
        for i, row in enumerate(rows):
            year, count, avg_conf = row
            entry = {"year": year, "count": count, "avg_conf": round(avg_conf, 2)}

            if i == 0:
                entry["trend"] = "first_appeared"
            elif prev_count is not None:
                if count > prev_count:
                    entry["trend"] = "growing"
                elif count < prev_count:
                    entry["trend"] = "declining"
                else:
                    entry["trend"] = "stable"
            else:
                entry["trend"] = "stable"

            prev_count = count
            result.append(entry)
        return result
