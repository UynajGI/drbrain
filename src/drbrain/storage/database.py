"""SQLite backend with schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    local_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT DEFAULT '',
    year INTEGER,
    paper_type TEXT NOT NULL DEFAULT 'paper'
        CHECK(paper_type IN ('paper','review','thesis','preprint','book','document')),
    status TEXT NOT NULL DEFAULT 'placeholder' CHECK(status IN ('uploaded', 'placeholder', 'merged', 'extracted')),
    journal TEXT DEFAULT '',
    publisher TEXT DEFAULT '',
    citation_count INTEGER DEFAULT 0,
    volume TEXT DEFAULT '',
    pages TEXT DEFAULT '',
    authors TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    node_id TEXT DEFAULT '',
    first_seen INTEGER,
    last_seen INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    node_id TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (src_id, dst_id, relation, source_paper)
);

CREATE TABLE IF NOT EXISTS aliases (
    variant TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    entity TEXT PRIMARY KEY,
    vec BLOB NOT NULL,
    dim INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tree_vectors (
    node_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    tree_layer TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tree_summaries (
    node_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    summary_text TEXT NOT NULL DEFAULT '',
    source_node_ids TEXT NOT NULL DEFAULT '',
    tree_layer INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vector_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_source_paper ON edges(source_paper);
CREATE INDEX IF NOT EXISTS idx_queue_status ON confidence_queue(status);
CREATE INDEX IF NOT EXISTS idx_concepts_local_id ON concepts(local_id);
CREATE INDEX IF NOT EXISTS idx_tree_vectors_paper ON tree_vectors(paper_id);
CREATE INDEX IF NOT EXISTS idx_tree_vectors_layer_paper ON tree_vectors(tree_layer, paper_id);
CREATE INDEX IF NOT EXISTS idx_tree_summaries_paper ON tree_summaries(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_ids_local ON paper_ids(local_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity);
-- v8 change_tracking indexes (updated_at/status) are created by _migrate_add_change_tracking
-- so that pre-v8 DBs can ALTER TABLE first, then index. Do not add them here.

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
CREATE INDEX IF NOT EXISTS idx_citation_cache_target ON citation_cache(target_title);

CREATE TABLE IF NOT EXISTS build_stages (
    paper_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (paper_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_build_stages_paper_stage ON build_stages(paper_id, stage);

CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active','archived','deleted')),
    model_config TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_messages (
    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT DEFAULT '',
    tool_calls_json TEXT DEFAULT '',
    tool_call_id TEXT DEFAULT '',
    tool_name TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_session
    ON agent_messages(session_id, seq);
"""


class Database:
    """Thin SQLite wrapper with schema auto-init."""

    def __init__(self, db_path: str | Path = "data/drbrain.db"):
        """Open SQLite database at *db_path*, enabling WAL mode and auto-migrating schema."""
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Apply pending schema migrations in order."""
        current = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_versions"
        ).fetchone()[0]

        migrations = [
            (1, "paper_type", self._migrate_add_paper_type),
            (2, "venue_columns", self._migrate_add_venue_columns),
            (3, "authors", self._migrate_add_authors),
            (4, "node_id", self._migrate_add_node_id),
            (5, "edge_provenance", self._migrate_add_edge_provenance),
            (6, "agent_sessions", self._migrate_add_agent_sessions),
            (7, "indexes_v2", self._migrate_add_indexes_v2),
            (8, "change_tracking", self._migrate_add_change_tracking),
        ]

        for version, name, fn in migrations:
            if current < version:
                logger.info("[db] applying migration v%d: %s", version, name)
                fn()
                self.conn.execute(
                    "INSERT OR IGNORE INTO schema_versions (version) VALUES (?)",
                    (version,),
                )
                self.conn.commit()
                logger.info("[db] migration v%d (%s) applied", version, name)
        if current >= len(migrations):
            logger.debug("[db] schema up to date (v%d)", current)

    def _migrate_add_paper_type(self) -> None:
        """Add paper_type column if missing (pre-v2 DBs)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "paper_type" not in cols:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN paper_type TEXT NOT NULL DEFAULT 'paper'"
            )

    def _migrate_add_venue_columns(self) -> None:
        """Add journal, publisher, citation_count, volume, pages columns if missing."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "journal" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN journal TEXT DEFAULT ''")
        if "publisher" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN publisher TEXT DEFAULT ''")
        if "citation_count" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN citation_count INTEGER DEFAULT 0")
        if "volume" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN volume TEXT DEFAULT ''")
        if "pages" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN pages TEXT DEFAULT ''")

    def _migrate_add_authors(self) -> None:
        """Add authors column if missing."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "authors" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN authors TEXT DEFAULT ''")

    def _migrate_add_node_id(self) -> None:
        """Add node_id columns to concepts and arguments for tree provenance."""
        concept_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(concepts)").fetchall()]
        if "node_id" not in concept_cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN node_id TEXT DEFAULT ''")
        arg_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(arguments)").fetchall()]
        if "node_id" not in arg_cols:
            self.conn.execute("ALTER TABLE arguments ADD COLUMN node_id TEXT DEFAULT ''")

    def _migrate_add_edge_provenance(self) -> None:
        """Add node_id and section columns to edges for provenance chain."""
        edge_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(edges)").fetchall()]
        if "node_id" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN node_id TEXT DEFAULT ''")
        if "section" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN section TEXT DEFAULT ''")

    def _migrate_add_agent_sessions(self) -> None:
        """Add agent_sessions and agent_messages tables (created via SCHEMA_SQL IF NOT EXISTS)."""
        pass  # Tables created by SCHEMA_SQL on init; this migration marks v6 as applied.

    def _migrate_add_indexes_v2(self) -> None:
        """Create performance indexes that reference columns added in earlier migrations."""
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_concepts_node_id ON concepts(node_id)"
            )
        except sqlite3.OperationalError:
            pass  # Column may not exist in very old schemas

    def _migrate_add_change_tracking(self) -> None:
        """Add updated_at columns to papers/concepts/edges for incremental updates.

        SQLite forbids non-constant DEFAULTs on ALTER TABLE ADD COLUMN, so we
        add the column as nullable then backfill with CURRENT_TIMESTAMP. New DBs
        get the column with a proper DEFAULT via SCHEMA_SQL instead.
        """
        # papers.updated_at
        paper_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "updated_at" not in paper_cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN updated_at TIMESTAMP")
            self.conn.execute("UPDATE papers SET updated_at = CURRENT_TIMESTAMP")
        # concepts.updated_at
        concept_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(concepts)").fetchall()]
        if "updated_at" not in concept_cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN updated_at TIMESTAMP")
            self.conn.execute("UPDATE concepts SET updated_at = CURRENT_TIMESTAMP")
        # edges.updated_at
        edge_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(edges)").fetchall()]
        if "updated_at" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN updated_at TIMESTAMP")
            self.conn.execute("UPDATE edges SET updated_at = CURRENT_TIMESTAMP")
        # Indexes (safe even if columns pre-existed). status index is guarded
        # because very old / synthetic schemas may lack the column.
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_updated_at ON papers(updated_at)")
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status)")
        except sqlite3.OperationalError:
            pass
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_updated_at ON edges(updated_at)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_concepts_updated_at ON concepts(updated_at)"
        )

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement and return the cursor."""
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        return self.conn.executemany(sql, seq)

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
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
        journal: str = "",
        publisher: str = "",
        citation_count: int = 0,
        volume: str = "",
        pages: str = "",
        authors: str = "",
    ) -> None:
        """Insert or ignore a paper record with full metadata fields.

        On conflict (existing local_id), bump updated_at to signal downstream
        incremental stages that this paper changed.
        """
        self.conn.execute(
            "INSERT INTO papers (local_id, title, year, status, paper_type, "
            "journal, publisher, citation_count, volume, pages, authors, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(local_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP",
            (
                local_id,
                title,
                year,
                status,
                paper_type,
                journal,
                publisher,
                citation_count,
                volume,
                pages,
                authors,
            ),
        )

    def insert_paper_ids(
        self, local_id: str, doi=None, arxiv=None, s2_id=None, openalex_id=None
    ) -> None:
        """Insert or ignore external identifier mappings for a paper."""
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_ids (local_id, doi, arxiv, s2_id, openalex_id) VALUES (?, ?, ?, ?, ?)",
            (local_id, doi, arxiv, s2_id, openalex_id),
        )

    def set_paper_abstract(self, local_id: str, abstract: str) -> None:
        """Update the abstract text for a paper."""
        self.conn.execute(
            "UPDATE papers SET abstract = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (abstract, local_id),
        )

    def upgrade_placeholder(self, local_id: str) -> None:
        """Promote a placeholder paper to uploaded status."""
        self.conn.execute(
            "UPDATE papers SET status = 'uploaded', updated_at = CURRENT_TIMESTAMP "
            "WHERE local_id = ? AND status = 'placeholder'",
            (local_id,),
        )

    def set_paper_status(self, local_id: str, status: str) -> None:
        """Update paper status and bump updated_at."""
        self.conn.execute(
            "UPDATE papers SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (status, local_id),
        )

    def touch_paper(self, local_id: str) -> None:
        """Bump updated_at timestamp on a paper to signal downstream stages."""
        self.conn.execute(
            "UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (local_id,),
        )

    def touch_edge(self, src_id: str, dst_id: str, relation: str, source_paper: str) -> None:
        """Bump updated_at on an edge to signal downstream stages."""
        self.conn.execute(
            "UPDATE edges SET updated_at = CURRENT_TIMESTAMP "
            "WHERE src_id = ? AND dst_id = ? AND relation = ? AND source_paper = ?",
            (src_id, dst_id, relation, source_paper),
        )

    def update_paper_venue(
        self,
        local_id: str,
        title: str = "",
        year: int | None = None,
        journal: str = "",
        publisher: str = "",
        citation_count: int = 0,
    ) -> None:
        """Update paper metadata after ingest (for upgraded placeholders)."""
        self.conn.execute(
            "UPDATE papers SET title = ?, year = ?, journal = ?, publisher = ?, "
            "citation_count = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (title, year, journal, publisher, citation_count, local_id),
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
        node_id: str = "",
    ) -> int:
        """Insert a concept with temporal tracking. Returns concept_id."""
        cur = self.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section, node_id, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (local_id, ctype, label, confidence, section, node_id, year, year),
        )
        return cur.lastrowid

    def insert_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        source_paper: str,
        weight: float = 1.0,
        node_id: str = "",
        section: str = "",
    ) -> None:
        """Insert an edge between concepts with tree provenance.

        On conflict (duplicate PK), bump updated_at so downstream incremental
        stages notice the edge was re-asserted.
        """
        self.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight, node_id, "
            "section, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(src_id, dst_id, relation, source_paper) "
            "DO UPDATE SET updated_at = CURRENT_TIMESTAMP, weight = excluded.weight",
            (src_id, dst_id, relation, source_paper, weight, node_id, section),
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

    # ── Centralized write helpers (SQL-leak consolidation) ──────────────
    # These methods exist so callers outside storage/ never need to write raw
    # SQL. They also enforce invariants (e.g. bumping updated_at, atomic
    # merges) that ad-hoc SQL bypassed.

    _VALID_EXTERNAL_IDS = ("doi", "arxiv", "s2_id", "openalex_id")

    def set_external_id(self, local_id: str, kind: str, value: str | None) -> None:
        """Update a single external identifier (doi/arxiv/s2_id/openalex_id).

        Raises ValueError for unknown kinds. Bumps the paper's updated_at so
        the change is visible to incremental stages.
        """
        if kind not in self._VALID_EXTERNAL_IDS:
            raise ValueError(f"unknown external id kind: {kind}")
        self.conn.execute(f"UPDATE paper_ids SET {kind} = ? WHERE local_id = ?", (value, local_id))
        self.touch_paper(local_id)

    def insert_citation_cache(
        self,
        source_paper: str,
        target_title: str,
        target_year: int | None,
        relation: str,
        target_doi: str | None = None,
        target_s2_id: str | None = None,
    ) -> None:
        """Insert a citation_cache row (idempotent on PK)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO citation_cache "
            "(source_paper, target_title, target_year, relation, target_doi, target_s2_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_paper, target_title, target_year, relation, target_doi, target_s2_id),
        )

    _VALID_PAPER_FIELDS = (
        "title",
        "abstract",
        "year",
        "paper_type",
        "journal",
        "publisher",
        "citation_count",
        "volume",
        "pages",
        "authors",
        "status",
    )

    def set_paper_field(self, local_id: str, field: str, value) -> None:
        """Update a single papers column by name.

        Allows callers (e.g. repair.py) to set one field without rewriting the
        whole row. Bumps updated_at. ``field`` is validated against an
        allowlist to prevent SQL injection via column names.
        """
        if field not in self._VALID_PAPER_FIELDS:
            raise ValueError(f"unknown paper field: {field}")
        self.conn.execute(
            f"UPDATE papers SET {field} = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (value, local_id),
        )

    def set_paper_type(self, local_id: str, paper_type: str) -> None:
        """Update paper_type and bump updated_at."""
        self.set_paper_field(local_id, "paper_type", paper_type)

    def delete_concept(self, concept_id: int) -> None:
        """Delete a single concept row by concept_id."""
        self.conn.execute("DELETE FROM concepts WHERE concept_id = ?", (concept_id,))

    def redirect_edge_endpoint(self, old_label: str, new_label: str) -> int:
        """Rewrite edges referencing ``old_label`` to ``new_label``.

        Used by concept-merge to retarget src_id/dst_id. Returns the number of
        rows touched. Each updated edge also gets updated_at bumped so the
        change is visible to incremental closure/embed.
        """
        n = 0
        cur = self.conn.execute(
            "UPDATE edges SET src_id = ?, updated_at = CURRENT_TIMESTAMP WHERE src_id = ?",
            (new_label, old_label),
        )
        n += cur.rowcount
        cur = self.conn.execute(
            "UPDATE edges SET dst_id = ?, updated_at = CURRENT_TIMESTAMP WHERE dst_id = ?",
            (new_label, old_label),
        )
        n += cur.rowcount
        return n

    def accept_queue_by_label(self, label: str) -> int:
        """Accept all pending queue items whose item_data contains ``label``.

        Returns the number of items accepted.
        """
        cur = self.conn.execute(
            "UPDATE confidence_queue SET status = 'accepted' "
            "WHERE status = 'pending' AND item_data LIKE ?",
            (f"%{label}%",),
        )
        return cur.rowcount

    def upsert_build_stage(
        self, paper_id: str, stage: str, status: str, result_json: str = ""
    ) -> None:
        """Insert or replace a build_stages row."""
        self.conn.execute(
            "INSERT OR REPLACE INTO build_stages (paper_id, stage, status, result_json, updated_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (paper_id, stage, status, result_json),
        )

    def merge_papers(self, keep_id: str, merge_id: str) -> dict:
        """Merge two paper records atomically, keeping ``keep_id``.

        Migrates concepts, arguments, and edges from merge_id onto keep_id,
        then deletes merge_id. Runs as a single transaction so a failure at any
        point rolls back — no torn state. Returns a dict of migrated counts.

        Note: unlike delete_paper(), this does NOT touch neighbors or clear
        closure/embed watermarks — the merge target (keep_id) already
        represents the surviving identity, and edges now point at it.
        """
        counts = {
            "concepts": 0,
            "arguments": 0,
            "edges_redirected": 0,
        }
        try:
            self.conn.execute("BEGIN")
            cur = self.conn.execute(
                "UPDATE concepts SET local_id = ? WHERE local_id = ?", (keep_id, merge_id)
            )
            counts["concepts"] = cur.rowcount
            cur = self.conn.execute(
                "UPDATE arguments SET source_paper = ? WHERE source_paper = ?",
                (keep_id, merge_id),
            )
            counts["arguments"] = cur.rowcount
            cur = self.conn.execute(
                "UPDATE edges SET source_paper = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE source_paper = ?",
                (keep_id, merge_id),
            )
            counts["edges_redirected"] = cur.rowcount
            self.conn.execute(
                "UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
                (keep_id,),
            )
            self.conn.execute("DELETE FROM papers WHERE local_id = ?", (merge_id,))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return counts

    # ── agent_sessions / agent_messages ─────────────────────────────────

    def insert_agent_session(
        self,
        session_id: str,
        title: str = "",
        system_prompt: str = "",
        model_config: str = "{}",
    ) -> None:
        """Create a new agent session row."""
        self.conn.execute(
            "INSERT INTO agent_sessions (session_id, title, system_prompt, model_config) "
            "VALUES (?, ?, ?, ?)",
            (session_id, title, system_prompt, model_config),
        )

    def soft_delete_session(self, session_id: str) -> None:
        """Mark an agent session as deleted (soft delete)."""
        self.conn.execute(
            "UPDATE agent_sessions SET status = 'deleted', updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = ?",
            (session_id,),
        )

    def touch_session(self, session_id: str) -> None:
        """Bump an agent session's updated_at."""
        self.conn.execute(
            "UPDATE agent_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )

    def insert_agent_message(
        self,
        session_id: str,
        seq: int,
        role: str,
        content: str = "",
        tool_calls_json: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
    ) -> None:
        """Append a message to an agent session."""
        self.conn.execute(
            "INSERT INTO agent_messages "
            "(session_id, seq, role, content, tool_calls_json, tool_call_id, tool_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, seq, role, content, tool_calls_json, tool_call_id, tool_name),
        )

    # -- Embeddings --

    def save_embedding(self, entity: str, vec, dim: int) -> None:
        """Persist a TransE entity/relation vector to the embeddings table."""
        import numpy as np

        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (entity, vec, dim) VALUES (?, ?, ?)",
            (entity, np.array(vec, dtype=np.float32).tobytes(), dim),
        )

    def load_embeddings(self) -> dict:
        """Load all entity/relation vectors into a dict keyed by entity label."""
        import numpy as np

        rows = self.conn.execute("SELECT entity, vec, dim FROM embeddings").fetchall()
        return {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}

    def clear_embeddings(self) -> None:
        """Delete all embeddings from the table (used before re-training)."""
        self.conn.execute("DELETE FROM embeddings")

    # -- Query helpers --

    def get_all_papers(self) -> list[dict]:
        """Return all papers as list of dicts."""
        rows = self.conn.execute(
            "SELECT p.local_id, p.title, p.abstract, p.year, p.paper_type, p.status, "
            "p.journal, p.publisher, p.citation_count, p.volume, p.pages, p.authors, p.created_at, "
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
            "journal",
            "publisher",
            "citation_count",
            "volume",
            "pages",
            "authors",
            "created_at",
            "doi",
            "arxiv",
            "s2_id",
            "openalex_id",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_dirty_papers(self) -> list[str]:
        """Return local_ids of papers needing (re)building.

        A paper is dirty if its status is not 'extracted' (never built or
        explicitly marked for rebuild) OR was touched after extraction.
        """
        rows = self.conn.execute(
            "SELECT local_id FROM papers WHERE status != 'extracted' ORDER BY updated_at"
        ).fetchall()
        return [r[0] for r in rows]

    def get_papers_since(self, ts: str | None) -> list[str]:
        """Return local_ids of papers with updated_at > ts.

        If ts is None, returns all papers (first run).
        """
        if ts is None:
            rows = self.conn.execute("SELECT local_id FROM papers").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT local_id FROM papers WHERE updated_at > ?", (ts,)
            ).fetchall()
        return [r[0] for r in rows]

    def get_paper_timestamp(self, local_id: str) -> str | None:
        """Return the updated_at timestamp of a paper, or None if not found."""
        row = self.conn.execute(
            "SELECT updated_at FROM papers WHERE local_id = ?", (local_id,)
        ).fetchone()
        return row[0] if row else None

    def get_max_paper_timestamp(self) -> str | None:
        """Return the max updated_at across all papers, or None if empty."""
        row = self.conn.execute("SELECT MAX(updated_at) FROM papers").fetchone()
        return row[0] if row and row[0] is not None else None

    def get_last_run(self, name: str) -> str | None:
        """Return the timestamp of the last successful run of a named stage.

        Stored in vector_metadata with key 'last_run:<name>'.
        """
        row = self.conn.execute(
            "SELECT value FROM vector_metadata WHERE key = ?", (f"last_run:{name}",)
        ).fetchone()
        return row[0] if row else None

    def set_last_run(self, name: str, ts: str | None = None) -> None:
        """Record the timestamp of a successful run of a named stage.

        Defaults to CURRENT_TIMESTAMP. Stored in vector_metadata.
        """
        if ts is None:
            ts_expr = "CURRENT_TIMESTAMP"
            self.conn.execute(
                "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES (?, " + ts_expr + ")",
                (f"last_run:{name}",),
            )
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES (?, ?)",
                (f"last_run:{name}", ts),
            )

    def get_paper(self, local_id: str) -> dict | None:
        """Get a single paper by local_id."""
        row = self.conn.execute(
            "SELECT p.local_id, p.title, p.abstract, p.year, p.paper_type, p.status, "
            "p.journal, p.publisher, p.citation_count, p.volume, p.pages, p.authors, "
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
            "journal",
            "publisher",
            "citation_count",
            "volume",
            "pages",
            "authors",
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
        node_id: str = "",
    ) -> int:
        """Insert an argument unit. Returns arg_id."""
        cur = self.conn.execute(
            "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, "
            "evidence_type, evidence_detail, mechanism, section, node_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                node_id,
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
        """Delete a paper and all associated data. Returns counts of deleted items.

        To keep downstream incremental stages consistent, neighbor papers
        sharing edges with this paper's concepts are touched (updated_at bumped)
        and the closure/embed/index watermarks are cleared, so the next pipeline
        run re-evaluates them instead of skipping.
        """
        # Collect this paper's concept labels BEFORE deletion so we can find
        # neighbor papers that shared edges with them.
        labels = [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT label FROM concepts WHERE local_id = ?", (local_id,)
            ).fetchall()
        ]

        concept_count = self.conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE local_id = ?", (local_id,)
        ).fetchone()[0]
        arg_count = self.conn.execute(
            "SELECT COUNT(*) FROM arguments WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]
        # edges.src_id/dst_id hold concept labels, not paper ids. A paper's
        # edges are those it asserted (source_paper) plus closure-inferred edges
        # whose endpoints reference its concepts.
        edge_count = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]
        queue_count = self.conn.execute(
            "SELECT COUNT(*) FROM confidence_queue WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]

        self.conn.execute("DELETE FROM concepts WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM arguments WHERE source_paper = ?", (local_id,))
        # Delete edges this paper asserted. (Closure-inferred edges with
        # source_paper='closure' are left for the next closure run to re-evaluate.)
        self.conn.execute("DELETE FROM edges WHERE source_paper = ?", (local_id,))
        self.conn.execute("DELETE FROM paper_ids WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM confidence_queue WHERE source_paper = ?", (local_id,))
        self.conn.execute("DELETE FROM tree_vectors WHERE paper_id = ?", (local_id,))
        self.conn.execute("DELETE FROM tree_summaries WHERE paper_id = ?", (local_id,))
        self.conn.execute("DELETE FROM papers WHERE local_id = ?", (local_id,))

        # Touch neighbor papers that shared edges with the deleted concepts so
        # the next closure/embed pass re-evaluates them.
        touched_neighbors = 0
        if labels:
            placeholders = ",".join("?" * len(labels))
            # Exclude the just-deleted paper and any already-removed ids
            neighbor_ids = {
                r[0]
                for r in self.conn.execute(
                    f"SELECT DISTINCT source_paper FROM edges "
                    f"WHERE source_paper != ? "
                    f"AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
                    (local_id, *labels, *labels),
                ).fetchall()
            }
            for nid in neighbor_ids:
                self.touch_paper(nid)
                touched_neighbors += 1

        # Invalidate stage watermarks so the next run doesn't skip the cleanup
        for stage in ("closure", "embed", "index"):
            self.conn.execute("DELETE FROM vector_metadata WHERE key = ?", (f"last_run:{stage}",))

        self.commit()

        return {
            "concepts": concept_count,
            "arguments": arg_count,
            "edges": edge_count,
            "queue_items": queue_count,
            "touched_neighbors": touched_neighbors,
        }

    # ── Temporal evolution signals ──────────────────────────────

    def detect_evolution_signals(self) -> list[dict]:
        """Detect evolution signals across all concepts.

        Signals:
        - emerging: first_seen in last 2 years, paper_count growing
        - established: paper_count > 10, avg_confidence > 0.8
        - declining: last_seen > 3 years ago
        - contested: avg_confidence < 0.7, paper_count > 5
        - resurging: dormant > 3 years, then new papers in last 2 years
        """
        from collections import defaultdict
        from datetime import datetime

        current_year = datetime.now().year

        rows = self.conn.execute(
            "SELECT c.label, c.type, MIN(p.year) as first_seen, MAX(p.year) as last_seen, "
            "COUNT(DISTINCT c.local_id) as paper_count, AVG(c.confidence) as avg_conf "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE p.year IS NOT NULL "
            "GROUP BY c.label, c.type"
        ).fetchall()

        # Batch-preload (label, type) → {year: count} to eliminate N+1 queries.
        # Previously _has_resurgence / _is_growing each ran a SQL query per label;
        # with L labels this was up to 2L extra queries. Now it's a single query.
        year_rows = self.conn.execute(
            "SELECT c.label, c.type, p.year, COUNT(*) as cnt "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE p.year IS NOT NULL "
            "GROUP BY c.label, c.type, p.year"
        ).fetchall()
        label_years: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        for lbl, ctype, year, cnt in year_rows:
            label_years[(lbl, ctype)][year] = cnt

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
                label_years=label_years.get((label, ctype)),
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
        *,
        label_years: dict[int, int] | None = None,
    ) -> str:
        if paper_count > 5 and avg_conf < 0.7:
            return "contested"
        if self._has_resurgence(label, current_year, label_years=label_years):
            return "resurging"
        if first_seen >= current_year - 2 and self._is_growing(
            label, current_year, label_years=label_years
        ):
            return "emerging"
        if last_seen < current_year - 3:
            return "declining"
        if paper_count > 10 and avg_conf > 0.8:
            return "established"
        return "unknown"

    def _has_resurgence(
        self,
        label: str,
        current_year: int,
        *,
        label_years: dict[int, int] | None = None,
    ) -> bool:
        if label_years is not None:
            years = sorted(label_years.keys())
        else:
            # Fallback: single-label query (used by get_concept_signal)
            rows = self.conn.execute(
                "SELECT DISTINCT p.year FROM concepts c JOIN papers p ON c.local_id = p.local_id "
                "WHERE c.label = ? AND p.year IS NOT NULL ORDER BY p.year",
                (label,),
            ).fetchall()
            years = sorted([r[0] for r in rows])
        if len(years) < 2:
            return False
        has_gap = any(years[i] - years[i - 1] > 3 for i in range(1, len(years)))
        return has_gap and years[-1] >= current_year - 1

    def _is_growing(
        self,
        label: str,
        current_year: int,
        *,
        label_years: dict[int, int] | None = None,
    ) -> bool:
        if label_years is not None:
            rows = sorted(label_years.items())  # [(year, count), ...]
        else:
            # Fallback: single-label query (used by get_concept_signal)
            rows = self.conn.execute(
                "SELECT p.year, COUNT(*) as cnt FROM concepts c JOIN papers p ON c.local_id = p.local_id "
                "WHERE c.label = ? AND p.year IS NOT NULL GROUP BY p.year ORDER BY p.year",
                (label,),
            ).fetchall()
        if len(rows) < 2:
            return False
        mid = len(rows) // 2
        early_avg = sum(r[1] for r in rows[:mid]) / mid
        late_avg = sum(r[1] for r in rows[mid:]) / (len(rows) - mid)
        return late_avg > early_avg

    def get_concept_signal(self, label: str) -> dict | None:
        """Classify a concept's temporal signal (emerging/established/declining/etc.)."""
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
        """Return year-by-year concept frequency with trend annotations."""
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

    # -- Stats queries --

    def get_stats(self, paper_ids: list[str] | None = None) -> dict:
        """Return aggregate counts for dashboard/stats display.

        When *paper_ids* is provided, counts for papers, concepts, edges,
        and arguments are filtered to those paper IDs.  Global tables
        (aliases, research_seeds, confidence_queue) always return total counts.
        """
        stats: dict = {}

        if paper_ids:
            ph = ",".join("?" for _ in paper_ids)
            params = tuple(paper_ids)

            stats["papers"] = self.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE local_id IN ({ph})", params
            ).fetchone()[0]
            stats["uploaded"] = self.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE status='uploaded' AND local_id IN ({ph})",
                params,
            ).fetchone()[0]
            stats["placeholders"] = self.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE status='placeholder' AND local_id IN ({ph})",
                params,
            ).fetchone()[0]
            stats["concepts"] = self.conn.execute(
                f"SELECT COUNT(*) FROM concepts WHERE local_id IN ({ph})", params
            ).fetchone()[0]
            stats["edges"] = self.conn.execute(
                f"SELECT COUNT(*) FROM edges WHERE source_paper IN ({ph})", params
            ).fetchone()[0]
            stats["arguments"] = self.conn.execute(
                f"SELECT COUNT(*) FROM arguments WHERE source_paper IN ({ph})", params
            ).fetchone()[0]
        else:
            stats["papers"] = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            stats["uploaded"] = self.conn.execute(
                "SELECT COUNT(*) FROM papers WHERE status='uploaded'"
            ).fetchone()[0]
            stats["placeholders"] = self.conn.execute(
                "SELECT COUNT(*) FROM papers WHERE status='placeholder'"
            ).fetchone()[0]
            stats["concepts"] = self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            stats["edges"] = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            stats["arguments"] = self.conn.execute("SELECT COUNT(*) FROM arguments").fetchone()[0]

        stats["aliases"] = self.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        stats["research_seeds"] = self.conn.execute(
            "SELECT COUNT(*) FROM research_seeds"
        ).fetchone()[0]
        stats["queue_pending"] = self.conn.execute(
            "SELECT COUNT(*) FROM confidence_queue WHERE status = 'pending'"
        ).fetchone()[0]

        return stats
