"""SQLite backend with schema management."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
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
    categories TEXT DEFAULT '',
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

CREATE TABLE IF NOT EXISTS paper_cite_keys (
    citing_local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    cited_key TEXT NOT NULL,
    cited_local_id TEXT REFERENCES papers(local_id) ON DELETE SET NULL,
    PRIMARY KEY (citing_local_id, cited_key)
);
CREATE INDEX IF NOT EXISTS idx_paper_cite_keys_cited ON paper_cite_keys(cited_key);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_id TEXT NOT NULL REFERENCES papers(local_id),
    type TEXT NOT NULL CHECK(type IN ('Problem', 'Method', 'Conclusion', 'Debate', 'Gap', 'Actor')),
    label TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    section TEXT DEFAULT '',
    node_id TEXT DEFAULT '',
    provenance TEXT DEFAULT '',
    authority TEXT DEFAULT '',
    valid_from INTEGER,
    valid_to INTEGER,
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
    owner_principal TEXT DEFAULT '',
    project_id TEXT NOT NULL DEFAULT 'prj-default',
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

-- ── Concept graph layer (v9) ───────────────────────────────────
-- Provenance mapping for corpus ingested from external academic APIs
-- (Sciverse / OpenAlex / ...). Enables unique_id-first dedup + incremental ingest.
CREATE TABLE IF NOT EXISTS corpus_sources (
    local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    source_unique_id TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_unique_id)
);

-- Concept graph nodes (unique normalized concept labels + aggregated stats).
-- v10 adds concept type / noise / singleton markers (populated downstream).
CREATE TABLE IF NOT EXISTS concept_nodes (
    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL UNIQUE,
    doc_freq INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    first_year INTEGER,
    last_year INTEGER,
    type TEXT DEFAULT 'other',
    is_noise INTEGER DEFAULT 0,
    is_singleton INTEGER DEFAULT 0
);

-- Concept co-occurrence edges, timestamped by publication year. Each paper
-- contributes a clique over its concepts; weight accumulates on re-assertion.
CREATE TABLE IF NOT EXISTS concept_cooccurrence (
    src_label TEXT NOT NULL,
    dst_label TEXT NOT NULL,
    year INTEGER,
    paper_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (src_label, dst_label, year, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_year ON concept_cooccurrence(year);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_src ON concept_cooccurrence(src_label);

-- Per-concept semantic embeddings (averaged across source abstracts).
CREATE TABLE IF NOT EXISTS concept_embeddings (
    label TEXT PRIMARY KEY,
    vec BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT DEFAULT ''
);

-- Paper-level citation edges harvested from external APIs.
CREATE TABLE IF NOT EXISTS paper_citations (
    citing_id TEXT NOT NULL,
    cited_id TEXT NOT NULL,
    source TEXT DEFAULT '',
    year INTEGER,
    PRIMARY KEY (citing_id, cited_id)
);

-- Source-provided keywords / topics per paper (zero-LLM concept source).
CREATE TABLE IF NOT EXISTS paper_terms (
    local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'keyword' CHECK(kind IN ('keyword', 'topic')),
    PRIMARY KEY (local_id, term, kind)
);

-- ── Epistemic layer (v11-v13) ──────────────────────────────────
-- Knowledge snapshots: versioned, reproducible points-in-time over the graph.
-- Each snapshot pins the set of entities/claims/evidence an answer was (or
-- could be) grounded in, so answers can be re-derived against the same state.
CREATE TABLE IF NOT EXISTS knowledge_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    revision_id TEXT DEFAULT '',
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Answer records: bind every generated answer to the evidence it cites, the
-- knowledge snapshot it was grounded in, and the model/retriever versions that
-- produced it — so answers are reproducible and auditable rather than
-- opaque model output.
CREATE TABLE IF NOT EXISTS answer_records (
    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    evidence_ids TEXT DEFAULT '',
    provenance TEXT DEFAULT '',
    model_version TEXT DEFAULT '',
    snapshot_id TEXT DEFAULT '',
    retriever_version TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evidence objects (v14): first-class groundings — the paper / page / verbatim
-- snippet / numeric value / unit / experimental conditions a claim rests on,
-- plus provenance and authority. One row per distinct ``paper:node`` grounding.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    paper_id TEXT DEFAULT '',
    node_id TEXT DEFAULT '',
    page TEXT DEFAULT '',
    snippet TEXT DEFAULT '',
    value TEXT DEFAULT '',
    unit TEXT DEFAULT '',
    conditions TEXT DEFAULT '',
    provenance TEXT DEFAULT '',
    authority TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id);

-- Claims (v15): first-class assertions with a TBox type (Problem / Method /
-- Conclusion / Gap / Debate / Actor — or '' when unknown), authority,
-- provenance, confidence, and a validity window.
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    label TEXT DEFAULT '',
    claim_text TEXT NOT NULL,
    claim_type TEXT DEFAULT '',
    authority TEXT DEFAULT '',
    provenance TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    valid_from INTEGER,
    valid_to INTEGER,
    run_id TEXT DEFAULT '',
    cycle INTEGER,
    job_id TEXT DEFAULT '',
    claim_ledger_id TEXT DEFAULT '',
    model TEXT DEFAULT '',
    prompt_hash TEXT DEFAULT '',
    evidence_node_ids TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);

-- Claim/evidence bindings (v17): preserve the actual groundings behind an
-- assertion rather than relying on a synthetic statement-level evidence row.
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
    PRIMARY KEY (claim_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_evidence ON claim_evidence(evidence_id);

-- ── Project scope + WebUI layer (v21) ──────────────────────────
-- A project is the durable namespace behind the WebUI project switcher: it
-- owns a corpus reference (an optional workspace directory) and a set of
-- conversation sessions.  ``project_id`` never changes on rename; legacy
-- rows belong to the seeded default project.
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    workspace_name TEXT UNIQUE,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Login sessions for the local WebUI.  Separate from ``agent_sessions``:
-- one is browser authentication, the other is a research conversation.
CREATE TABLE IF NOT EXISTS webui_sessions (
    session_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL,
    remote_addr TEXT DEFAULT '',
    user_agent TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_webui_sessions_token ON webui_sessions(token_hash);

-- Minimal interface audit trail (login/logout/rotate).  Not a business log.
CREATE TABLE IF NOT EXISTS webui_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    detail TEXT DEFAULT '',
    remote_addr TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_webui_audit_event ON webui_audit(event, created_at);

-- Session memory entries: explicit, provenance-carrying rows.  Run-derived
-- entries are written idempotently (``dedup_key``); project-layer rows only
-- exist after an explicit promotion action.
CREATE TABLE IF NOT EXISTS session_memory (
    memory_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    session_id TEXT DEFAULT '',
    run_id TEXT DEFAULT '',
    layer TEXT NOT NULL CHECK(layer IN ('project','session','run')),
    kind TEXT NOT NULL DEFAULT 'note',
    content TEXT NOT NULL,
    source_ref TEXT DEFAULT '',
    dedup_key TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (project_id, session_id, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_session_memory_scope ON session_memory(project_id, session_id);

-- Plugin conformance reports (M3).  Bound to the plugin version/ABI so an
-- upgraded plugin makes an old report visibly stale instead of silently
-- "passed".
CREATE TABLE IF NOT EXISTS plugin_conformance (
    check_id TEXT PRIMARY KEY,
    plugin_name TEXT NOT NULL,
    plugin_version TEXT DEFAULT '',
    plugin_fingerprint TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','passed','failed')),
    checks_json TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_plugin_conformance_plugin
    ON plugin_conformance(plugin_name, created_at DESC);
"""


# Derived sqlite-vec shadow copies of ``tree_vectors`` plus the quantization
# scale table. Identity changes invalidate the whole derived index set — the
# next vector sync rebuilds them from the base rows.
_VEC_SHADOW_TABLES = (
    "tree_vectors_vec",
    "tree_vectors_vec_f32_bak",
    "tree_vectors_vec_i8",
    "vec_i8_scale",
)

# Ingest writes this literal when metadata carries no usable title; a merge
# must treat it as "no title" so a richer source row can fill it.
_PLACEHOLDER_TITLE = "Untitled"

# paper_ids external identifier kinds, in column order.
_EXTERNAL_ID_COLUMNS = ("doi", "arxiv", "s2_id", "openalex_id")


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the optional sqlite-vec extension into *conn*.

    Module-level so tests and tooling can stub the availability decision.
    """
    from drbrain.storage.vector_index import load_vec

    return load_vec(conn)


def _redact_payload(value: str | None) -> str:
    """Project a free-form payload through the credential scrubber.

    Writer boundary for durable rows: JSON payloads stay parseable with their
    sensitive keys redacted, plain text has credential assignments scrubbed.
    """
    from drbrain.security import redact_sensitive_text

    return redact_sensitive_text(value) or ""


def _split_evidence_id(identifier: str) -> tuple[str, str]:
    """Split a ``paper_id:node_id`` evidence identifier into its parts.

    ``paper_id`` is a local_id (sanitized, never contains ':') while ``node_id``
    may be a hierarchical tree node containing colons, so we split on the
    *first* colon. A bare identifier with no colon is treated as a paper-only
    grounding (empty node). Returns ``(paper_id, node_id)``.
    """
    identifier = (identifier or "").strip()
    if not identifier:
        return "", ""
    if ":" in identifier:
        paper_id, node_id = identifier.split(":", 1)
        return paper_id, node_id
    return identifier, ""


class Database:
    """Thin SQLite wrapper with schema auto-init."""

    def __init__(self, db_path: str | Path = "data/drbrain.db"):
        """Open SQLite database at *db_path*, enabling WAL mode and auto-migrating schema.

        ``check_same_thread=False``: the research loop shares one Database
        across the director thread and retrieval worker threads
        (``asyncio.to_thread(retrieve_documents, ...)``). CPython's sqlite3
        runs in serialized threading mode, so cross-thread use of the shared
        connection is safe; ``busy_timeout`` below absorbs write contention.
        """
        import os

        from drbrain.runtime import RuntimeContext, _is_special_path, _is_uri

        if _is_uri(db_path):
            raise ValueError(f"database path must be a local filesystem path, not a URI: {db_path}")
        selector = os.environ.get("DRBRAIN_ROOT") or os.environ.get("DRBRAIN_RUNTIME_ROOT")
        if selector and _is_special_path(db_path):
            # The in-memory sentinel must not require a valid disk root.
            self.path = Path(str(db_path))
        elif selector:
            self.path = RuntimeContext.create().assert_within_root(db_path, label="database path")
        else:
            self.path = Path(db_path)
        self._write_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self._migrate()
        self.conn.commit()

    @staticmethod
    def _validate_local_id(local_id: str) -> str:
        """Enforce the filesystem paper-ID contract at the SQL write boundary.

        Local ids double as filesystem path components, so the rules the
        ingest pipeline applies also bind here: non-empty, no surrounding
        whitespace, no NUL bytes.
        """
        value = str(local_id) if not isinstance(local_id, str) else local_id
        if not value or value != value.strip() or "\x00" in value:
            raise ValueError(f"invalid paper local_id: {value!r}")
        return value

    @contextmanager
    def _write_scope(self):
        """Composable write scope for writer methods.

        Inside a caller-open transaction the scope runs in a savepoint so a
        failure rolls back only its own writes; a writer never commits an
        outer transaction. Called standalone, the scope commits on success
        and rolls back on failure, preserving the historic writer behavior.
        """
        if self.conn.in_transaction:
            self.conn.execute("SAVEPOINT drbrain_write_scope")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK TO drbrain_write_scope")
                self.conn.execute("RELEASE drbrain_write_scope")
                raise
            self.conn.execute("RELEASE drbrain_write_scope")
        else:
            try:
                yield
            except BaseException:
                self.conn.rollback()
                raise
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
            (9, "concept_graph", self._migrate_add_concept_graph),
            (10, "concept_node_columns", self._migrate_add_concept_columns),
            (11, "concept_epistemic", self._migrate_add_concept_epistemic),
            (12, "knowledge_snapshots", self._migrate_add_knowledge_snapshots),
            (13, "answer_records", self._migrate_add_answer_records),
            (14, "evidence", self._migrate_add_evidence),
            (15, "claims", self._migrate_add_claims),
            (16, "agent_session_principal", self._migrate_add_agent_session_principal),
            (17, "claim_evidence", self._migrate_add_claim_evidence),
            (18, "paper_categories", self._migrate_add_paper_categories),
            (19, "claim_provenance", self._migrate_add_claim_provenance),
            (20, "embedding_revision", self._migrate_add_embedding_revision),
            (21, "project_scope", self._migrate_add_project_scope),
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

    def _migrate_add_agent_session_principal(self) -> None:
        """Bind agent sessions to an optional authenticated principal."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(agent_sessions)").fetchall()]
        if "owner_principal" not in cols:
            self.conn.execute(
                "ALTER TABLE agent_sessions ADD COLUMN owner_principal TEXT DEFAULT ''"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_owner ON agent_sessions(owner_principal)"
        )

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

    def _migrate_add_concept_graph(self) -> None:
        """Create concept graph layer tables (v9). Idempotent.

        Fresh databases already get these tables via SCHEMA_SQL; this migration
        records v9 in schema_versions for pre-existing databases and re-asserts
        the tables/indexes with IF NOT EXISTS for safety.
        """
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS corpus_sources (
                local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
                source TEXT NOT NULL,
                source_unique_id TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, source_unique_id)
            );
            CREATE TABLE IF NOT EXISTS concept_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                doc_freq INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                first_year INTEGER,
                last_year INTEGER
            );
            CREATE TABLE IF NOT EXISTS concept_cooccurrence (
                src_label TEXT NOT NULL,
                dst_label TEXT NOT NULL,
                year INTEGER,
                paper_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (src_label, dst_label, year, paper_id)
            );
            CREATE INDEX IF NOT EXISTS idx_cooccurrence_year ON concept_cooccurrence(year);
            CREATE INDEX IF NOT EXISTS idx_cooccurrence_src ON concept_cooccurrence(src_label);
            CREATE TABLE IF NOT EXISTS concept_embeddings (
                label TEXT PRIMARY KEY,
                vec BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS paper_citations (
                citing_id TEXT NOT NULL,
                cited_id TEXT NOT NULL,
                source TEXT DEFAULT '',
                year INTEGER,
                PRIMARY KEY (citing_id, cited_id)
            );
            CREATE TABLE IF NOT EXISTS paper_terms (
                local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
                term TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'keyword' CHECK(kind IN ('keyword', 'topic')),
                PRIMARY KEY (local_id, term, kind)
            );
            """
        )

    def _migrate_add_concept_columns(self) -> None:
        """Add concept type / noise / singleton flag columns to concept_nodes (v10).

        Mirrors the research-side concept cleanup layer: typed concepts
        (material / property / method / phenomenon / other) plus is_noise and
        is_singleton markers. Idempotent: fresh databases already get these
        columns via SCHEMA_SQL; pre-existing (v9) databases get them via ALTER.

        This migration only creates the columns. The values are populated by
        downstream logic, not here: is_singleton = doc_freq == 1; is_noise =
        label-pattern heuristics (e.g. single chars, numeric-only, biomed
        residue) combined with low doc_freq.
        """
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(concept_nodes)").fetchall()]
        if "type" not in cols:
            self.conn.execute("ALTER TABLE concept_nodes ADD COLUMN type TEXT DEFAULT 'other'")
        if "is_noise" not in cols:
            self.conn.execute("ALTER TABLE concept_nodes ADD COLUMN is_noise INTEGER DEFAULT 0")
        if "is_singleton" not in cols:
            self.conn.execute("ALTER TABLE concept_nodes ADD COLUMN is_singleton INTEGER DEFAULT 0")

    def _migrate_add_concept_epistemic(self) -> None:
        """Add epistemic columns to concepts (v11): provenance / authority / validity window.

        Upgrades the flat `concepts` table toward first-class Claim/Evidence
        objects: `provenance` records where the claim came from (SOURCE /
        RETRIEVED / TOOL_RESULT / MODEL_INFERRED / USER_STATED), `authority`
        records the trust tier (official_db / signed_contract / approved_policy /
        internal_wiki / email / chat, or a paper-domain equivalent), and
        `valid_from` / `valid_to` bound the claim's temporal validity (NULL =
        unknown start / still valid).

        Idempotent: fresh databases already get these columns via SCHEMA_SQL;
        pre-existing (v10 and earlier) databases get them via ALTER. No values
        are backfilled — they are populated by downstream epistemic ingestion.
        """
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(concepts)").fetchall()]
        if "provenance" not in cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN provenance TEXT DEFAULT ''")
        if "authority" not in cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN authority TEXT DEFAULT ''")
        if "valid_from" not in cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN valid_from INTEGER")
        if "valid_to" not in cols:
            self.conn.execute("ALTER TABLE concepts ADD COLUMN valid_to INTEGER")

    def _migrate_add_knowledge_snapshots(self) -> None:
        """Create knowledge_snapshots table (v12). Idempotent.

        Fresh databases already get this table via SCHEMA_SQL; this migration
        records v12 in schema_versions for pre-existing databases and re-asserts
        the table with IF NOT EXISTS for safety.
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                revision_id TEXT DEFAULT '',
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _migrate_add_answer_records(self) -> None:
        """Create answer_records table (v13). Idempotent.

        Fresh databases already get this table via SCHEMA_SQL; this migration
        records v13 in schema_versions for pre-existing databases and re-asserts
        the table with IF NOT EXISTS for safety.
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_records (
                answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_ids TEXT DEFAULT '',
                provenance TEXT DEFAULT '',
                model_version TEXT DEFAULT '',
                snapshot_id TEXT DEFAULT '',
                retriever_version TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _migrate_add_evidence(self) -> None:
        """Create evidence table (v14). Idempotent.

        Fresh databases already get this table via SCHEMA_SQL; this migration
        records v14 in schema_versions for pre-existing databases and re-asserts
        the table with IF NOT EXISTS for safety.
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                paper_id TEXT DEFAULT '',
                node_id TEXT DEFAULT '',
                page TEXT DEFAULT '',
                snippet TEXT DEFAULT '',
                value TEXT DEFAULT '',
                unit TEXT DEFAULT '',
                conditions TEXT DEFAULT '',
                provenance TEXT DEFAULT '',
                authority TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _migrate_add_claims(self) -> None:
        """Create claims table (v15). Idempotent.

        Fresh databases already get this table via SCHEMA_SQL; this migration
        records v15 in schema_versions for pre-existing databases and re-asserts
        the table with IF NOT EXISTS for safety.
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                label TEXT DEFAULT '',
                claim_text TEXT NOT NULL,
                claim_type TEXT DEFAULT '',
                authority TEXT DEFAULT '',
                provenance TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                valid_from INTEGER,
                valid_to INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _migrate_add_claim_evidence(self) -> None:
        """Add the many-to-many relationship between first-class claims and evidence."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claim_evidence (
                claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                PRIMARY KEY (claim_id, evidence_id)
            );
            CREATE INDEX IF NOT EXISTS idx_claim_evidence_evidence
                ON claim_evidence(evidence_id);
            """
        )

    def _migrate_add_claim_provenance(self) -> None:
        """v19: per-claim provenance columns (run, cycle, job, model, prompt)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(claims)").fetchall()]
        add_column = {
            "run_id": "TEXT DEFAULT ''",
            "cycle": "INTEGER",
            "job_id": "TEXT DEFAULT ''",
            "claim_ledger_id": "TEXT DEFAULT ''",
            "model": "TEXT DEFAULT ''",
            "prompt_hash": "TEXT DEFAULT ''",
            "evidence_node_ids": "TEXT DEFAULT ''",
        }
        for column, decl in add_column.items():
            if column not in cols:
                self.conn.execute(f"ALTER TABLE claims ADD COLUMN {column} {decl}")

    def _migrate_add_embedding_revision(self) -> None:
        """v20: seed the embedding model generation watermark (idempotent).

        The revision lives in vector_metadata so readers (graph engine cache)
        can detect a cleared or re-trained TransE model without rescanning
        the embeddings table. Missing row means generation 0.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO vector_metadata (key, value) VALUES ('embedding_revision', '0')"
        )

    def _migrate_add_paper_categories(self) -> None:
        """Add arXiv category metadata + corpus citation edges (physics ingest)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)").fetchall()]
        if "categories" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN categories TEXT DEFAULT ''")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_cite_keys (
                citing_local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
                cited_key TEXT NOT NULL,
                cited_local_id TEXT REFERENCES papers(local_id) ON DELETE SET NULL,
                PRIMARY KEY (citing_local_id, cited_key)
            );
            CREATE INDEX IF NOT EXISTS idx_paper_cite_keys_cited
                ON paper_cite_keys(cited_key);
            """
        )

    def _migrate_add_project_scope(self) -> None:
        """v21: project namespace, WebUI auth sessions, memory, conformance.

        New databases already carry the tables and the ``project_id`` column
        via ``SCHEMA_SQL``; this migration records the version and upgrades
        pre-v21 databases in place.  It is idempotent: every step re-checks
        the object it is about to create or alter, so applying it twice (or on
        a database where a subset already exists) is a no-op.
        """
        from drbrain.projects import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                workspace_name TEXT UNIQUE,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS webui_sessions (
                session_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked_at REAL,
                remote_addr TEXT DEFAULT '',
                user_agent TEXT DEFAULT ''
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_webui_sessions_token
                ON webui_sessions(token_hash);
            CREATE TABLE IF NOT EXISTS webui_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                detail TEXT DEFAULT '',
                remote_addr TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_webui_audit_event
                ON webui_audit(event, created_at);
            CREATE TABLE IF NOT EXISTS session_memory (
                memory_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                run_id TEXT DEFAULT '',
                layer TEXT NOT NULL CHECK(layer IN ('project','session','run')),
                kind TEXT NOT NULL DEFAULT 'note',
                content TEXT NOT NULL,
                source_ref TEXT DEFAULT '',
                dedup_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (project_id, session_id, dedup_key)
            );
            CREATE INDEX IF NOT EXISTS idx_session_memory_scope
                ON session_memory(project_id, session_id);
            CREATE TABLE IF NOT EXISTS plugin_conformance (
                check_id TEXT PRIMARY KEY,
                plugin_name TEXT NOT NULL,
                plugin_version TEXT DEFAULT '',
                plugin_fingerprint TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','passed','failed')),
                checks_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_plugin_conformance_plugin
                ON plugin_conformance(plugin_name, created_at DESC);
            """
        )
        session_cols = [
            r[1] for r in self.conn.execute("PRAGMA table_info(agent_sessions)").fetchall()
        ]
        if "project_id" not in session_cols:
            # SQLite stores the default as schema text, so the identifier is
            # interpolated (it is a fixed module constant, never user input).
            self.conn.execute(
                "ALTER TABLE agent_sessions ADD COLUMN project_id TEXT NOT NULL "
                f"DEFAULT '{DEFAULT_PROJECT_ID}'"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_project "
            "ON agent_sessions(project_id, status)"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO projects(project_id, name, description, is_default) "
            "VALUES (?, ?, '', 1)",
            (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME),
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
        categories: str = "",
        strict: bool = False,
    ) -> None:
        """Insert or ignore a paper record with full metadata fields.

        On conflict (existing local_id), bump updated_at to signal downstream
        incremental stages that this paper changed.
        """
        self._validate_local_id(local_id)
        if strict and self.get_paper(local_id) is not None:
            raise ValueError("paper identity already exists")
        self.conn.execute(
            "INSERT INTO papers (local_id, title, year, status, paper_type, "
            "journal, publisher, citation_count, volume, pages, authors, categories, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            + (
                ""
                if strict
                else "ON CONFLICT(local_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP"
            ),
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
                categories,
            ),
        )

    def insert_paper_ids(
        self,
        local_id: str,
        doi=None,
        arxiv=None,
        s2_id=None,
        openalex_id=None,
        *,
        strict: bool = False,
    ) -> None:
        """Insert or idempotently merge external identifier mappings.

        Values are normalized through the shared resolver so equivalent
        spellings compare equal. In ``strict`` mode a normalized value owned
        by a different paper, or a conflicting value for the paper's own
        existing mapping, is rejected; a later caller that skips strict mode
        keeps the lenient historical behavior (unique conflicts are ignored).
        Existing raw column values are never rewritten — only NULL fields
        are filled.
        """
        self._validate_local_id(local_id)
        from drbrain.dedup.resolver import _normalize_optional

        incoming: dict[str, str] = {}
        for kind, value in (
            ("doi", doi),
            ("arxiv", arxiv),
            ("s2_id", s2_id),
            ("openalex_id", openalex_id),
        ):
            normalized = _normalize_optional(kind, value)
            if normalized is None:
                continue
            incoming[kind] = normalized

        row = self.conn.execute(
            f"SELECT {', '.join(_EXTERNAL_ID_COLUMNS)} FROM paper_ids WHERE local_id = ?",
            (local_id,),
        ).fetchone()
        if strict:
            current = dict(zip(_EXTERNAL_ID_COLUMNS, row)) if row is not None else {}
            for kind, normalized in incoming.items():
                owner = self.get_paper_by_external_id(kind, normalized)
                if owner is not None and owner != local_id:
                    raise ValueError(
                        f"external identifier {kind} already belongs to another paper "
                        f"{owner}: {normalized}"
                    )
                stored = current.get(kind)
                if stored and _normalize_optional(kind, stored) != normalized:
                    raise ValueError(
                        f"external identifier {kind} already mapped for {local_id}: {stored}"
                    )
        if row is None:
            self.conn.execute(
                "INSERT OR IGNORE INTO paper_ids (local_id, doi, arxiv, s2_id, openalex_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    local_id,
                    incoming.get("doi"),
                    incoming.get("arxiv"),
                    incoming.get("s2_id"),
                    incoming.get("openalex_id"),
                ),
            )
        else:
            fills = {
                kind: value
                for kind, value in incoming.items()
                if not row[_EXTERNAL_ID_COLUMNS.index(kind)]
            }
            if fills:
                assignments = ", ".join(f"{kind} = ?" for kind in fills)
                self.conn.execute(
                    f"UPDATE OR IGNORE paper_ids SET {assignments} WHERE local_id = ?",
                    (*fills.values(), local_id),
                )

    def set_paper_abstract(self, local_id: str, abstract: str) -> None:
        """Update the abstract text for a paper."""
        self._validate_local_id(local_id)
        self.conn.execute(
            "UPDATE papers SET abstract = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (abstract, local_id),
        )

    def set_paper_categories(self, local_id: str, categories: str) -> None:
        """Store the space-separated arXiv category list for a paper."""
        self._validate_local_id(local_id)
        self.conn.execute(
            "UPDATE papers SET categories = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (categories, local_id),
        )

    def iter_paper_categories(self) -> list[tuple[str, str]]:
        """Return ``(local_id, categories)`` for every paper that has categories."""
        return [
            (r[0], r[1])
            for r in self.conn.execute(
                "SELECT local_id, categories FROM papers WHERE categories != ''"
            ).fetchall()
        ]

    def insert_paper_cite_keys(self, citing_local_id: str, cited_keys: list[str]) -> None:
        """Record raw citation keys (\\cite arguments) extracted from one paper.

        Resolution of ``cited_key`` (arXiv id / DOI / bib key) to an in-corpus
        ``cited_local_id`` happens later, once the full corpus is ingested.
        """
        self._validate_local_id(citing_local_id)
        if not cited_keys:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO paper_cite_keys (citing_local_id, cited_key) VALUES (?, ?)",
            [(citing_local_id, k) for k in cited_keys],
        )

    def resolve_paper_cite_keys(self, key_to_local_id: dict[str, str]) -> int:
        """Fill ``cited_local_id`` for citations whose key is now in-corpus.

        Returns the number of citation rows resolved.
        """
        with self.conn:
            rows = self.conn.execute(
                "SELECT rowid, cited_key FROM paper_cite_keys WHERE cited_local_id IS NULL"
            ).fetchall()
            updates = []
            for rowid, key in rows:
                target = key_to_local_id.get(str(key))
                if not target:
                    continue
                # A deferred resolution must respect the same identity
                # contract as a direct write.
                updates.append((self._validate_local_id(target), rowid))
            if updates:
                self.conn.executemany(
                    "UPDATE paper_cite_keys SET cited_local_id = ? WHERE rowid = ?",
                    updates,
                )
        return len(updates)

    def upgrade_placeholder(self, local_id: str) -> None:
        """Promote a placeholder paper to uploaded status."""
        self._validate_local_id(local_id)
        self.conn.execute(
            "UPDATE papers SET status = 'uploaded', updated_at = CURRENT_TIMESTAMP "
            "WHERE local_id = ? AND status = 'placeholder'",
            (local_id,),
        )

    def set_paper_status(self, local_id: str, status: str) -> None:
        """Update paper status and bump updated_at."""
        self._validate_local_id(local_id)
        self.conn.execute(
            "UPDATE papers SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (status, local_id),
        )

    def touch_paper(self, local_id: str) -> None:
        """Bump updated_at timestamp on a paper to signal downstream stages."""
        self._validate_local_id(local_id)
        self.conn.execute(
            "UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (local_id,),
        )

    def touch_edge(self, src_id: str, dst_id: str, relation: str, source_paper: str) -> None:
        """Bump updated_at on an edge to signal downstream stages."""
        self._validate_local_id(source_paper)
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
        self._validate_local_id(local_id)
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
        return cur.lastrowid or 0

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
        return cur.lastrowid or 0

    # -- Concept graph layer write helpers (v9) -------------------------

    def upsert_concept_node(
        self,
        label: str,
        doc_freq: int = 0,
        word_count: int = 0,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> None:
        """Insert a concept node or update its aggregated statistics."""
        self.conn.execute(
            "INSERT INTO concept_nodes (label, doc_freq, word_count, first_year, last_year) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(label) DO UPDATE SET "
            "doc_freq = excluded.doc_freq, word_count = excluded.word_count, "
            "first_year = excluded.first_year, last_year = excluded.last_year",
            (label, doc_freq, word_count, first_year, last_year),
        )

    def insert_cooccurrence(
        self,
        src_label: str,
        dst_label: str,
        year: int | None,
        paper_id: str,
        weight: float = 1.0,
    ) -> None:
        """Insert a co-occurrence edge; accumulate weight on re-assertion."""
        self._validate_local_id(paper_id)
        self.conn.execute(
            "INSERT INTO concept_cooccurrence (src_label, dst_label, year, paper_id, weight) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(src_label, dst_label, year, paper_id) "
            "DO UPDATE SET weight = concept_cooccurrence.weight + excluded.weight",
            (src_label, dst_label, year, paper_id, weight),
        )

    def insert_concept_embedding(self, label: str, vec: bytes, dim: int, model: str = "") -> None:
        """Insert or replace the semantic embedding for a concept label."""
        self.conn.execute(
            "INSERT INTO concept_embeddings (label, vec, dim, model) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(label) DO UPDATE SET vec = excluded.vec, dim = excluded.dim, "
            "model = excluded.model",
            (label, vec, dim, model),
        )

    def insert_corpus_source(self, local_id: str, source: str, source_unique_id: str) -> None:
        """Record the external-source provenance for an ingested paper."""
        self.conn.execute(
            "INSERT OR IGNORE INTO corpus_sources (local_id, source, source_unique_id) "
            "VALUES (?, ?, ?)",
            (local_id, source, source_unique_id),
        )

    def find_corpus_source(self, source: str, source_unique_id: str) -> str | None:
        """Return the local_id already ingested for a source unique_id, if any."""
        row = self.conn.execute(
            "SELECT local_id FROM corpus_sources WHERE source = ? AND source_unique_id = ?",
            (source, source_unique_id),
        ).fetchone()
        return row[0] if row else None

    def find_local_id_by_doi(self, doi: str) -> str | None:
        """Return the local_id mapped to a DOI, if any (secondary dedup key)."""
        row = self.conn.execute("SELECT local_id FROM paper_ids WHERE doi = ?", (doi,)).fetchone()
        return row[0] if row else None

    def insert_paper_citation(
        self, citing_id: str, cited_id: str, source: str = "", year: int | None = None
    ) -> None:
        """Insert a paper-level citation edge (citing -> cited)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_citations (citing_id, cited_id, source, year) "
            "VALUES (?, ?, ?, ?)",
            (citing_id, cited_id, source, year),
        )

    def insert_paper_term(self, local_id: str, term: str, kind: str = "keyword") -> None:
        """Insert a source-provided keyword/topic term for a paper."""
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_terms (local_id, term, kind) VALUES (?, ?, ?)",
            (local_id, term, kind),
        )

    def get_paper_terms(self, local_id: str) -> list[tuple[str, str]]:
        """Return (term, kind) pairs for a paper."""
        return self.conn.execute(
            "SELECT term, kind FROM paper_terms WHERE local_id = ?", (local_id,)
        ).fetchall()

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
        self._validate_local_id(local_id)
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

        Migrates concepts, arguments, edges, external identifiers, provenance
        rows, citation edges, and epistemic evidence from merge_id onto
        keep_id, fills keep's placeholder metadata from the source row, then
        deletes merge_id. Derived indexes (embeddings, tree vectors, sqlite-vec
        shadow tables) are invalidated rather than copied so no stale identity
        survives; the embedding revision is bumped so cached readers reload.

        All rejections (identity errors, external-ID conflicts, citation-cache
        collisions) happen in a preflight before any row is written, and the
        migration itself runs in a savepoint: on failure only the merge's own
        writes roll back and the caller's transaction stays intact. Returns a
        dict of migrated/deleted counts.
        """
        self._validate_local_id(keep_id)
        self._validate_local_id(merge_id)
        if keep_id == merge_id:
            raise ValueError("merge requires two different papers")
        if self.get_paper(keep_id) is None:
            raise ValueError(f"merge source paper not found: {keep_id}")
        if self.get_paper(merge_id) is None:
            raise ValueError(f"merge target paper not found: {merge_id}")

        self._preflight_merge_paper_ids(keep_id, merge_id)
        self._preflight_merge_citation_cache(keep_id, merge_id)
        self._preflight_merge_vec_tables()

        caller_transactional = self.conn.in_transaction
        self.conn.execute("SAVEPOINT drbrain_merge_papers")
        try:
            counts = self._apply_paper_merge(keep_id, merge_id)
        except BaseException:
            self.conn.execute("ROLLBACK TO drbrain_merge_papers")
            self.conn.execute("RELEASE drbrain_merge_papers")
            raise
        self.conn.execute("RELEASE drbrain_merge_papers")
        if not caller_transactional:
            self.conn.commit()
        return counts

    def _preflight_merge_paper_ids(self, keep_id: str, merge_id: str) -> None:
        """Reject a merge whose external identifiers conflict per kind."""
        from drbrain.dedup.resolver import _normalize_optional

        keep_ids = self._external_id_row(keep_id)
        merge_ids = self._external_id_row(merge_id)
        for kind in _EXTERNAL_ID_COLUMNS:
            keep_value, merge_value = keep_ids[kind], merge_ids[kind]
            if not (keep_value and merge_value):
                continue
            if _normalize_optional(kind, keep_value) != _normalize_optional(kind, merge_value):
                raise ValueError(
                    f"external identifier {kind} conflict between {keep_id} and {merge_id}: "
                    f"{keep_value!r} vs {merge_value!r}"
                )

    def _preflight_merge_citation_cache(self, keep_id: str, merge_id: str) -> None:
        """Reject a merge whose citation-cache rows share a title but disagree.

        Two rows with the same target title but different DOIs or years are
        ambiguous references; migrating them silently would fabricate a merged
        citation record, so the merge aborts instead.
        """
        keep_rows = self.conn.execute(
            "SELECT target_title, target_year, target_doi, target_s2_id "
            "FROM citation_cache WHERE source_paper = ?",
            (keep_id,),
        ).fetchall()
        keep_by_title = {row[0]: row for row in keep_rows}
        for title, year, doi, s2_id in self.conn.execute(
            "SELECT target_title, target_year, target_doi, target_s2_id "
            "FROM citation_cache WHERE source_paper = ?",
            (merge_id,),
        ).fetchall():
            existing = keep_by_title.get(title)
            if existing is None:
                continue
            if (
                (existing[1] or None) != (year or None)
                or (existing[2] or None) != (doi or None)
                or (existing[3] or None) != (s2_id or None)
            ):
                raise ValueError(
                    f"citation cache conflict for target title {title!r} between "
                    f"{keep_id} and {merge_id}"
                )

    def _preflight_merge_vec_tables(self) -> None:
        """Fail closed when ANN shadow tables exist without sqlite-vec loaded.

        Identity migration deletes the derived vec0 tables' rows; that write
        requires the extension. A backfill process may have created the
        virtual tables on a connection that had it loaded while this
        connection does not, so the merge aborts before mutating anything.
        """
        virtual = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND sql LIKE 'CREATE VIRTUAL TABLE%'"
            ).fetchall()
        }
        if not (virtual & set(_VEC_SHADOW_TABLES)):
            return
        if not _load_sqlite_vec(self.conn):
            raise ValueError(
                "sqlite-vec extension unavailable: cannot migrate virtual ANN shadow tables"
            )

    def _external_id_row(self, local_id: str) -> dict[str, str | None]:
        """Return the paper_ids row for *local_id* as a kind→value dict."""
        row = self.conn.execute(
            f"SELECT {', '.join(_EXTERNAL_ID_COLUMNS)} FROM paper_ids WHERE local_id = ?",
            (local_id,),
        ).fetchone()
        if row is None:
            return dict.fromkeys(_EXTERNAL_ID_COLUMNS)
        return dict(zip(_EXTERNAL_ID_COLUMNS, row))

    @staticmethod
    def _merge_person_names(*values: str | None) -> str:
        """Union ';'-separated author lists, keeping first-seen order."""
        names: list[str] = []
        for value in values:
            for chunk in (value or "").split(";"):
                name = chunk.strip()
                if name and name not in names:
                    names.append(name)
        return "; ".join(names)

    @staticmethod
    def _merge_category_tokens(*values: str | None) -> str:
        """Union whitespace/comma-separated category tokens, keeping order."""
        tokens: list[str] = []
        for value in values:
            for chunk in (value or "").replace(",", " ").split():
                if chunk and chunk not in tokens:
                    tokens.append(chunk)
        return " ".join(tokens)

    def _merge_paper_metadata(self, keep_id: str, merge_id: str) -> None:
        """Fill keep's placeholder/empty fields from the source row.

        Canonical values win: a real keep title/year/status/venue is never
        overwritten by the duplicate. Authors and categories are unions
        rather than pick-one fields.
        """
        cols = (
            "title",
            "abstract",
            "year",
            "status",
            "paper_type",
            "journal",
            "publisher",
            "citation_count",
            "volume",
            "pages",
            "authors",
            "categories",
        )

        def paper_row(paper_id: str) -> dict:
            row = self.conn.execute(
                f"SELECT {', '.join(cols)} FROM papers WHERE local_id = ?",
                (paper_id,),
            ).fetchone()
            return dict(zip(cols, row))

        keep, gone = paper_row(keep_id), paper_row(merge_id)
        merged = {
            "title": (
                keep["title"]
                if keep["title"] and keep["title"].strip() and keep["title"] != _PLACEHOLDER_TITLE
                else (gone["title"] or keep["title"])
            ),
            "abstract": keep["abstract"] or gone["abstract"],
            "year": keep["year"] if keep["year"] is not None else gone["year"],
            "status": keep["status"] if keep["status"] != "placeholder" else gone["status"],
            "paper_type": (
                keep["paper_type"]
                if keep["paper_type"] != "paper"
                else (gone["paper_type"] or keep["paper_type"])
            ),
            "journal": keep["journal"] or gone["journal"],
            "publisher": keep["publisher"] or gone["publisher"],
            "citation_count": keep["citation_count"] or gone["citation_count"],
            "volume": keep["volume"] or gone["volume"],
            "pages": keep["pages"] or gone["pages"],
            "authors": self._merge_person_names(keep["authors"], gone["authors"]),
            "categories": self._merge_category_tokens(keep["categories"], gone["categories"]),
        }
        assignments = ", ".join(f"{col} = ?" for col in cols)
        self.conn.execute(
            f"UPDATE papers SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (*merged.values(), keep_id),
        )

    def _apply_paper_merge(self, keep_id: str, merge_id: str) -> dict:
        """Run the row-level migration of one merge inside the savepoint."""
        counts = {
            "concepts": 0,
            "arguments": 0,
            "edges_redirected": 0,
            "paper_ids": 0,
            "corpus_sources": 0,
            "paper_terms": 0,
            "concept_cooccurrence": 0,
            "paper_cite_keys": 0,
            "paper_citations": 0,
            "citation_cache": 0,
            "build_stages": 0,
            "queue_items": 0,
            "evidence": 0,
            "tree_vectors": 0,
            "tree_summaries": 0,
            "embeddings": 0,
        }
        cur = self.conn.execute(
            "UPDATE concepts SET local_id = ? WHERE local_id = ?", (keep_id, merge_id)
        )
        counts["concepts"] = cur.rowcount
        cur = self.conn.execute(
            "UPDATE arguments SET source_paper = ? WHERE source_paper = ?",
            (keep_id, merge_id),
        )
        counts["arguments"] = cur.rowcount
        # Redirect edges that reference merge_id as an endpoint (src or dst).
        # In DrBrain edges can use either concept labels or paper local_ids
        # as endpoints (papers are graph nodes too), so this retargeting is
        # NOT dead code — it handles the paper-as-node case.
        cur = self.conn.execute(
            "UPDATE edges SET src_id = ?, updated_at = CURRENT_TIMESTAMP WHERE src_id = ?",
            (keep_id, merge_id),
        )
        counts["edges_redirected"] += cur.rowcount
        cur = self.conn.execute(
            "UPDATE edges SET dst_id = ?, updated_at = CURRENT_TIMESTAMP WHERE dst_id = ?",
            (keep_id, merge_id),
        )
        counts["edges_redirected"] += cur.rowcount
        cur = self.conn.execute(
            "UPDATE edges SET source_paper = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE source_paper = ?",
            (keep_id, merge_id),
        )
        counts["edges_redirected"] += cur.rowcount

        # External identifiers: keep's non-null fields win, gone's fill the
        # rest, then the source row disappears. The source row is deleted
        # BEFORE the update so its own UNIQUE-indexed values don't collide
        # with the merged row being written.
        keep_ids = self._external_id_row(keep_id)
        merge_ids = self._external_id_row(merge_id)
        if any(value is not None for value in merge_ids.values()):
            merged_ids = {
                kind: keep_ids[kind] if keep_ids[kind] is not None else merge_ids[kind]
                for kind in _EXTERNAL_ID_COLUMNS
            }
            self.conn.execute("DELETE FROM paper_ids WHERE local_id = ?", (merge_id,))
            self.conn.execute(
                f"UPDATE paper_ids SET {', '.join(f'{kind} = ?' for kind in _EXTERNAL_ID_COLUMNS)} "
                f"WHERE local_id = ?",
                (*merged_ids.values(), keep_id),
            )
            counts["paper_ids"] = 1

        # Provenance + derived rows owned by the source paper.
        for table, column in (
            ("corpus_sources", "local_id"),
            ("paper_terms", "local_id"),
            ("concept_cooccurrence", "paper_id"),
            ("citation_cache", "source_paper"),
            ("evidence", "paper_id"),
        ):
            cur = self.conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                (keep_id, merge_id),
            )
            counts[table] = cur.rowcount
        cur = self.conn.execute(
            "UPDATE confidence_queue SET source_paper = ? WHERE source_paper = ?",
            (keep_id, merge_id),
        )
        counts["queue_items"] = cur.rowcount

        # Citation keys: canonicalize resolved targets BEFORE retargeting the
        # citing side so keep-row and gone-row entries for the same key fold
        # into one equivalent row instead of falsely conflicting.
        self.conn.execute(
            "UPDATE paper_cite_keys SET cited_local_id = ? WHERE cited_local_id = ?",
            (keep_id, merge_id),
        )
        moved = self.conn.execute(
            "SELECT cited_key, cited_local_id FROM paper_cite_keys WHERE citing_local_id = ?",
            (merge_id,),
        ).fetchall()
        for cited_key, cited_local_id in moved:
            self.conn.execute(
                "INSERT OR IGNORE INTO paper_cite_keys (citing_local_id, cited_key, cited_local_id) "
                "VALUES (?, ?, ?)",
                (keep_id, cited_key, cited_local_id),
            )
        cur = self.conn.execute(
            "DELETE FROM paper_cite_keys WHERE citing_local_id = ?", (merge_id,)
        )
        counts["paper_cite_keys"] = len(moved) or cur.rowcount

        # Paper-level citation edges, both directions; a self-citation of the
        # source identity becomes a self-citation of the survivor.
        cur = self.conn.execute(
            "UPDATE OR IGNORE paper_citations SET citing_id = ? WHERE citing_id = ?",
            (keep_id, merge_id),
        )
        counts["paper_citations"] += cur.rowcount
        cur = self.conn.execute(
            "UPDATE OR IGNORE paper_citations SET cited_id = ? WHERE cited_id = ?",
            (keep_id, merge_id),
        )
        counts["paper_citations"] += cur.rowcount

        # Build stages are invalidated: the stage result describes the source
        # extraction, which no longer exists, so the target rebuilds.
        cur = self.conn.execute(
            "UPDATE build_stages SET paper_id = ?, result_json = '', "
            "updated_at = CURRENT_TIMESTAMP WHERE paper_id = ?",
            (keep_id, merge_id),
        )
        counts["build_stages"] = cur.rowcount

        # Derived tree indexes are invalidated rather than copied with stale
        # ``gone:`` node IDs; the target can rebuild them from its canonical tree.
        cur = self.conn.execute(
            "DELETE FROM tree_vectors WHERE paper_id IN (?, ?)", (keep_id, merge_id)
        )
        counts["tree_vectors"] = cur.rowcount
        cur = self.conn.execute(
            "DELETE FROM tree_summaries WHERE paper_id IN (?, ?)", (keep_id, merge_id)
        )
        counts["tree_summaries"] = cur.rowcount

        self._merge_paper_metadata(keep_id, merge_id)
        self.conn.execute(
            "UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE local_id = ?",
            (keep_id,),
        )
        self.conn.execute("DELETE FROM papers WHERE local_id = ?", (merge_id,))

        # Identity change invalidates every derived vector artifact: the TransE
        # cache is a global artifact and the ANN shadows mirror tree_vectors.
        cur = self.conn.execute("DELETE FROM embeddings")
        counts["embeddings"] = cur.rowcount
        self._purge_vec_shadow_tables()
        self._mark_vec_dirty()
        self._bump_embedding_revision()
        return counts

    def _purge_vec_shadow_tables(self) -> int:
        """Delete every derived ANN row; the next vector sync rebuilds them.

        The shadow tables are full copies of ``tree_vectors``, so any base-row
        identity change invalidates the whole set — partial pruning would leave
        quantized copies inconsistent with their base rows. Tables are created
        lazily by the backfill/quantize tooling and simply may not exist.
        """
        deleted = 0
        for table in _VEC_SHADOW_TABLES:
            try:
                cur = self.conn.execute(f"DELETE FROM {table}")
                deleted += max(cur.rowcount, 0)
            except sqlite3.OperationalError:
                continue
        return deleted

    def _mark_vec_dirty(self) -> None:
        """Clear the sqlite-vec synced watermark so retrieval falls back to
        the brute-force path until the next full sync."""
        from drbrain.storage.vector_index import mark_vec_dirty

        mark_vec_dirty(self.conn)

    def _bump_embedding_revision(self) -> None:
        """Advance the embedding model generation so cached readers reload."""
        row = self.conn.execute(
            "SELECT value FROM vector_metadata WHERE key = 'embedding_revision'"
        ).fetchone()
        try:
            current = int(row[0]) if row else 0
        except (TypeError, ValueError):
            current = 0
        self.conn.execute(
            "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES ('embedding_revision', ?)",
            (str(current + 1),),
        )

    def get_embedding_revision(self) -> int:
        """Return the current persisted embedding model generation (0 default)."""
        row = self.conn.execute(
            "SELECT value FROM vector_metadata WHERE key = 'embedding_revision'"
        ).fetchone()
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    # ── agent_sessions / agent_messages ─────────────────────────────────

    def insert_agent_session(
        self,
        session_id: str,
        title: str = "",
        system_prompt: str = "",
        model_config: str = "{}",
        owner_principal: str = "",
    ) -> None:
        """Create a new agent session row.

        Free-form payloads are redacted before storage: session rows are
        durable artifacts, so credentials passed through by direct low-level
        callers must not survive verbatim.
        """
        self.conn.execute(
            "INSERT INTO agent_sessions "
            "(session_id, title, system_prompt, model_config, owner_principal) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                _redact_payload(title),
                _redact_payload(system_prompt),
                _redact_payload(model_config),
                owner_principal,
            ),
        )

    def session_principal_matches(self, session_id: str, principal: str | None) -> bool:
        """Return whether an active session belongs to ``principal``.

        ``principal=None`` retains the historic local-CLI behavior.  Supplying
        a principal is fail-closed: unowned legacy sessions and other owners
        do not match.
        """
        row = self.conn.execute(
            "SELECT owner_principal FROM agent_sessions "
            "WHERE session_id = ? AND status != 'deleted'",
            (session_id,),
        ).fetchone()
        if row is None:
            return False
        if principal is not None and not str(principal).strip():
            return False
        return principal is None or str(row[0] or "") == principal

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
        """Append a message to an agent session.

        Message content, serialized tool payloads, and tool names are redacted
        before storage; structural identifiers (session, seq, role, tool_call_id)
        are kept verbatim so the transcript stays navigable.
        """
        self.conn.execute(
            "INSERT INTO agent_messages "
            "(session_id, seq, role, content, tool_calls_json, tool_call_id, tool_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                seq,
                role,
                _redact_payload(content),
                _redact_payload(tool_calls_json),
                tool_call_id,
                _redact_payload(tool_name),
            ),
        )

    def record_knowledge_snapshot(
        self, snapshot_id: str, revision_id: str = "", description: str = ""
    ) -> str:
        """Register one knowledge snapshot (the v12 table's production writer).

        A snapshot marks the knowledge state a settle (or answer) was grounded
        in: ``revision_id`` pins the retrieval generation, ``description``
        carries the human-readable cycle summary. Idempotent on
        ``snapshot_id`` — a deterministic id (e.g. hash of the settled claim
        set) makes re-settling the same outcome a no-op.
        """
        with self._write_scope():
            self.conn.execute(
                "INSERT INTO knowledge_snapshots (snapshot_id, revision_id, description) "
                "VALUES (?, ?, ?) ON CONFLICT(snapshot_id) DO NOTHING",
                (snapshot_id, revision_id, description),
            )
        return snapshot_id

    def record_answer(
        self,
        question: str,
        answer: str,
        *,
        session_id: str | None = None,
        evidence_ids: list[str] | None = None,
        provenance: str = "",
        model_version: str = "",
        snapshot_id: str = "",
        retriever_version: str = "",
    ) -> int:
        """Persist an answer bound to its supporting evidence (auditability).

        ``evidence_ids`` is serialized to a JSON string (the
        ``answer_records.evidence_ids`` column is TEXT) so it can be restored
        with ``json.loads`` later. Returns the new ``answer_id``.

        The answer row, its materialized evidence, and the claim are written
        as one unit: a failure while materializing the claim rolls back the
        whole record instead of leaving a dangling answer behind.
        """
        import json

        identifiers = list(evidence_ids or [])
        # Validate every grounding BEFORE persisting so a malformed
        # identifier cannot leave a partial answer row behind.
        for identifier in identifiers:
            paper_id, _node_id = _split_evidence_id(identifier)
            self._validate_local_id(paper_id)

        evidence_json = json.dumps(identifiers, ensure_ascii=False)
        with self._write_scope():
            cur = self.conn.execute(
                "INSERT INTO answer_records "
                "(session_id, question, answer, evidence_ids, provenance, "
                " model_version, snapshot_id, retriever_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    question,
                    answer,
                    evidence_json,
                    provenance,
                    model_version,
                    snapshot_id,
                    retriever_version,
                ),
            )
            answer_id = cur.lastrowid or 0

            # Materialize first-class evidence rows: each ``paper:node`` identifier
            # becomes an ``evidence`` row. At answer time we only have the grounding
            # id and provenance — page/snippet/value are left blank rather than
            # fabricated.
            for identifier in identifiers:
                paper_id, node_id = _split_evidence_id(identifier)
                if paper_id or node_id:
                    self.record_evidence(paper_id, node_id, provenance=provenance)

            # Materialize the answer itself as a first-class claim. The TBox type
            # (Problem/Method/Conclusion/…) is not known at answer time, so it is
            # left blank rather than guessed.
            self.record_claim(question, answer, provenance=provenance)
        return answer_id

    def record_evidence(
        self,
        paper_id: str,
        node_id: str = "",
        *,
        page: str = "",
        snippet: str = "",
        value: str = "",
        unit: str = "",
        conditions: str = "",
        provenance: str = "",
        authority: str = "",
        evidence_id: str | None = None,
    ) -> str:
        """Insert (or replace) a first-class evidence row. Returns ``evidence_id``.

        ``evidence_id`` defaults to ``paper_id:node_id`` (or whichever part is
        present), so re-recording the same grounding is idempotent. A later
        record with the same id replaces the row, letting richer fields (page /
        snippet / value) overwrite the sparse answer-time grounding.
        """
        self._validate_local_id(paper_id)
        if evidence_id is None:
            evidence_id = (
                f"{paper_id}:{node_id}" if (paper_id and node_id) else (paper_id or node_id)
            )
        with self._write_scope():
            self.conn.execute(
                "INSERT INTO evidence "
                "(evidence_id, paper_id, node_id, page, snippet, value, unit, "
                " conditions, provenance, authority) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(evidence_id) DO UPDATE SET "
                "paper_id = excluded.paper_id, node_id = excluded.node_id, "
                "page = excluded.page, snippet = excluded.snippet, value = excluded.value, "
                "unit = excluded.unit, conditions = excluded.conditions, "
                "provenance = excluded.provenance, authority = excluded.authority",
                (
                    evidence_id,
                    paper_id,
                    node_id,
                    page,
                    snippet,
                    value,
                    unit,
                    conditions,
                    provenance,
                    authority,
                ),
            )
        return evidence_id

    def record_claim_evidence(self, claim_id: str, evidence_ids: list[str]) -> list[str]:
        """Bind a persisted claim to existing first-class evidence rows.

        The relation is additive and idempotent. SQLite foreign keys prevent
        unknown claim/evidence identifiers from becoming dangling provenance.
        """
        unique_ids = list(dict.fromkeys(str(value) for value in evidence_ids if str(value)))
        if not unique_ids:
            return []
        with self._write_scope():
            self.conn.executemany(
                "INSERT INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?) "
                "ON CONFLICT(claim_id, evidence_id) DO NOTHING",
                [(claim_id, evidence_id) for evidence_id in unique_ids],
            )
        return unique_ids

    def record_claim(
        self,
        label: str,
        claim_text: str,
        *,
        claim_type: str = "",
        authority: str = "",
        provenance: str = "",
        confidence: float = 1.0,
        valid_from: int | None = None,
        valid_to: int | None = None,
        claim_id: str | None = None,
        run_id: str = "",
        cycle: int | None = None,
        job_id: str = "",
        claim_ledger_id: str = "",
        model: str = "",
        prompt_hash: str = "",
        evidence_node_ids: str = "",
    ) -> str:
        """Insert or update a first-class claim row. Returns ``claim_id``.

        ``claim_id`` defaults to a stable hash of ``label`` + ``claim_text`` so
        re-recording the same assertion is idempotent without replacing the
        row, preserving claim-to-evidence foreign-key relationships. The
        provenance columns (v19) pin a claim to the run/cycle/job/model that
        produced it — a claim without provenance cannot be audited.
        """
        import hashlib

        if claim_id is None:
            # claim_type participates in the identity: a statement verified in
            # one run and falsified in another must coexist (the review's
            # "Rejected→Conclusion 翻转静默覆盖" bug) so authority rules can
            # arbitrate instead of last-writer-wins.
            # NOTE (upgrade window): pre-v19 rows were keyed by
            # label+claim_text only; the first re-record of the same
            # assertion inserts a new-keyed row and the legacy row lingers
            # with empty provenance. authority arbitration sees both — the
            # stale orphan simply loses on freshness, so no re-key migration
            # is required, but expect one duplicate per legacy claim.
            digest = hashlib.sha1(f"{label}\x00{claim_text}\x00{claim_type}".encode()).hexdigest()
            claim_id = f"claim_{digest[:16]}"
        with self._write_scope():
            self.conn.execute(
                "INSERT INTO claims "
                "(claim_id, label, claim_text, claim_type, authority, provenance, "
                " confidence, valid_from, valid_to, run_id, cycle, job_id, "
                " claim_ledger_id, model, prompt_hash, evidence_node_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(claim_id) DO UPDATE SET "
                "label = excluded.label, claim_text = excluded.claim_text, "
                "claim_type = excluded.claim_type, authority = excluded.authority, "
                "provenance = excluded.provenance, confidence = excluded.confidence, "
                "valid_from = excluded.valid_from, valid_to = excluded.valid_to, "
                # v19 溯源列只在传入非空时覆盖：record_answer 走同一条 INSERT 且不
                # 带溯源参数，无条件 UPDATE 会把 loop 写入的 run_id/job_id/model
                # 等审计事实抹成空串（OCR r5 bug·high）。
                "run_id = CASE WHEN excluded.run_id != '' THEN excluded.run_id ELSE claims.run_id END, "
                "cycle = CASE WHEN excluded.cycle IS NOT NULL THEN excluded.cycle ELSE claims.cycle END, "
                "job_id = CASE WHEN excluded.job_id != '' THEN excluded.job_id ELSE claims.job_id END, "
                "claim_ledger_id = CASE WHEN excluded.claim_ledger_id != '' "
                "THEN excluded.claim_ledger_id ELSE claims.claim_ledger_id END, "
                "model = CASE WHEN excluded.model != '' THEN excluded.model ELSE claims.model END, "
                "prompt_hash = CASE WHEN excluded.prompt_hash != '' THEN excluded.prompt_hash "
                "ELSE claims.prompt_hash END, "
                "evidence_node_ids = CASE WHEN excluded.evidence_node_ids != '' "
                "THEN excluded.evidence_node_ids ELSE claims.evidence_node_ids END",
                (
                    claim_id,
                    label,
                    claim_text,
                    claim_type,
                    authority,
                    provenance,
                    confidence,
                    valid_from,
                    valid_to,
                    run_id,
                    cycle,
                    job_id,
                    claim_ledger_id,
                    model,
                    prompt_hash,
                    evidence_node_ids,
                ),
            )
        return claim_id

    # -- Embeddings --

    def save_embedding(self, entity: str, vec, dim: int) -> None:
        """Persist a TransE entity/relation vector and advance the model generation."""
        import numpy as np

        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (entity, vec, dim) VALUES (?, ?, ?)",
            (entity, np.array(vec, dtype=np.float32).tobytes(), dim),
        )
        self._bump_embedding_revision()

    def load_embeddings(self) -> dict:
        """Load all entity/relation vectors into a dict keyed by entity label."""
        import numpy as np

        rows = self.conn.execute("SELECT entity, vec, dim FROM embeddings").fetchall()
        return {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}

    def clear_embeddings(self) -> int:
        """Delete all embeddings (before re-training).

        Returns the number of deleted rows and advances the model generation
        so readers with a cached model detect the invalidation.
        """
        cur = self.conn.execute("DELETE FROM embeddings")
        deleted = max(cur.rowcount, 0)
        self._bump_embedding_revision()
        return deleted

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

    def list_papers(
        self,
        *,
        paper_ids: list[str] | None = None,
        query: str = "",
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Paged paper listing, optionally restricted to a membership set.

        ``paper_ids=None`` lists the whole library (the default project);
        ``query`` matches title/authors/abstract with LIKE escaping so a user
        supplied ``%`` cannot turn into a wildcard.  Returns ``(rows, total)``
        where ``total`` is the unpaged count for the same filters.
        """
        clauses: list[str] = []
        params: list = []
        if paper_ids is not None:
            # json_each keeps the statement valid for large workspaces without
            # building one SQL placeholder per paper id.
            clauses.append("p.local_id IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(list(paper_ids)))
        if status:
            clauses.append("p.status = ?")
            params.append(status)
        q = query.strip()
        if q:
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like = f"%{escaped}%"
            clauses.append(
                "(p.title LIKE ? ESCAPE '\\' OR p.authors LIKE ? ESCAPE '\\' "
                "OR p.abstract LIKE ? ESCAPE '\\')"
            )
            params.extend([like, like, like])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        total_row = self.conn.execute(
            f"SELECT COUNT(*) FROM papers p {where}", tuple(params)
        ).fetchone()
        rows = self.conn.execute(
            f"""
            SELECT p.local_id, p.title, p.abstract, p.year, p.paper_type, p.status,
                   p.journal, p.authors, p.citation_count, p.categories, p.updated_at
            FROM papers p {where}
            ORDER BY COALESCE(p.updated_at, p.created_at) DESC, p.local_id
            LIMIT ? OFFSET ?
            """,
            tuple(params) + (max(1, int(limit)), max(0, int(offset))),
        ).fetchall()
        cols = [
            "local_id",
            "title",
            "abstract",
            "year",
            "paper_type",
            "status",
            "journal",
            "authors",
            "citation_count",
            "categories",
            "updated_at",
        ]
        return [dict(zip(cols, row)) for row in rows], int(total_row[0] if total_row else 0)

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
        self._validate_local_id(source_paper)
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
        return cur.lastrowid or 0

    def insert_queue_item(
        self, source_paper: str, item_type: str, item_data: str, confidence: float
    ) -> int:
        """Insert a confidence queue item. Returns queue_id."""
        self._validate_local_id(source_paper)
        cur = self.conn.execute(
            "INSERT INTO confidence_queue (source_paper, item_type, item_data, confidence, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (source_paper, item_type, item_data, confidence),
        )
        return cur.lastrowid or 0

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

        Deletion also purges the same derived-vector shadows a merge would
        (tree vectors, sqlite-vec ANN copies, synced watermark, embedding
        revision), clears lazy pipeline caches, and converts evidence rows
        into audit tombstones so historical answers/claims stay resolvable
        without retaining the deleted paper's content.
        """
        self._validate_local_id(local_id)

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
        # Edges die when this paper asserted them OR when its local_id is used
        # as a graph endpoint (papers are nodes too), even if another paper
        # asserted the edge.
        edge_count = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_paper = ? OR src_id = ? OR dst_id = ?",
            (local_id, local_id, local_id),
        ).fetchone()[0]
        queue_count = self.conn.execute(
            "SELECT COUNT(*) FROM confidence_queue WHERE source_paper = ?", (local_id,)
        ).fetchone()[0]
        evidence_count = self.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE paper_id = ?", (local_id,)
        ).fetchone()[0]

        self.conn.execute("DELETE FROM concepts WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM arguments WHERE source_paper = ?", (local_id,))
        self.conn.execute(
            "DELETE FROM edges WHERE source_paper = ? OR src_id = ? OR dst_id = ?",
            (local_id, local_id, local_id),
        )
        self.conn.execute("DELETE FROM paper_ids WHERE local_id = ?", (local_id,))
        self.conn.execute("DELETE FROM confidence_queue WHERE source_paper = ?", (local_id,))
        vectors_deleted = self.conn.execute(
            "DELETE FROM tree_vectors WHERE paper_id = ?", (local_id,)
        ).rowcount
        self.conn.execute("DELETE FROM tree_summaries WHERE paper_id = ?", (local_id,))

        # Lazy pipeline caches are keyed by local_id; re-ingesting the paper
        # must not reuse derived rows computed under the old identity.
        cache_deleted = self._delete_lazy_cache("paper_concepts_cache", local_id)
        l1_deleted = self._delete_lazy_cache("kg_l1_attempted", local_id)

        # Historical answers/claims keep resolvable evidence IDs; the row
        # becomes a tombstone so its content is redacted but FK-backed audit
        # references (claim_evidence, answer_records.evidence_ids) survive.
        if evidence_count:
            self.conn.execute(
                "UPDATE evidence SET paper_id = '', node_id = '', page = '', "
                "snippet = '', value = '', unit = '', conditions = '', "
                "provenance = ? WHERE paper_id = ?",
                (f"PAPER_DELETED:{local_id}", local_id),
            )

        self.conn.execute("DELETE FROM papers WHERE local_id = ?", (local_id,))

        # Identity removal invalidates the derived vector artifacts exactly
        # like a merge does.
        shadow_deleted = self._purge_vec_shadow_tables()
        if max(vectors_deleted, 0) + shadow_deleted > 0:
            self._mark_vec_dirty()
            self._bump_embedding_revision()

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
            "evidence": evidence_count,
            "paper_concepts_cache": cache_deleted,
            "kg_l1_attempted": l1_deleted,
            "touched_neighbors": touched_neighbors,
        }

    def _delete_lazy_cache(self, table: str, local_id: str) -> int:
        """Delete one paper's rows from a lazily-created cache table."""
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if not exists:
            return 0
        cur = self.conn.execute(f"DELETE FROM {table} WHERE local_id = ?", (local_id,))
        return max(cur.rowcount, 0)

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

    @contextmanager
    def write_lock(self):
        """Serialize callers sharing this database's batch-write lock."""
        with self._write_lock:
            yield

    # -- Project scope (v21) --

    def upsert_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        workspace_name: str | None = None,
        *,
        is_default: bool = False,
    ) -> None:
        """Create or rename a project.

        The ``project_id`` is the durable identity: re-running this with the
        same id but a new name renames the project without breaking the
        sessions/runs that reference it.
        """
        if not str(project_id).strip() or not str(name).strip():
            raise ValueError("project id and name must not be empty")
        with self._write_scope():
            self.conn.execute(
                """
                INSERT INTO projects(project_id, name, description, workspace_name, is_default)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    workspace_name = excluded.workspace_name,
                    is_default = excluded.is_default,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(project_id),
                    _redact_payload(name),
                    _redact_payload(description),
                    workspace_name,
                    1 if is_default else 0,
                ),
            )

    def get_project(self, project_id: str) -> dict | None:
        """Return one project row as a plain dict."""
        row = self.conn.execute(
            "SELECT project_id, name, description, workspace_name, is_default "
            "FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return self._project_row(row)

    def find_project_by_workspace(self, workspace_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT project_id, name, description, workspace_name, is_default "
            "FROM projects WHERE workspace_name = ?",
            (workspace_name,),
        ).fetchone()
        return self._project_row(row)

    def list_projects(self) -> list[dict]:
        """List projects, default first, then most recently touched."""
        rows = self.conn.execute(
            """
            SELECT project_id, name, description, workspace_name, is_default
            FROM projects ORDER BY is_default DESC, name COLLATE NOCASE
            """
        ).fetchall()
        return [self._project_row(row) for row in rows]

    @staticmethod
    def _project_row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {
            "project_id": str(row[0]),
            "name": str(row[1] or ""),
            "description": str(row[2] or ""),
            "workspace_name": row[3] if row[3] is None else str(row[3]),
            "is_default": bool(row[4]),
        }

    # -- WebUI login sessions --

    def insert_webui_session(
        self,
        session_id: str,
        token_hash: str,
        *,
        expires_at: float,
        created_at: float,
        remote_addr: str = "",
        user_agent: str = "",
    ) -> None:
        """Persist one browser login session (hashes only, never the token)."""
        with self._write_scope():
            self.conn.execute(
                """
                INSERT INTO webui_sessions
                    (session_id, token_hash, created_at, last_seen_at, expires_at,
                     remote_addr, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    token_hash,
                    float(created_at),
                    float(created_at),
                    float(expires_at),
                    _redact_payload(remote_addr),
                    _redact_payload(user_agent),
                ),
            )

    def find_webui_session_by_hash(self, token_hash: str) -> dict | None:
        """Return a live login session for a cookie hash, or ``None``."""
        row = self.conn.execute(
            """
            SELECT session_id, token_hash, created_at, last_seen_at, expires_at, revoked_at
            FROM webui_sessions WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row[0]),
            "token_hash": str(row[1]),
            "created_at": float(row[2]),
            "last_seen_at": float(row[3]),
            "expires_at": float(row[4]),
            "revoked_at": None if row[5] is None else float(row[5]),
        }

    def touch_webui_session(self, session_id: str, *, last_seen_at: float) -> None:
        with self._write_scope():
            self.conn.execute(
                "UPDATE webui_sessions SET last_seen_at = ? WHERE session_id = ?",
                (float(last_seen_at), session_id),
            )

    def revoke_webui_session(self, session_id: str, *, revoked_at: float) -> None:
        with self._write_scope():
            self.conn.execute(
                "UPDATE webui_sessions SET revoked_at = ? "
                "WHERE session_id = ? AND revoked_at IS NULL",
                (float(revoked_at), session_id),
            )

    def revoke_all_webui_sessions(self, *, revoked_at: float) -> int:
        """Revoke every live login session; returns the number revoked."""
        with self._write_scope():
            cursor = self.conn.execute(
                "UPDATE webui_sessions SET revoked_at = ? WHERE revoked_at IS NULL",
                (float(revoked_at),),
            )
            return int(cursor.rowcount or 0)

    def list_webui_sessions(self, *, limit: int = 20, live_only: bool = True) -> list[dict]:
        where = "WHERE revoked_at IS NULL" if live_only else ""
        rows = self.conn.execute(
            f"""
            SELECT session_id, created_at, last_seen_at, expires_at, revoked_at,
                   remote_addr
            FROM webui_sessions {where}
            ORDER BY last_seen_at DESC LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "session_id": str(r[0]),
                "created_at": float(r[1]),
                "last_seen_at": float(r[2]),
                "expires_at": float(r[3]),
                "revoked_at": None if r[4] is None else float(r[4]),
                "remote_addr": str(r[5] or ""),
            }
            for r in rows
        ]

    def record_webui_audit(self, event: str, detail: str = "", remote_addr: str = "") -> None:
        with self._write_scope():
            self.conn.execute(
                "INSERT INTO webui_audit(event, detail, remote_addr) VALUES (?, ?, ?)",
                (str(event), _redact_payload(detail), _redact_payload(remote_addr)),
            )

    def list_webui_audit(self, *, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT event, detail, remote_addr, created_at "
            "FROM webui_audit ORDER BY audit_id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "event": str(r[0]),
                "detail": str(r[1] or ""),
                "remote_addr": str(r[2] or ""),
                "created_at": r[3],
            }
            for r in rows
        ]

    # -- Research conversations --

    def list_agent_sessions(
        self,
        project_id: str,
        *,
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List conversation sessions scoped to one project."""
        rows = self.conn.execute(
            """
            SELECT s.session_id, s.title, s.status, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM agent_messages m WHERE m.session_id = s.session_id)
                       AS messages
            FROM agent_sessions s
            WHERE s.project_id = ? AND s.status = ?
            ORDER BY s.updated_at DESC, s.session_id
            LIMIT ? OFFSET ?
            """,
            (project_id, status, max(1, int(limit)), max(0, int(offset))),
        ).fetchall()
        return [
            {
                "session_id": str(r[0]),
                "title": str(r[1] or ""),
                "status": str(r[2]),
                "created_at": r[3],
                "updated_at": r[4],
                "messages": int(r[5] or 0),
            }
            for r in rows
        ]

    def count_agent_sessions(self, project_id: str, *, status: str = "active") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM agent_sessions WHERE project_id = ? AND status = ?",
            (project_id, status),
        ).fetchone()
        return int(row[0]) if row else 0

    def get_agent_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT session_id, title, system_prompt, status, model_config,
                   owner_principal, project_id, created_at, updated_at
            FROM agent_sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row[0]),
            "title": str(row[1] or ""),
            "system_prompt": str(row[2] or ""),
            "status": str(row[3]),
            "model_config": str(row[4] or "{}"),
            "owner_principal": str(row[5] or ""),
            "project_id": str(row[6] or ""),
            "created_at": row[7],
            "updated_at": row[8],
        }

    def set_session_project(self, session_id: str, project_id: str) -> None:
        with self._write_scope():
            self.conn.execute(
                "UPDATE agent_sessions SET project_id = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE session_id = ?",
                (project_id, session_id),
            )

    def get_agent_messages(
        self, session_id: str, *, after_seq: int = -1, limit: int = 200
    ) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT msg_id, seq, role, content, tool_name, created_at
            FROM agent_messages
            WHERE session_id = ? AND seq > ?
            ORDER BY seq LIMIT ?
            """,
            (session_id, int(after_seq), max(1, int(limit))),
        ).fetchall()
        return [
            {
                "msg_id": int(r[0]),
                "seq": int(r[1]),
                "role": str(r[2]),
                "content": str(r[3] or ""),
                "tool_name": str(r[4] or ""),
                "created_at": r[5],
            }
            for r in rows
        ]

    def next_agent_message_seq(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    # -- Session memory --

    def insert_session_memory(
        self,
        memory_id: str,
        project_id: str,
        session_id: str,
        layer: str,
        content: str,
        *,
        run_id: str = "",
        kind: str = "note",
        source_ref: str = "",
        dedup_key: str = "",
        now: float | None = None,
    ) -> bool:
        """Insert one memory entry; duplicate ``(project, session, dedup_key)`` is a no-op.

        Returns ``True`` when a new row was written.  Replaying the same run
        settlement therefore cannot duplicate memory, which is what makes the
        settle→memory write-back idempotent.
        """
        if layer not in ("project", "session", "run"):
            raise ValueError(f"invalid memory layer: {layer!r}")
        timestamp = time.time() if now is None else float(now)
        key = dedup_key or memory_id
        with self._write_scope():
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO session_memory
                    (memory_id, project_id, session_id, run_id, layer, kind,
                     content, source_ref, dedup_key, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    project_id,
                    session_id,
                    run_id,
                    layer,
                    kind,
                    _redact_payload(content),
                    source_ref,
                    key,
                    timestamp,
                    timestamp,
                ),
            )
            return bool(cursor.rowcount)

    def list_session_memory(
        self,
        project_id: str,
        *,
        session_id: str | None = None,
        layers: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses = ["project_id = ?"]
        params: list = [project_id]
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if layers:
            clauses.append("layer IN (" + ",".join("?" for _ in layers) + ")")
            params.extend(layers)
        params.append(max(1, int(limit)))
        rows = self.conn.execute(
            f"""
            SELECT memory_id, session_id, run_id, layer, kind, content, source_ref,
                   dedup_key, created_at, updated_at
            FROM session_memory WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC, memory_id LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "memory_id": str(r[0]),
                "session_id": str(r[1] or ""),
                "run_id": str(r[2] or ""),
                "layer": str(r[3]),
                "kind": str(r[4]),
                "content": str(r[5] or ""),
                "source_ref": str(r[6] or ""),
                "dedup_key": str(r[7] or ""),
                "created_at": r[8],
                "updated_at": r[9],
            }
            for r in rows
        ]

    # -- Plugin conformance --

    def insert_plugin_conformance(
        self,
        check_id: str,
        plugin_name: str,
        plugin_version: str,
        plugin_fingerprint: str,
        status: str,
        checks_json: str,
        *,
        completed_at: float | None = None,
    ) -> None:
        if status not in ("pending", "passed", "failed"):
            raise ValueError(f"invalid conformance status: {status!r}")
        with self._write_scope():
            self.conn.execute(
                """
                INSERT INTO plugin_conformance
                    (check_id, plugin_name, plugin_version, plugin_fingerprint,
                     status, checks_json, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(check_id) DO UPDATE SET
                    plugin_name = excluded.plugin_name,
                    plugin_version = excluded.plugin_version,
                    plugin_fingerprint = excluded.plugin_fingerprint,
                    status = excluded.status,
                    checks_json = excluded.checks_json,
                    completed_at = excluded.completed_at
                """,
                (
                    check_id,
                    plugin_name,
                    plugin_version,
                    plugin_fingerprint,
                    status,
                    checks_json,
                    completed_at,
                ),
            )

    def get_plugin_conformance(self, check_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT check_id, plugin_name, plugin_version, plugin_fingerprint,
                   status, checks_json, created_at, completed_at
            FROM plugin_conformance WHERE check_id = ?
            """,
            (check_id,),
        ).fetchone()
        return self._conformance_row(row)

    def list_plugin_conformance(self, plugin_name: str, *, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT check_id, plugin_name, plugin_version, plugin_fingerprint,
                   status, checks_json, created_at, completed_at
            FROM plugin_conformance WHERE plugin_name = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (plugin_name, max(1, int(limit))),
        ).fetchall()
        return [self._conformance_row(r) for r in rows]

    @staticmethod
    def _conformance_row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        checks: list = []
        try:
            parsed = json.loads(row[5] or "[]")
            if isinstance(parsed, list):
                checks = parsed
        except (TypeError, ValueError):
            checks = []
        return {
            "check_id": str(row[0]),
            "plugin_name": str(row[1]),
            "plugin_version": str(row[2] or ""),
            "plugin_fingerprint": str(row[3] or ""),
            "status": str(row[4]),
            "checks": checks,
            "created_at": row[6],
            "completed_at": row[7],
        }
