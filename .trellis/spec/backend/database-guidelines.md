# Database Guidelines

## Overview

- **Library**: Raw `sqlite3` — no ORM. `Database` class in `src/drbrain/storage/database.py`.
- **Schema**: Defined as `SCHEMA_SQL` string constant in `database.py`, executed via `executescript()`.
- **WAL mode**: `PRAGMA journal_mode=WAL` enabled in `Database.__init__()`.
- **Foreign keys**: `PRAGMA foreign_keys = ON`.

## Query Patterns

- Use `self.conn.execute(sql, params)` for parameterized queries.
- Use `self.conn.executemany(sql, seq)` for batch inserts.
- Always use `?` placeholders, never string interpolation.
- Return dicts via `dict(zip(cols, row))` pattern:
```python
rows = self.conn.execute("SELECT ...").fetchall()
cols = ["col1", "col2"]
return [dict(zip(cols, row)) for row in rows]
```

## Migrations

- **Version tracking**: `schema_versions` table `(version INTEGER PRIMARY KEY, applied_at TIMESTAMP)`.
- **Pattern**: Add `_migrate_add_X()` method, register in `_migrate()` with next version number.
```python
def _migrate_add_venue_columns(self):
    cols = [r[1] for r in self.conn.execute("PRAGMA table_info(papers)")]
    if "journal" not in cols:
        self.conn.execute("ALTER TABLE papers ADD COLUMN journal TEXT DEFAULT ''")
```
- Migrations run automatically on `Database.__init__()`.
- Each migration is idempotent (checks `PRAGMA table_info` before adding).
- `self.conn.commit()` after each migration step.

## Naming Conventions

- Table names: `snake_case` (`paper_ids`, `confidence_queue`, `research_seeds`).
- Column names: `snake_case` (`local_id`, `source_paper`, `created_at`).
- Index names: `idx_<table>_<column>` (`idx_concepts_type`, `idx_edges_relation`).
- Primary keys: `local_id` for papers (UUID-derived), `INTEGER PRIMARY KEY AUTOINCREMENT` for auto-id tables.
- Timestamps: `TIMESTAMP DEFAULT CURRENT_TIMESTAMP`.

## Common Mistakes

- Forgetting `self.conn.commit()` after writes — data is silently lost.
- Using `INSERT` instead of `INSERT OR IGNORE` for dedup tables (edges, aliases).
- Not checking `PRAGMA table_info` before `ALTER TABLE` in migrations.
- Using `:memory:` databases in tests — behaves differently from file-based. Always use `tempfile.TemporaryDirectory()`.
