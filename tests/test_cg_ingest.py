"""Tests for corpus ingestion, citation harvesting and the v9 schema migration."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from drbrain.concept_graph.ingest import (
    ingest_citations,
    ingest_corpus,
    make_local_id,
)
from drbrain.concept_graph.sources.base import PaperRecord, PaperRelations
from drbrain.storage.database import Database


class FakeSource:
    """In-memory corpus source for ingestion tests."""

    name = "fake"

    def __init__(
        self, records: list[PaperRecord], relations: dict[str, PaperRelations] | None = None
    ):
        self._records = records
        self._relations = relations or {}

    def search(
        self, query=None, *, year_from=None, year_to=None, venues=None, sort=None, limit=100
    ) -> Iterator[PaperRecord]:
        yield from self._records[:limit]

    def fetch_relations(self, unique_id: str) -> PaperRelations | None:
        return self._relations.get(unique_id)

    def catalog(self) -> dict:
        return {}


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


def _rec(uid: str, doi: str | None = None, **kw) -> PaperRecord:
    return PaperRecord(unique_id=uid, title=f"T-{uid}", doi=doi, source="fake", **kw)


# ── schema migration ─────────────────────────────────────────────────────────


def test_schema_v9_tables_created() -> None:
    db, td = _tmp_db()
    try:
        tables = {
            r[0]
            for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for expected in (
            "corpus_sources",
            "concept_nodes",
            "concept_cooccurrence",
            "concept_embeddings",
            "paper_citations",
        ):
            assert expected in tables
        version = db.conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0]
        assert version >= 9
    finally:
        db.close()
        td.cleanup()


# ── local_id derivation ──────────────────────────────────────────────────────


def test_make_local_id_prefers_doi() -> None:
    assert make_local_id(_rec("u1", doi="10.1/ABC")) == "10.1/abc"


def test_make_local_id_falls_back_to_slug() -> None:
    lid = make_local_id(_rec("paper:10.1038/s41586"))
    assert lid.startswith("fake-")
    assert " " not in lid


# ── corpus ingestion ─────────────────────────────────────────────────────────


def test_ingest_corpus_inserts_papers() -> None:
    db, td = _tmp_db()
    try:
        src = FakeSource(
            [_rec("u1", doi="10.1/a", year=2020, abstract="abs"), _rec("u2", year=2021)]
        )
        stats = ingest_corpus(db, src, limit=10)
        assert stats.fetched == 2
        assert stats.inserted == 2
        count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert count == 2
        prov = db.conn.execute("SELECT COUNT(*) FROM corpus_sources").fetchone()[0]
        assert prov == 2
        abstract = db.conn.execute(
            "SELECT abstract FROM papers WHERE local_id='10.1/a'"
        ).fetchone()[0]
        assert abstract == "abs"
    finally:
        db.close()
        td.cleanup()


def test_ingest_corpus_dedup_by_unique_id() -> None:
    db, td = _tmp_db()
    try:
        src = FakeSource([_rec("u1", doi="10.1/a")])
        ingest_corpus(db, src, limit=10)
        stats2 = ingest_corpus(db, src, limit=10)
        assert stats2.inserted == 0
        assert stats2.skipped == 1
        count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert count == 1
    finally:
        db.close()
        td.cleanup()


def test_ingest_corpus_doi_secondary_dedup() -> None:
    db, td = _tmp_db()
    try:
        # Pre-existing paper keyed by DOI under a different local_id.
        db.insert_paper("existing_id", "Old", 2019, "uploaded")
        db.insert_paper_ids("existing_id", doi="10.1/a")
        db.conn.commit()

        src = FakeSource([_rec("u-new", doi="10.1/a")])
        stats = ingest_corpus(db, src, limit=10)
        assert stats.inserted == 0
        assert stats.skipped == 1
        # provenance linked to the existing local_id, no duplicate paper
        linked = db.find_corpus_source("fake", "u-new")
        assert linked == "existing_id"
        assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    finally:
        db.close()
        td.cleanup()


# ── citation harvesting ──────────────────────────────────────────────────────


def test_ingest_citations_writes_edges() -> None:
    db, td = _tmp_db()
    try:
        src = FakeSource(
            [_rec("u1", doi="10.1/a")],
            relations={
                "u1": PaperRelations(
                    unique_id="u1", references=[{"id": "10.9/z"}, {"id": "10.9/y"}]
                )
            },
        )
        ingest_corpus(db, src, limit=10)
        stats = ingest_citations(db, src)
        assert stats.citations == 2
        edges = db.conn.execute("SELECT COUNT(*) FROM paper_citations").fetchone()[0]
        assert edges == 2
    finally:
        db.close()
        td.cleanup()
