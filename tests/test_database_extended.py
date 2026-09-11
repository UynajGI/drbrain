"""Tests for database methods not covered by existing tests."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest


def test_implicit_database_path_follows_runtime_root(tmp_path, monkeypatch):
    """Database() must not write to the process cwd when a root is selected."""
    from drbrain.storage.database import Database

    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
    db = Database()
    try:
        assert db.path == tmp_path / "data" / "drbrain.db"
        assert db.path.exists()
    finally:
        db.close()


def test_relative_database_path_follows_runtime_root(tmp_path, monkeypatch):
    """Direct callers cannot redirect a relative DB into the caller CWD."""
    from drbrain.storage.database import Database

    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
    db = Database("data/custom.sqlite")
    try:
        assert db.path == tmp_path / "data" / "custom.sqlite"
    finally:
        db.close()


@pytest.mark.parametrize("root_value", ["", "/definitely/not/a/runtime"])
def test_memory_database_ignores_invalid_runtime_selector(monkeypatch, root_value):
    """The SQLite in-memory sentinel must not require a valid disk root."""
    from drbrain.storage.database import Database

    monkeypatch.setenv("DRBRAIN_ROOT", root_value)

    db = Database(Path(":memory:"))
    try:
        assert db.conn.execute("SELECT 1").fetchone() == (1,)
        assert db.path == Path(":memory:")
    finally:
        db.close()


def test_explicit_external_database_path_is_rejected_when_root_is_selected(tmp_path, monkeypatch):
    """A direct Database caller cannot bypass an active runtime namespace."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside" / "library.sqlite"
    monkeypatch.setenv("DRBRAIN_ROOT", str(runtime_root))

    with pytest.raises(ValueError, match="database path.*escapes runtime root"):
        from drbrain.storage.database import Database

        Database(outside)


def test_database_rejects_sqlite_uri_without_runtime_root(monkeypatch):
    """The primary Database write surface must never accept SQLite URI syntax."""
    from drbrain.storage.database import Database

    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)

    with pytest.raises(ValueError, match="local filesystem path.*URI"):
        Database("file:/tmp/drbrain-hidden-test.db?mode=rwc")


def test_all_paper_id_write_helpers_fail_closed(tmp_db):
    """Every paper-bearing SQL helper shares the canonical ID validator."""
    invalid = " bad"
    calls = [
        lambda: tmp_db.set_paper_abstract(invalid, "text"),
        lambda: tmp_db.set_paper_categories(invalid, "cat"),
        lambda: tmp_db.insert_paper_cite_keys(invalid, ["ref"]),
        lambda: tmp_db.upgrade_placeholder(invalid),
        lambda: tmp_db.set_paper_status(invalid, "uploaded"),
        lambda: tmp_db.touch_paper(invalid),
        lambda: tmp_db.touch_edge("a", "b", "rel", invalid),
        lambda: tmp_db.update_paper_venue(invalid),
        lambda: tmp_db.insert_cooccurrence("a", "b", 2024, invalid),
        lambda: tmp_db.set_paper_field(invalid, "title", "text"),
        lambda: tmp_db.insert_argument(invalid, "claim", "supports", "target", "Method"),
        lambda: tmp_db.insert_queue_item(invalid, "gap", "{}", 0.5),
        lambda: tmp_db.record_evidence(invalid, "node-1"),
    ]

    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_record_answer_validates_evidence_before_persisting(tmp_db):
    """A malformed grounding must not leave a partial answer row behind."""
    with pytest.raises(ValueError):
        tmp_db.record_answer("question", "answer", evidence_ids=["bad\x00id:node"])

    assert tmp_db.conn.execute("SELECT COUNT(*) FROM answer_records").fetchone()[0] == 0


def test_record_answer_rolls_back_answer_and_evidence_as_one_unit(tmp_db, monkeypatch):
    """A late claim failure must not leave answer/evidence rows behind."""

    def fail_claim(*args, **kwargs):
        raise RuntimeError("claim write failed")

    monkeypatch.setattr(tmp_db, "record_claim", fail_claim)
    with pytest.raises(RuntimeError, match="claim write failed"):
        tmp_db.record_answer("question", "answer", evidence_ids=["p1:node"])

    assert tmp_db.conn.execute("SELECT COUNT(*) FROM answer_records").fetchone()[0] == 0
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_resolve_paper_cite_keys_validates_mapped_ids(tmp_db):
    """Deferred citation resolution cannot introduce an unsafe local ID."""
    tmp_db.insert_paper("p1", "Citing", 2024, "uploaded")
    tmp_db.insert_paper_cite_keys("p1", ["ref-key"])
    tmp_db.commit()

    with pytest.raises(ValueError):
        tmp_db.resolve_paper_cite_keys({"ref-key": " bad"})


@pytest.mark.parametrize("local_id", ["", " paper", "paper ", "paper\x00id"])
def test_database_rejects_unsafe_paper_identity(tmp_db, local_id):
    """The SQL write boundary shares the filesystem paper-ID contract."""
    with pytest.raises(ValueError):
        tmp_db.insert_paper(local_id, "Unsafe", 2024, "uploaded")


def test_paper_external_id_conflicts_fail_closed(tmp_db):
    """A unique external ID must never be silently assigned to another paper."""
    tmp_db.insert_paper("p1", "One", 2024, "uploaded")
    tmp_db.insert_paper_ids("p1", doi="DOI: https://doi.org/10.1234/ABC", strict=True)
    with pytest.raises(ValueError, match="doi.*already belongs"):
        tmp_db.insert_paper("p2", "Two", 2024, "uploaded")
        tmp_db.insert_paper_ids("p2", doi="10.1234/abc", strict=True)


def test_paper_external_ids_are_idempotently_merged(tmp_db):
    """Adding another identifier to an existing paper preserves one mapping row."""
    tmp_db.insert_paper("p1", "One", 2024, "uploaded")
    tmp_db.insert_paper_ids("p1", doi="10.1234/abc")
    tmp_db.insert_paper_ids("p1", arxiv="2401.00001v2")
    rows = tmp_db.execute("SELECT doi, arxiv FROM paper_ids WHERE local_id = 'p1'").fetchall()
    assert rows == [("10.1234/abc", "2401.00001")]


def test_strict_external_id_conflict_does_not_rewrite_prior_equivalent_id(tmp_db):
    """A late strict conflict must leave all earlier identifier fields untouched."""
    tmp_db.insert_paper("p1", "One", 2024, "uploaded")
    # Simulate a legacy spelling that is equivalent after normalization.
    tmp_db.conn.execute(
        "INSERT INTO paper_ids (local_id, doi, arxiv) VALUES (?, ?, ?)",
        ("p1", "DOI: 10.1/x", "2401.00001"),
    )
    tmp_db.commit()

    with pytest.raises(ValueError, match="arxiv.*already mapped"):
        tmp_db.insert_paper_ids("p1", doi="10.1/x", arxiv="2401.00002", strict=True)

    assert tmp_db.conn.execute(
        "SELECT doi, arxiv FROM paper_ids WHERE local_id = ?", ("p1",)
    ).fetchone() == ("DOI: 10.1/x", "2401.00001")


def test_merge_papers_composes_with_outer_transaction(tmp_db):
    """A successful merge must not commit unrelated caller writes."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.commit()

    tmp_db.insert_paper("pending", "Pending", 2025, "uploaded")
    tmp_db.merge_papers("keep", "gone")

    assert tmp_db.get_paper("pending") is not None
    assert tmp_db.get_paper("gone") is None
    tmp_db.conn.rollback()

    assert tmp_db.get_paper("pending") is None
    assert tmp_db.get_paper("gone") is not None


def test_merge_papers_preserves_richer_paper_metadata(tmp_db):
    """A rich source row must not disappear when it replaces a placeholder."""
    tmp_db.insert_paper(
        "keep",
        "Untitled",
        None,
        "placeholder",
        paper_type="paper",
        categories="ml",
    )
    tmp_db.insert_paper(
        "gone",
        "A richer title",
        2024,
        "extracted",
        paper_type="review",
        journal="Journal of Testing",
        publisher="Example Press",
        citation_count=42,
        volume="12",
        pages="1-20",
        authors="A. Author; B. Author",
        categories="physics, quantum",
    )
    tmp_db.set_paper_abstract("gone", "A complete abstract from the extracted record.")
    tmp_db.commit()

    tmp_db.merge_papers("keep", "gone")

    row = tmp_db.conn.execute(
        "SELECT title, abstract, year, status, paper_type, journal, publisher, "
        "citation_count, volume, pages, authors, categories "
        "FROM papers WHERE local_id = 'keep'"
    ).fetchone()
    assert row == (
        "A richer title",
        "A complete abstract from the extracted record.",
        2024,
        "extracted",
        "review",
        "Journal of Testing",
        "Example Press",
        42,
        "12",
        "1-20",
        "A. Author; B. Author",
        "ml physics quantum",
    )
    assert tmp_db.get_paper("gone") is None


def test_merge_papers_keeps_canonical_metadata_and_unions_authors(tmp_db):
    """A conflicting duplicate must not overwrite the canonical record silently."""
    tmp_db.insert_paper(
        "keep",
        "Correct descriptive title",
        2020,
        "uploaded",
        authors="Alice Smith",
    )
    tmp_db.insert_paper(
        "gone",
        "Short extraction label",
        2021,
        "extracted",
        authors="Bob Jones",
    )
    tmp_db.commit()

    tmp_db.merge_papers("keep", "gone")

    row = tmp_db.conn.execute(
        "SELECT title, year, authors FROM papers WHERE local_id = 'keep'"
    ).fetchone()
    assert row[0:2] == ("Correct descriptive title", 2020)
    assert set(part.strip() for part in row[2].split(";")) == {"Alice Smith", "Bob Jones"}


def test_merge_papers_clears_all_tree_and_embedding_shadows(tmp_db):
    """Identity changes invalidate both canonical and legacy derived indexes."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.conn.executemany(
        "INSERT INTO embeddings (entity, vec, dim) VALUES (?, ?, ?)",
        [("keep", b"k", 1), ("gone", b"g", 1), ("other", b"o", 1), ("__rel__cites", b"r", 1)],
    )
    tmp_db.conn.executemany(
        "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("keep:n1", "keep", b"k", "h", "pageindex"),
            ("raptor_keep_L1_old", "keep", b"k", "h", "raptor_L1"),
            ("gone:n1", "gone", b"g", "h", "pageindex"),
            ("raptor_gone_L1_old", "gone", b"g", "h", "raptor_L1"),
        ],
    )
    for table in (
        "tree_vectors_vec",
        "tree_vectors_vec_f32_bak",
        "tree_vectors_vec_i8",
        "vec_i8_scale",
    ):
        if table == "vec_i8_scale":
            tmp_db.conn.execute(
                f"CREATE TABLE {table} (node_id TEXT PRIMARY KEY, scale BLOB NOT NULL)"
            )
            tmp_db.conn.executemany(
                f"INSERT INTO {table} VALUES (?, ?)",
                [("raptor_gone_L1_orphan", b"s"), ("0001", b"legacy")],
            )
        else:
            tmp_db.conn.execute(f"CREATE TABLE {table} (node_id TEXT PRIMARY KEY, embedding BLOB)")
            tmp_db.conn.executemany(
                f"INSERT INTO {table} VALUES (?, ?)",
                [
                    ("raptor_gone_L1_orphan", b"v"),
                    ("0001", b"legacy"),
                    ("other:n1", b"other"),
                ],
            )
    tmp_db.conn.execute("CREATE TABLE vector_meta (key TEXT PRIMARY KEY, value TEXT)")
    tmp_db.conn.execute("INSERT INTO vector_meta VALUES ('synced', '1')")
    tmp_db.commit()

    counts = tmp_db.merge_papers("keep", "gone")

    assert counts["embeddings"] == 4
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM tree_vectors WHERE paper_id IN ('keep', 'gone')"
        ).fetchone()[0]
        == 0
    )
    for table in (
        "tree_vectors_vec",
        "tree_vectors_vec_f32_bak",
        "tree_vectors_vec_i8",
        "vec_i8_scale",
    ):
        assert tmp_db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert (
        tmp_db.conn.execute("SELECT value FROM vector_meta WHERE key = 'synced'").fetchone()[0]
        == "0"
    )


def test_embedding_revision_advances_when_vectors_are_cleared(tmp_db):
    """Readers can detect a cleared TransE cache without trusting old rows."""
    assert tmp_db.get_embedding_revision() == 0
    tmp_db.save_embedding("entity", [1.0], 1)
    first = tmp_db.get_embedding_revision()
    assert first >= 1
    assert tmp_db.clear_embeddings() == 1
    assert tmp_db.get_embedding_revision() > first
    assert tmp_db.load_embeddings() == {}


def test_delete_paper_clears_legacy_tree_vector_shadows(tmp_db):
    """Deleting a paper cannot leave RAPTOR or unscoped ANN rows behind."""
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.conn.execute(
        "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
        "VALUES ('raptor_gone_L1_old', 'gone', ?, 'h', 'raptor_L1')",
        (b"g",),
    )
    tmp_db.conn.execute("CREATE TABLE tree_vectors_vec (node_id TEXT PRIMARY KEY, embedding BLOB)")
    tmp_db.conn.executemany(
        "INSERT INTO tree_vectors_vec VALUES (?, ?)",
        [("raptor_gone_L1_old", b"g"), ("0001", b"legacy"), ("other:n1", b"other")],
    )
    tmp_db.commit()

    tmp_db.delete_paper("gone")

    assert tmp_db.conn.execute("SELECT COUNT(*) FROM tree_vectors_vec").fetchone()[0] == 0


def test_merge_papers_fails_before_mutation_for_unloadable_vec0(tmp_db, monkeypatch):
    """A virtual ANN table requires an extension before identity migration."""
    sqlite_vec = pytest.importorskip("sqlite_vec")
    from drbrain.storage.database import Database

    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    try:
        tmp_db.conn.enable_load_extension(True)
        sqlite_vec.load(tmp_db.conn)
        tmp_db.conn.enable_load_extension(False)
        tmp_db.conn.execute(
            "CREATE VIRTUAL TABLE tree_vectors_vec USING vec0(node_id TEXT PRIMARY KEY, embedding float[1])"
        )
        tmp_db.conn.execute(
            "INSERT INTO tree_vectors_vec VALUES ('raptor_gone_L1_old', ?)", (b"\x00\x00\x80?",)
        )
        tmp_db.commit()
    except Exception as exc:
        pytest.skip(f"sqlite-vec virtual table unavailable: {exc}")

    # Reopen the file first: this models the production failure mode where the
    # vec0 module was registered by a backfill process but is absent from a
    # later ordinary Database connection.  The fixture connection is closed;
    # its path remains available for the replacement handle.
    db_path = tmp_db.path
    tmp_db.close()
    reopened = Database(db_path)

    # Force the production preflight to observe an unavailable loader while
    # retaining the actual virtual table in the database.
    monkeypatch.setattr("drbrain.storage.database._load_sqlite_vec", lambda conn: False)
    with pytest.raises(ValueError, match="sqlite-vec"):
        reopened.merge_papers("keep", "gone")
    assert reopened.get_paper("gone") is not None
    assert reopened.get_paper("keep") is not None
    reopened.close()


def test_merge_papers_migrates_identity_and_provenance_rows(tmp_db):
    """Merging duplicate papers must preserve source identity and paper data."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "extracted")
    tmp_db.insert_paper("other", "Other", 2022, "uploaded")
    tmp_db.insert_paper_ids("keep", doi="10.1000/keeper")
    tmp_db.insert_paper_ids("gone", arxiv="2401.00001v2", s2_id="s2-gone")
    tmp_db.insert_corpus_source("gone", "openalex", "W-gone")
    tmp_db.insert_paper_term("gone", "quantum", "keyword")
    tmp_db.insert_cooccurrence("alpha", "beta", 2023, "gone", weight=2.0)
    tmp_db.insert_paper_cite_keys("gone", ["ref-gone"])
    tmp_db.conn.execute(
        "INSERT INTO paper_cite_keys (citing_local_id, cited_key, cited_local_id) VALUES (?, ?, ?)",
        ("other", "ref-gone-incoming", "gone"),
    )
    tmp_db.insert_paper_citation("gone", "other", source="openalex", year=2023)
    tmp_db.insert_paper_citation("other", "gone", source="openalex", year=2022)
    tmp_db.insert_citation_cache(
        "gone", "A cited work", 2021, "references", target_doi="10.1000/cited"
    )
    tmp_db.upsert_build_stage("gone", "ontology", "complete", '{"concepts": 1}')
    tmp_db.insert_queue_item("gone", "concept", '{"label":"alpha"}', 0.4)
    tmp_db.conn.execute(
        "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        ("gone:n1", "gone", b"vec", "h", "pageindex"),
    )
    tmp_db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        ("gone:s1", "gone", "summary", "gone:n1", 1),
    )
    evidence_id = tmp_db.record_evidence("gone", "n1", snippet="source passage")
    tmp_db.commit()

    counts = tmp_db.merge_papers("keep", "gone")

    assert counts["paper_ids"] == 1
    ids = tmp_db.conn.execute(
        "SELECT doi, arxiv, s2_id FROM paper_ids WHERE local_id = ?", ("keep",)
    ).fetchone()
    assert ids == ("10.1000/keeper", "2401.00001", "s2-gone")
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM paper_ids WHERE local_id = ?", ("gone",)
        ).fetchone()[0]
        == 0
    )
    assert (
        tmp_db.conn.execute(
            "SELECT local_id FROM corpus_sources WHERE source_unique_id = ?", ("W-gone",)
        ).fetchone()[0]
        == "keep"
    )
    assert (
        tmp_db.conn.execute(
            "SELECT term FROM paper_terms WHERE local_id = ?", ("keep",)
        ).fetchone()[0]
        == "quantum"
    )
    assert (
        tmp_db.conn.execute(
            "SELECT weight FROM concept_cooccurrence WHERE paper_id = ?", ("keep",)
        ).fetchone()[0]
        == 2.0
    )
    assert (
        tmp_db.conn.execute(
            "SELECT cited_local_id FROM paper_cite_keys "
            "WHERE citing_local_id = 'other' AND cited_key = 'ref-gone-incoming'"
        ).fetchone()[0]
        == "keep"
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM paper_cite_keys "
            "WHERE citing_local_id = 'keep' AND cited_key = 'ref-gone'"
        ).fetchone()[0]
        == 1
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM paper_citations WHERE citing_id = 'gone' OR cited_id = 'gone'"
        ).fetchone()[0]
        == 0
    )
    assert {
        tuple(row)
        for row in tmp_db.conn.execute(
            "SELECT citing_id, cited_id FROM paper_citations ORDER BY citing_id, cited_id"
        )
    } == {("keep", "other"), ("other", "keep")}
    assert (
        tmp_db.conn.execute(
            "SELECT source_paper FROM citation_cache WHERE target_title = 'A cited work'"
        ).fetchone()[0]
        == "keep"
    )
    assert tmp_db.conn.execute(
        "SELECT paper_id, status, result_json FROM build_stages WHERE stage = 'ontology'"
    ).fetchone() == ("keep", "complete", "")
    assert (
        tmp_db.conn.execute(
            "SELECT source_paper FROM confidence_queue WHERE item_type = 'concept'"
        ).fetchone()[0]
        == "keep"
    )
    # Derived tree indexes are invalidated rather than copied with stale
    # ``gone:`` node IDs; the target can rebuild them from its canonical tree.
    assert (
        tmp_db.conn.execute("SELECT COUNT(*) FROM tree_vectors WHERE paper_id = 'gone'").fetchone()[
            0
        ]
        == 0
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM tree_summaries WHERE paper_id = 'gone'"
        ).fetchone()[0]
        == 0
    )
    assert tmp_db.conn.execute(
        "SELECT paper_id, node_id, snippet FROM evidence WHERE evidence_id = ?",
        (evidence_id,),
    ).fetchone() == ("keep", "n1", "source passage")
    assert tmp_db.get_paper("gone") is None


def test_merge_papers_retargets_paper_node_edge_endpoints(tmp_db):
    """A paper local ID used as an edge endpoint is redirected on merge."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.insert_paper("other", "Other", 2022, "uploaded")
    # These edges are asserted by another paper, so moving only source_paper
    # would leave ``gone`` as a dangling graph node.
    tmp_db.insert_edge("gone", "other", "cites", "other")
    tmp_db.insert_edge("other", "gone", "cites", "other")
    tmp_db.commit()

    counts = tmp_db.merge_papers("keep", "gone")

    assert counts["edges_redirected"] >= 2
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?", ("gone", "gone")
        ).fetchone()[0]
        == 0
    )
    assert {
        tuple(row)
        for row in tmp_db.conn.execute(
            "SELECT src_id, dst_id, source_paper FROM edges "
            "WHERE source_paper = 'other' ORDER BY src_id, dst_id"
        )
    } == {("keep", "other", "other"), ("other", "keep", "other")}


def test_merge_papers_preserves_self_citation_and_cleans_vector_shadows(tmp_db):
    """Identity migration cannot silently drop self-citations or ANN orphans."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.insert_paper_citation("gone", "gone", source="openalex", year=2023)
    tmp_db.conn.executemany(
        "INSERT INTO embeddings (entity, vec, dim) VALUES (?, ?, ?)",
        [("keep", b"k", 1), ("gone", b"g", 1), ("other", b"o", 1)],
    )
    # Model the active vec table plus an interrupted quantization backup.  The
    # orphan row intentionally has no tree_vectors base counterpart.
    tmp_db.conn.execute("CREATE TABLE tree_vectors_vec (node_id TEXT PRIMARY KEY, embedding BLOB)")
    tmp_db.conn.execute(
        "CREATE TABLE tree_vectors_vec_f32_bak (node_id TEXT PRIMARY KEY, embedding BLOB)"
    )
    tmp_db.conn.execute("CREATE TABLE vec_i8_scale (node_id TEXT PRIMARY KEY, scale BLOB NOT NULL)")
    tmp_db.conn.execute("CREATE TABLE vector_meta (key TEXT PRIMARY KEY, value TEXT)")
    tmp_db.conn.executemany(
        "INSERT INTO tree_vectors_vec VALUES (?, ?)",
        [("gone:n1", b"1"), ("gone:orphan", b"2"), ("keep:n2", b"3")],
    )
    tmp_db.conn.execute("INSERT INTO tree_vectors_vec_f32_bak VALUES (?, ?)", ("gone:n1", b"4"))
    tmp_db.conn.execute("INSERT INTO vec_i8_scale VALUES (?, ?)", ("gone:orphan", b"5"))
    tmp_db.conn.execute("INSERT INTO vector_meta VALUES ('synced', '1')")
    tmp_db.commit()

    tmp_db.merge_papers("keep", "gone")

    assert tmp_db.conn.execute(
        "SELECT source, year FROM paper_citations WHERE citing_id = 'keep' AND cited_id = 'keep'"
    ).fetchone() == ("openalex", 2023)
    assert tmp_db.conn.execute("SELECT entity FROM embeddings ORDER BY entity").fetchall() == []
    for table in ("tree_vectors_vec", "tree_vectors_vec_f32_bak", "vec_i8_scale"):
        assert (
            tmp_db.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE substr(node_id, 1, length(?) + 1) = ? || ':'",
                ("gone", "gone"),
            ).fetchone()[0]
            == 0
        )
    assert (
        tmp_db.conn.execute("SELECT value FROM vector_meta WHERE key = 'synced'").fetchone()[0]
        == "0"
    )


def test_merge_papers_canonicalizes_resolved_cite_key_targets_before_merge(tmp_db):
    """Equivalent keep/source cite-key rows must not false-conflict."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.conn.execute(
        "INSERT INTO paper_cite_keys (citing_local_id, cited_key, cited_local_id) "
        "VALUES ('keep', 'same-key', 'gone')"
    )
    tmp_db.conn.execute(
        "INSERT INTO paper_cite_keys (citing_local_id, cited_key, cited_local_id) "
        "VALUES ('gone', 'same-key', 'gone')"
    )
    tmp_db.commit()

    tmp_db.merge_papers("keep", "gone")

    assert (
        tmp_db.conn.execute(
            "SELECT cited_local_id FROM paper_cite_keys "
            "WHERE citing_local_id = 'keep' AND cited_key = 'same-key'"
        ).fetchone()[0]
        == "keep"
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM paper_cite_keys WHERE citing_local_id = 'gone'"
        ).fetchone()[0]
        == 0
    )


def test_merge_papers_rejects_conflicting_citation_cache_metadata(tmp_db):
    """A same-title cache collision with different IDs/years must roll back."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.insert_citation_cache(
        "keep",
        "Ambiguous target",
        2020,
        "references",
        target_doi="10.1000/keep-target",
    )
    tmp_db.insert_citation_cache(
        "gone",
        "Ambiguous target",
        2021,
        "references",
        target_doi="10.1000/gone-target",
    )
    tmp_db.commit()

    with pytest.raises(ValueError, match="citation cache conflict"):
        tmp_db.merge_papers("keep", "gone")

    assert tmp_db.get_paper("gone") is not None
    assert tmp_db.conn.execute(
        "SELECT source_paper, target_year, target_doi FROM citation_cache "
        "WHERE target_title = 'Ambiguous target' ORDER BY source_paper"
    ).fetchall() == [
        ("gone", 2021, "10.1000/gone-target"),
        ("keep", 2020, "10.1000/keep-target"),
    ]


def test_merge_papers_external_id_conflict_is_atomic(tmp_db):
    """A conflicting same-kind ID aborts before any source row is migrated."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    tmp_db.insert_paper_ids("keep", doi="10.1000/keeper")
    tmp_db.insert_paper_ids("gone", doi="10.1000/goner")
    tmp_db.insert_corpus_source("gone", "openalex", "W-gone")
    tmp_db.commit()

    with pytest.raises(ValueError, match="doi.*conflict"):
        tmp_db.merge_papers("keep", "gone")

    assert tmp_db.get_paper("gone") is not None
    assert (
        tmp_db.conn.execute(
            "SELECT local_id FROM corpus_sources WHERE source_unique_id = ?", ("W-gone",)
        ).fetchone()[0]
        == "gone"
    )
    assert (
        tmp_db.conn.execute("SELECT doi FROM paper_ids WHERE local_id = ?", ("keep",)).fetchone()[0]
        == "10.1000/keeper"
    )


@pytest.mark.parametrize(
    ("keep_id", "merge_id"),
    [("same", "same"), ("missing", "present"), ("present", "missing")],
)
def test_merge_papers_rejects_invalid_identity_pairs(tmp_db, keep_id, merge_id):
    """Merge must never delete a paper when either identity is invalid."""
    tmp_db.insert_paper("present", "Present", 2024, "uploaded")
    tmp_db.commit()

    with pytest.raises(ValueError, match="different papers|not found"):
        tmp_db.merge_papers(keep_id, merge_id)

    assert tmp_db.get_paper("present") is not None


def test_merge_papers_failure_only_rolls_back_its_savepoint(tmp_db):
    """A failed nested merge must preserve the caller transaction and source paper."""
    tmp_db.insert_paper("keep", "Keeper", 2024, "uploaded")
    tmp_db.insert_paper("gone", "Goner", 2023, "uploaded")
    # Redirecting gone's source_paper to keep would collide with this row.
    tmp_db.insert_edge("src", "dst", "rel", "keep")
    tmp_db.insert_edge("src", "dst", "rel", "gone")
    tmp_db.commit()

    tmp_db.insert_paper("pending", "Pending", 2025, "uploaded")
    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.merge_papers("keep", "gone")

    assert tmp_db.get_paper("pending") is not None
    assert tmp_db.get_paper("gone") is not None
    tmp_db.conn.rollback()
    assert tmp_db.get_paper("pending") is None
    assert tmp_db.get_paper("gone") is not None


def test_agent_writers_redact_direct_low_level_payloads(tmp_db):
    """Direct Database callers cannot persist credentials in session rows."""
    tmp_db.insert_agent_session(
        "session-1",
        title="run api_key=title-secret",
        system_prompt='{"nested":{"Authorization":"Bearer prompt-secret"}}',
        model_config='{"provider":"openai","headers":{"api_key":"config-secret"}}',
    )
    tmp_db.insert_agent_message(
        "session-1",
        1,
        "assistant",
        content="provider failed: token=content-secret",
        tool_calls_json=('{"function":{"arguments":"{\\"api_key\\":\\"call-secret\\"}"}}'),
        tool_call_id="call_1",
        tool_name="tool-secret=tool-name-secret",
    )

    session = tmp_db.conn.execute(
        "SELECT title, system_prompt, model_config FROM agent_sessions WHERE session_id = ?",
        ("session-1",),
    ).fetchone()
    message = tmp_db.conn.execute(
        "SELECT content, tool_calls_json, tool_call_id, tool_name "
        "FROM agent_messages WHERE session_id = ?",
        ("session-1",),
    ).fetchone()
    rendered = "\n".join(str(value) for value in (*session, *message))
    for secret in (
        "title-secret",
        "prompt-secret",
        "config-secret",
        "content-secret",
        "call-secret",
        "tool-name-secret",
    ):
        assert secret not in rendered
    assert message[2] == "call_1"
    assert json.loads(session[2])["headers"]["api_key"] == "[REDACTED]"
    nested_args = json.loads(json.loads(message[1])["function"]["arguments"])
    assert nested_args["api_key"] == "[REDACTED]"


def test_epistemic_writers_do_not_commit_an_outer_transaction(tmp_db):
    """Snapshot, evidence, claim, and binding writers compose with callers."""
    tmp_db.insert_paper("pending", "Pending", 2025, "uploaded")
    snapshot_id = tmp_db.record_knowledge_snapshot("snap-pending")
    evidence_id = tmp_db.record_evidence("pending", "node", evidence_id="ev-pending")
    claim_id = tmp_db.record_claim("question", "answer")
    tmp_db.record_claim_evidence(claim_id, [evidence_id])

    assert tmp_db.conn.in_transaction
    tmp_db.conn.rollback()

    assert tmp_db.get_paper("pending") is None
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM knowledge_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        tmp_db.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()[0]
        == 0
    )


def test_execute_and_commit(tmp_db):
    """execute returns a cursor, commit persists changes."""
    cur = tmp_db.execute("SELECT 1")
    assert cur.fetchone() == (1,)
    tmp_db.commit()


def test_executemany(tmp_db):
    """executemany inserts multiple rows."""
    tmp_db.insert_paper("p1", "A", 2020, "uploaded")
    tmp_db.insert_paper("p2", "B", 2021, "uploaded")
    tmp_db.insert_paper("p3", "C", 2022, "uploaded")
    tmp_db.commit()

    papers = tmp_db.get_all_papers()
    assert len(papers) == 3


def test_get_paper_not_found(tmp_db):
    """get_paper returns None for unknown ID."""
    assert tmp_db.get_paper("nonexistent") is None


def test_upgrade_placeholder(tmp_db):
    """upgrade_placeholder changes status from placeholder to uploaded."""
    tmp_db.insert_paper("p1", "Test", 2024, "placeholder")
    tmp_db.commit()

    before = tmp_db.get_paper("p1")
    assert before["status"] == "placeholder"

    tmp_db.upgrade_placeholder("p1")
    tmp_db.commit()

    after = tmp_db.get_paper("p1")
    assert after["status"] == "uploaded"


def test_upgrade_placeholder_noop_for_uploaded(tmp_db):
    """upgrade_placeholder does nothing for already uploaded papers."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    tmp_db.commit()

    tmp_db.upgrade_placeholder("p1")
    tmp_db.commit()

    paper = tmp_db.get_paper("p1")
    assert paper["status"] == "uploaded"


def test_insert_and_get_concepts_by_paper(tmp_db):
    """insert_concept + get_concepts_by_paper round-trip."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    cid = tmp_db.insert_concept("p1", "Problem", "ML Scalability", confidence=0.95, year=2024)
    assert cid is not None

    concepts = tmp_db.get_concepts_by_paper("p1")
    assert len(concepts) == 1
    assert concepts[0]["label"] == "ML Scalability"
    assert concepts[0]["type"] == "Problem"
    assert concepts[0]["confidence"] == 0.95


def test_insert_alias(tmp_db):
    """insert_alias stores variant->canonical mapping."""
    # Need a paper first (concepts FK to papers)
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    # Need a concept first for alias FK
    cid = tmp_db.insert_concept("p1", "Method", "transformers", year=2024)
    tmp_db.insert_alias("transformers", str(cid))
    tmp_db.insert_alias("Transformer", str(cid))
    tmp_db.commit()

    row = tmp_db.conn.execute(
        "SELECT canonical_id FROM aliases WHERE variant='transformers'"
    ).fetchone()
    assert row[0] == str(cid)


def test_insert_and_get_seeds(tmp_db):
    """insert_seed + get_all_seeds round-trip."""
    sid = tmp_db.insert_seed("unaddressed_gap", "No method addresses Gap X", confidence=0.8)
    assert sid is not None

    seeds = tmp_db.get_all_seeds()
    assert len(seeds) == 1
    assert seeds[0]["pattern_type"] == "unaddressed_gap"


def test_delete_seed(tmp_db):
    """delete_seed removes a research seed."""
    sid = tmp_db.insert_seed("test", "Test seed")
    tmp_db.commit()

    assert len(tmp_db.get_all_seeds()) == 1
    tmp_db.delete_seed(sid)
    tmp_db.commit()
    assert len(tmp_db.get_all_seeds()) == 0


def test_insert_and_get_arguments(tmp_db):
    """insert_argument + get_arguments_by_paper round-trip."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    aid = tmp_db.insert_argument(
        "p1",
        "Method X outperforms Y",
        "supports",
        "Method X",
        "Method",
        "empirical",
        "See Table 3",
        0.9,
    )
    assert aid is not None

    args = tmp_db.get_arguments_by_paper("p1")
    assert len(args) == 1
    assert args[0]["claim"] == "Method X outperforms Y"
    assert args[0]["claim_type"] == "supports"


def test_confidence_queue_lifecycle(tmp_db):
    """Queue items flow: insert -> pending -> accept/reject."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "Test Concept"}', 0.6)
    assert qid is not None

    pending = tmp_db.get_queue_pending()
    assert len(pending) == 1
    assert pending[0]["queue_id"] == qid

    tmp_db.accept_queue_item(qid)
    tmp_db.commit()
    assert len(tmp_db.get_queue_pending()) == 0


def test_queue_reject(tmp_db):
    """reject_queue_item marks item as rejected."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.5)
    tmp_db.commit()

    tmp_db.reject_queue_item(qid)
    tmp_db.commit()

    status = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()[0]
    assert status == "rejected"


def test_insert_edge_dedup(tmp_db):
    """Duplicate edges are ignored (INSERT OR IGNORE)."""
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_edge("p1", "p2", "cites", "p1")  # Duplicate
    tmp_db.commit()

    count = tmp_db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert count == 1


def test_save_paper_artifacts():
    """_save_paper_artifacts writes raw.md, source.pdf, and images into per-paper dir."""
    import tempfile
    from types import SimpleNamespace

    from drbrain.cli.commands import _save_paper_artifacts

    with tempfile.TemporaryDirectory() as td:
        paper_dir = Path(td) / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        # Create a fake source PDF
        src_pdf = Path(td) / "input.pdf"
        src_pdf.write_bytes(b"fake pdf content")

        parsed = SimpleNamespace(
            raw_md="# Title\n\nAbstract text here.",
            images_dir=None,
        )
        _save_paper_artifacts(parsed, "p1", paper_dir, src_pdf)
        assert (paper_dir / "raw.md").exists()
        assert (paper_dir / "source.pdf").exists()
        content = (paper_dir / "raw.md").read_text()
        assert "Title" in content


def test_save_paper_artifacts_copies_images():
    """_save_paper_artifacts copies images into per-paper dir."""
    import tempfile
    from types import SimpleNamespace

    from drbrain.cli.commands import _save_paper_artifacts

    with tempfile.TemporaryDirectory() as td:
        paper_dir = Path(td) / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        src_pdf = Path(td) / "input.pdf"
        src_pdf.write_bytes(b"fake pdf")

        # Create source images dir
        img_dir = Path(td) / "src_images"
        img_dir.mkdir()
        (img_dir / "abc.jpg").write_bytes(b"fake image")

        parsed = SimpleNamespace(
            raw_md="# Title\n\n![img](images/abc.jpg)\n\nAbstract.",
            images_dir=img_dir,
        )
        _save_paper_artifacts(parsed, "p1", paper_dir, src_pdf)

        assert (paper_dir / "images" / "abc.jpg").exists()
        content = (paper_dir / "raw.md").read_text()
        assert "images/abc.jpg" in content


def test_save_paper_artifacts_rejects_symlinked_image_destination(tmp_path):
    """copytree must not follow a pre-existing images directory alias."""
    from types import SimpleNamespace

    from drbrain.cli.commands import _save_paper_artifacts

    paper_dir = tmp_path / "papers" / "p1"
    paper_dir.mkdir(parents=True)
    source_pdf = tmp_path / "input.pdf"
    source_pdf.write_bytes(b"fake pdf")
    source_images = tmp_path / "source_images"
    source_images.mkdir()
    (source_images / "x.jpg").write_bytes(b"image")
    external = tmp_path / "external"
    external.mkdir()
    (paper_dir / "images").symlink_to(external, target_is_directory=True)

    parsed = SimpleNamespace(raw_md="# title", images_dir=source_images)
    with pytest.raises(ValueError, match="symlink"):
        _save_paper_artifacts(parsed, "p1", paper_dir, source_pdf)
    assert not (external / "x.jpg").exists()


# -- Volume/pages interface tests --


def test_insert_paper_accepts_volume_pages(tmp_db):
    """insert_paper must accept and store volume/pages."""
    db = tmp_db
    db.insert_paper("ptest", "Test Paper", 2024, "uploaded", volume="42", pages="100-120")
    p = db.get_paper("ptest")
    assert p["volume"] == "42"
    assert p["pages"] == "100-120"


def test_get_paper_returns_volume_pages(tmp_db):
    """get_paper must include volume and pages in result dict."""
    db = tmp_db
    db.insert_paper("ptest2", "Test", 2023, "uploaded", volume="10", pages="50-55")
    p = db.get_paper("ptest2")
    assert "volume" in p
    assert "pages" in p
    assert p["volume"] == "10"
    assert p["pages"] == "50-55"


def test_insert_paper_volume_pages_default_empty(tmp_db):
    """insert_paper defaults volume/pages to empty strings."""
    db = tmp_db
    db.insert_paper("ptest3", "Test", 2022, "uploaded")
    p = db.get_paper("ptest3")
    assert p["volume"] == ""
    assert p["pages"] == ""


# ── Temporal evolution signals ──────────────────────────────────


def _seed_papers_and_concepts(db, label, ctype, year_confidence_pairs):
    for i, (year, conf) in enumerate(year_confidence_pairs):
        pid = f"p{i:03d}_{label.replace(' ', '_')}"
        db.insert_paper(pid, f"Paper about {label} ({year})", year, "uploaded")
        db.insert_concept(pid, ctype, label, conf, year=year)
    db.commit()


def test_signal_emerging(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "quantum transformer",
        "Method",
        [
            (current - 2, 0.9),
            (current - 1, 0.88),
            (current - 1, 0.91),
            (current, 0.85),
            (current, 0.90),
            (current, 0.87),
            (current, 0.92),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "quantum transformer"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "emerging"


def test_signal_established(tmp_db):
    current = datetime.now().year
    pairs = [(current - 5 + (i % 6), 0.85 + (i % 10) * 0.01) for i in range(12)]
    _seed_papers_and_concepts(tmp_db, "attention mechanism", "Method", pairs)
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "attention mechanism"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "established"


def test_signal_declining(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "rnn language model",
        "Method",
        [
            (current - 8, 0.9),
            (current - 7, 0.88),
            (current - 5, 0.85),
            (current - 4, 0.82),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "rnn language model"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "declining"


def test_signal_contested(tmp_db):
    _seed_papers_and_concepts(
        tmp_db,
        "consciousness in llm",
        "Debate",
        [
            (2023, 0.5),
            (2023, 0.6),
            (2024, 0.55),
            (2024, 0.65),
            (2024, 0.45),
            (2025, 0.6),
            (2025, 0.5),
            (2025, 0.7),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "consciousness in llm"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "contested"


def test_signal_resurging(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "symbolic ai",
        "Method",
        [
            (current - 10, 0.9),
            (current - 9, 0.88),
            (current - 8, 0.85),
            (current - 1, 0.75),
            (current, 0.80),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "symbolic ai"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "resurging"


def test_signal_unknown(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "obscure method",
        "Method",
        [
            (current - 2, 0.9),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "obscure method"]
    assert len(matching) == 1
    assert matching[0]["signal"] in ("unknown", "established")


def test_get_concept_signal(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "transformer",
        "Method",
        [
            (current - 5, 0.95),
            (current - 4, 0.93),
            (current - 4, 0.91),
            (current - 3, 0.90),
            (current - 3, 0.88),
        ],
    )
    signal = tmp_db.get_concept_signal("transformer")
    assert signal is not None
    assert "label" in signal
    assert "signal" in signal
    assert tmp_db.get_concept_signal("nonexistent") is None


def test_get_concept_evolution(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "diffusion model",
        "Method",
        [
            (current - 3, 0.9),
            (current - 2, 0.88),
            (current - 2, 0.91),
            (current - 1, 0.85),
            (current - 1, 0.90),
            (current - 1, 0.87),
        ],
    )
    evolution = tmp_db.get_concept_evolution("diffusion model")
    assert len(evolution) == 3
    assert evolution[0]["year"] == current - 3
    assert "trend" in evolution[0]
    last = evolution[-1]
    assert last["year"] == current - 1
    assert last["count"] == 3


# ── get_stats ─────────────────────────────────────────────────────


def test_get_stats_returns_counts(tmp_db):
    """get_stats returns zero counts for an empty database."""
    stats = tmp_db.get_stats()
    assert stats["papers"] == 0
    assert stats["concepts"] == 0
    assert stats["edges"] == 0
    assert stats["arguments"] == 0
    assert stats["aliases"] == 0
    assert stats["research_seeds"] == 0
    assert stats["queue_pending"] == 0
    assert stats["uploaded"] == 0
    assert stats["placeholders"] == 0


def test_get_stats_with_data(tmp_db):
    """get_stats returns correct counts after inserting data."""
    tmp_db.insert_paper("p1", "A", 2024, "extracted")
    tmp_db.insert_paper("p2", "B", 2024, "uploaded")
    tmp_db.insert_paper("p3", "C", 2024, "placeholder")
    tmp_db.insert_concept("p1", "Method", "X", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Problem", "Y", 0.8, year=2024)
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_argument("p1", "claim", "supports", "Y", "Method")
    tmp_db.insert_queue_item("p1", "concept", '{"label": "Z"}', 0.6)
    tmp_db.commit()

    stats = tmp_db.get_stats()
    assert stats["papers"] == 3
    assert stats["uploaded"] == 1
    assert stats["placeholders"] == 1
    assert stats["concepts"] == 2
    assert stats["edges"] == 1
    assert stats["arguments"] == 1
    assert stats["queue_pending"] == 1


def test_get_stats_with_paper_ids_filter(tmp_db):
    """get_stats filters counts when paper_ids is provided."""
    tmp_db.insert_paper("p1", "A", 2024, "uploaded")
    tmp_db.insert_paper("p2", "B", 2024, "placeholder")
    tmp_db.insert_paper("p3", "C", 2024, "extracted")
    tmp_db.insert_concept("p1", "Method", "X", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Problem", "Y", 0.8, year=2024)
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_argument("p1", "claim", "supports", "Y", "Method")
    tmp_db.commit()

    stats = tmp_db.get_stats(paper_ids=["p1"])
    assert stats["papers"] == 1
    assert stats["uploaded"] == 1
    assert stats["placeholders"] == 0
    assert stats["concepts"] == 1
    assert stats["edges"] == 1
    assert stats["arguments"] == 1

    stats_all = tmp_db.get_stats(paper_ids=["p1", "p2"])
    assert stats_all["papers"] == 2
    assert stats_all["concepts"] == 2


def test_paper_categories_roundtrip(tmp_db):
    """v18: papers.categories + helpers (review §6.2)."""
    tmp_db.insert_paper("pA", "Kagome", 2024, "uploaded", categories="cond-mat.str-el quant-ph")
    tmp_db.insert_paper("pB", "Corrosion", 2020, "uploaded")
    tmp_db.set_paper_categories("pB", "physics.chem-ph")

    pairs = dict(tmp_db.iter_paper_categories())
    assert pairs == {"pA": "cond-mat.str-el quant-ph", "pB": "physics.chem-ph"}
    row = tmp_db.execute("SELECT categories FROM papers WHERE local_id = 'pA'").fetchone()
    assert row == ("cond-mat.str-el quant-ph",)


def test_paper_citations_insert_and_resolve(tmp_db):
    """v18: citation edges resolve against the corpus after full ingest."""
    tmp_db.insert_paper("p1", "Citing", 2024, "uploaded")
    tmp_db.insert_paper("p2", "Cited", 2023, "uploaded")

    tmp_db.insert_paper_cite_keys("p1", ["hep-lat/9107001", "kane2011"])
    tmp_db.insert_paper_cite_keys("p1", ["hep-lat/9107001"])  # PK dedupes
    tmp_db.insert_paper_cite_keys("p1", [])
    tmp_db.commit()

    unresolved = tmp_db.execute(
        "SELECT cited_key, cited_local_id FROM paper_cite_keys ORDER BY cited_key"
    ).fetchall()
    assert unresolved == [("hep-lat/9107001", None), ("kane2011", None)]

    resolved = tmp_db.resolve_paper_cite_keys({"hep-lat/9107001": "p2", "unknown": "p1"})
    assert resolved == 1
    now = tmp_db.execute(
        "SELECT cited_key, cited_local_id FROM paper_cite_keys WHERE cited_key = 'hep-lat/9107001'"
    ).fetchone()
    assert now == ("hep-lat/9107001", "p2")
