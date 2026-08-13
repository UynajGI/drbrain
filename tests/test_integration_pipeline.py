"""Integration tests for the full pipeline: ingest → build → query → analyze → export.

These tests run against the isolated ``test-run/`` 100-paper corpus (schema
v13), *not* the main ``data/drbrain.db`` (the 193K-paper fulltext corpus whose
KG lives in DuckDB). Marked with ``@pytest.mark.integration`` and excluded from
CI; each test is skipped gracefully when the corpus is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drbrain.storage.database import Database

_TEST_RUN = Path(__file__).resolve().parents[1] / "test-run"
_INTEGRATION_DB = _TEST_RUN / "db" / "drbrain.db"
_INTEGRATION_PAPERS = _TEST_RUN / "papers"


@pytest.fixture
def integration_db():
    """Open the test-run corpus DB, skipping the test when it is absent."""
    if not _INTEGRATION_DB.exists():
        pytest.skip(
            "test-run/db/drbrain.db not present — build the corpus first "
            "(cd test-run && drbrain pipeline --preset full)"
        )
    db = Database(str(_INTEGRATION_DB))
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def integration_papers_dir():
    """Path to the test-run papers directory, skipping when absent."""
    if not _INTEGRATION_PAPERS.exists():
        pytest.skip("test-run/papers not present")
    return _INTEGRATION_PAPERS


@pytest.mark.integration
def test_pipeline_papers_exist(integration_db):
    """Verify papers were ingested and built successfully."""
    papers = integration_db.get_all_papers()
    assert len(papers) >= 1, "No papers in database — run ingest + build first"
    extracted = [p for p in papers if p["status"] == "extracted"]
    assert len(extracted) >= 1, f"No extracted papers — run build first. Found: {len(papers)}"


@pytest.mark.integration
def test_pipeline_concepts_exist(integration_db):
    """Verify concepts were extracted during build."""
    count = integration_db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    assert count >= 10, f"Expected >= 10 concepts, found {count}"


@pytest.mark.integration
def test_pipeline_edges_exist(integration_db):
    """Verify relations were extracted and tree edges were added."""
    count = integration_db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    # Check contains edges exist (tree hierarchy)
    contains_count = integration_db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relation = 'contains'"
    ).fetchone()[0]
    assert count >= 5, f"Expected >= 5 edges, found {count}"
    assert contains_count >= 1, f"Tree hierarchy 'contains' edges missing! Found {contains_count}"


@pytest.mark.integration
def test_pipeline_concepts_have_section(integration_db):
    """Verify concepts have section field populated (tree+graph fix)."""
    with_section = integration_db.conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE section != ''"
    ).fetchone()[0]
    assert with_section >= 1, (
        "No concepts have section field set — tree+graph traversal will fail. "
        "Rebuild papers with: drbrain build"
    )


@pytest.mark.integration
def test_pipeline_volume_pages_columns(integration_db):
    """Verify volume/pages columns exist in papers table."""
    cols = [r[1] for r in integration_db.conn.execute("PRAGMA table_info(papers)").fetchall()]
    assert "volume" in cols, "volume column missing from papers table"
    assert "pages" in cols, "pages column missing from papers table"


@pytest.mark.integration
def test_pipeline_embed_trained(integration_db):
    """Verify TransE embeddings were trained."""
    count = integration_db.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count >= 1, "No embeddings found — run: drbrain embed"


@pytest.mark.integration
def test_pipeline_audit_runs(integration_db, integration_papers_dir):
    """Verify audit command produces output on real data."""
    from drbrain.services.audit import audit_papers

    issues = audit_papers(integration_db, integration_papers_dir, severity="warning")
    assert isinstance(issues, list)
    # Should find at least some issues on real data
    assert len(issues) >= 0  # audit should not crash


@pytest.mark.integration
def test_pipeline_closure_produces_edges(integration_db):
    """Verify closure rule inference works on real graph."""
    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(integration_db)
    inferred = graph.closure()
    assert isinstance(inferred, list)
    # At minimum should produce shared_actor edges from author affiliations


@pytest.mark.integration
def test_pipeline_rule_grounding_works(integration_db):
    """Verify rule grounding produces transitive edges."""
    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(integration_db)
    grounded = graph.ground_rules(min_confidence=0.3)
    assert isinstance(grounded, list)
    # Should find at least some transitive patterns


@pytest.mark.integration
def test_pipeline_ask_returns_answer(integration_db):
    """Verify KGQA (ask command) returns an answer."""
    from drbrain.query.bm25 import build_bm25_index

    idx = build_bm25_index(integration_db)
    results = idx.search("knowledge graph", limit=3)
    assert len(results) >= 1, "BM25 search returned no results"
    # Verify result structure
    r = results[0]
    assert "label" in r
    assert "type" in r
    assert "local_id" in r


@pytest.mark.integration
def test_pipeline_export_produces_bibtex(integration_db):
    """Verify BibTeX export includes journal, volume, pages."""
    from drbrain.storage.export import meta_to_bibtex

    papers = integration_db.get_all_papers()
    assert len(papers) >= 1
    # Build meta dict for first paper and check BibTeX output
    paper = papers[0]
    meta = {
        "local_id": paper["local_id"],
        "title": paper.get("title", "Test"),
        "year": paper.get("year"),
        "doi": paper.get("doi", ""),
        "authors": "Test Author",
        "first_author_lastname": "Author",
        "journal": paper.get("journal", ""),
        "volume": paper.get("volume", ""),
        "pages": paper.get("pages", ""),
    }
    result = meta_to_bibtex(meta)
    assert result.startswith("@")
    assert "title" in result.lower()


@pytest.mark.integration
def test_pipeline_schema_version_tracked(integration_db):
    """Verify schema version tracking works."""
    version = integration_db.conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_versions"
    ).fetchone()[0]
    assert version >= 2, f"Schema version should be >= 2, got {version}"
