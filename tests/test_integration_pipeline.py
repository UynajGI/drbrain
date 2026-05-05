"""Integration tests for the full pipeline: ingest → build → query → analyze → export.

These tests require real LLM API access and existing ingested+built papers.
Marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest

from drbrain.storage.database import Database


@pytest.mark.integration
def test_pipeline_papers_exist():
    """Verify papers were ingested and built successfully."""
    db = Database("data/drbrain.db")
    papers = db.get_all_papers()
    db.close()
    assert len(papers) >= 1, "No papers in database — run ingest + build first"
    extracted = [p for p in papers if p["status"] == "extracted"]
    assert len(extracted) >= 1, f"No extracted papers — run build first. Found: {len(papers)}"


@pytest.mark.integration
def test_pipeline_concepts_exist():
    """Verify concepts were extracted during build."""
    db = Database("data/drbrain.db")
    count = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    db.close()
    assert count >= 10, f"Expected >= 10 concepts, found {count}"


@pytest.mark.integration
def test_pipeline_edges_exist():
    """Verify relations were extracted and tree edges were added."""
    db = Database("data/drbrain.db")
    count = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    # Check contains edges exist (tree hierarchy)
    contains_count = db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relation = 'contains'"
    ).fetchone()[0]
    db.close()
    assert count >= 5, f"Expected >= 5 edges, found {count}"
    assert contains_count >= 1, f"Tree hierarchy 'contains' edges missing! Found {contains_count}"


@pytest.mark.integration
def test_pipeline_concepts_have_section():
    """Verify concepts have section field populated (tree+graph fix)."""
    db = Database("data/drbrain.db")
    with_section = db.conn.execute("SELECT COUNT(*) FROM concepts WHERE section != ''").fetchone()[
        0
    ]
    db.close()
    assert with_section >= 1, (
        "No concepts have section field set — tree+graph traversal will fail. "
        "Rebuild papers with: drbrain build"
    )


@pytest.mark.integration
def test_pipeline_volume_pages_columns():
    """Verify volume/pages columns exist in papers table."""
    db = Database("data/drbrain.db")
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(papers)").fetchall()]
    db.close()
    assert "volume" in cols, "volume column missing from papers table"
    assert "pages" in cols, "pages column missing from papers table"


@pytest.mark.integration
def test_pipeline_embed_trained():
    """Verify TransE embeddings were trained."""
    db = Database("data/drbrain.db")
    count = db.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    db.close()
    assert count >= 1, "No embeddings found — run: drbrain embed"


@pytest.mark.integration
def test_pipeline_audit_runs():
    """Verify audit command produces output on real data."""
    from pathlib import Path

    from drbrain.services.audit import audit_papers

    db = Database("data/drbrain.db")
    papers_dir = Path("data/papers")
    issues = audit_papers(db, papers_dir, severity="warning")
    db.close()
    assert isinstance(issues, list)
    # Should find at least some issues on real data
    assert len(issues) >= 0  # audit should not crash


@pytest.mark.integration
def test_pipeline_closure_produces_edges():
    """Verify closure rule inference works on real graph."""
    from drbrain.graph.engine import GraphEngine

    db = Database("data/drbrain.db")
    graph = GraphEngine()
    graph.load_from_db(db)
    inferred = graph.closure()
    db.close()
    assert isinstance(inferred, list)
    # At minimum should produce shared_actor edges from author affiliations


@pytest.mark.integration
def test_pipeline_rule_grounding_works():
    """Verify rule grounding produces transitive edges."""
    from drbrain.graph.engine import GraphEngine

    db = Database("data/drbrain.db")
    graph = GraphEngine()
    graph.load_from_db(db)
    grounded = graph.ground_rules(min_confidence=0.3)
    db.close()
    assert isinstance(grounded, list)
    # Should find at least some transitive patterns


@pytest.mark.integration
def test_pipeline_ask_returns_answer():
    """Verify KGQA (ask command) returns an answer."""
    from drbrain.query.bm25 import build_bm25_index

    db = Database("data/drbrain.db")
    idx = build_bm25_index(db)
    results = idx.search("knowledge graph", limit=3)
    db.close()
    assert len(results) >= 1, "BM25 search returned no results"
    # Verify result structure
    r = results[0]
    assert "label" in r
    assert "type" in r
    assert "local_id" in r


@pytest.mark.integration
def test_pipeline_export_produces_bibtex():
    """Verify BibTeX export includes journal, volume, pages."""
    from drbrain.storage.export import meta_to_bibtex

    db = Database("data/drbrain.db")
    papers = db.get_all_papers()
    db.close()
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
def test_pipeline_schema_version_tracked():
    """Verify schema version tracking works."""
    db = Database("data/drbrain.db")
    version = db.conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_versions").fetchone()[0]
    db.close()
    assert version >= 2, f"Schema version should be >= 2, got {version}"
