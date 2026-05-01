"""Tests for citation graph queries."""

import sqlite3

from drbrain.storage.citation_graph import (
    find_shared_refs,
    get_citation_counts,
    query_citation_graph,
)


def test_find_shared_refs_basic():
    """Two papers sharing references but not citing each other."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            local_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT DEFAULT '',
            year INTEGER, paper_type TEXT DEFAULT 'paper',
            status TEXT DEFAULT 'uploaded', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS edges (
            src_id TEXT NOT NULL, dst_id TEXT NOT NULL, relation TEXT NOT NULL,
            source_paper TEXT NOT NULL, weight REAL DEFAULT 1.0,
            PRIMARY KEY (src_id, dst_id, relation, source_paper)
        );
        CREATE TABLE IF NOT EXISTS citation_cache (
            source_paper TEXT NOT NULL, target_title TEXT NOT NULL,
            target_year INTEGER, relation TEXT NOT NULL,
            target_doi TEXT, target_s2_id TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_paper, target_title)
        );
    """)
    conn.execute("INSERT INTO papers (local_id, title) VALUES ('pA', 'Paper A')")
    conn.execute("INSERT INTO papers (local_id, title) VALUES ('pB', 'Paper B')")
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('pA', 'Shared Ref 1', 2023, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('pA', 'Shared Ref 2', 2022, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('pB', 'Shared Ref 1', 2023, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('pB', 'Shared Ref 2', 2022, 'references')"
    )

    result = find_shared_refs("pA", conn)

    assert len(result) >= 1
    assert result[0]["shared_with"] == "pB"
    assert result[0]["shared_count"] == 2
    assert result[0]["status"] == "unlinked"


def test_find_shared_refs_empty():
    """Paper with no shared references returns empty list."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            local_id TEXT PRIMARY KEY, title TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges (
            src_id TEXT NOT NULL, dst_id TEXT NOT NULL, relation TEXT NOT NULL,
            source_paper TEXT NOT NULL, weight REAL DEFAULT 1.0,
            PRIMARY KEY (src_id, dst_id, relation, source_paper)
        );
        CREATE TABLE IF NOT EXISTS citation_cache (
            source_paper TEXT NOT NULL, target_title TEXT NOT NULL,
            target_year INTEGER, relation TEXT NOT NULL,
            PRIMARY KEY (source_paper, target_title)
        );
    """)
    result = find_shared_refs("px", conn)
    assert result == []


def test_get_citation_counts():
    """get_citation_counts returns correct ref/citing/shared counts."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS citation_cache ("
        "source_paper TEXT, target_title TEXT, target_year INTEGER,"
        "relation TEXT CHECK(relation IN ('references','citing')),"
        "PRIMARY KEY (source_paper, target_title))"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('p1', 'R1', 2023, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('p1', 'R2', 2022, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('p1', 'C1', 2024, 'citing')"
    )

    counts = get_citation_counts("p1", conn)
    assert counts["references"] == 2
    assert counts["citing"] == 1


def test_query_citation_graph():
    """query_citation_graph returns full citation data."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            local_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT DEFAULT '',
            year INTEGER, paper_type TEXT DEFAULT 'paper',
            status TEXT DEFAULT 'uploaded', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS citation_cache (
            source_paper TEXT NOT NULL, target_title TEXT NOT NULL,
            target_year INTEGER, relation TEXT NOT NULL,
            target_doi TEXT, target_s2_id TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_paper, target_title)
        );
    """)
    conn.execute("INSERT INTO papers (local_id, title, year) VALUES ('p1', 'Test Paper', 2024)")
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('p1', 'Ref A', 2023, 'references')"
    )
    conn.execute(
        "INSERT INTO citation_cache (source_paper, target_title, target_year, relation) "
        "VALUES ('p1', 'Cite B', 2025, 'citing')"
    )

    result = query_citation_graph("p1", conn)
    assert result["paper"]["title"] == "Test Paper"
    assert len(result["refs"]) == 1
    assert result["refs"][0]["title"] == "Ref A"
    assert len(result["citing"]) == 1
    assert result["citing"][0]["title"] == "Cite B"
