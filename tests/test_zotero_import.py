"""Tests for Zotero import."""

import sqlite3
import tempfile
from pathlib import Path

from drbrain.services.zotero_import import (
    import_bibtex_file,
    import_zotero_db,
)


def test_import_zotero_db_minimal():
    """Import from a minimal Zotero SQLite DB."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemType TEXT, key TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, value TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, lastName TEXT, firstName TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, creatorType TEXT);
        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT);
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
        INSERT INTO fields VALUES (1, 'title'), (2, 'date'), (3, 'DOI'),
               (4, 'url'), (5, 'publicationTitle'), (6, 'volume'), (7, 'pages');
        INSERT INTO items VALUES (1, 'journalArticle', 'ABC123');
        INSERT INTO itemData VALUES (1, 1, 'Test Paper'), (1, 2, '2024'), (1, 3, '10.1234/test');
        INSERT INTO creators VALUES (1, 'Smith', 'John');
        INSERT INTO itemCreators VALUES (1, 1, 'author');
        INSERT INTO collections VALUES (1, 'My Papers');
        INSERT INTO collectionItems VALUES (1, 1);
    """)

    papers = import_zotero_db(conn)
    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper"
    assert papers[0]["year"] == 2024
    assert papers[0]["doi"] == "10.1234/test"
    assert "John Smith" in papers[0]["authors"]
    assert papers[0]["paper_type"] == "paper"


def test_import_zotero_db_empty():
    """Empty Zotero DB returns empty list."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemType TEXT, key TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, value TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
    """)
    assert import_zotero_db(conn) == []


def test_import_bibtex_file():
    """Import from a BibTeX file."""
    bib = """
@article{smith2024deep,
  title = {Deep Learning for Graphs},
  author = {Smith, John and Jones, Alice},
  year = {2024},
  journal = {Journal of AI Research},
  doi = {10.1234/test}
}
"""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False)
    f.write(bib)
    path = f.name
    f.close()

    try:
        papers = import_bibtex_file(Path(path))
        assert len(papers) == 1
        assert papers[0]["title"] == "Deep Learning for Graphs"
        assert "Smith" in papers[0]["authors"]
    finally:
        Path(path).unlink()


def test_import_bibtex_minimal():
    """Minimal BibTeX entry works."""
    bib = "@misc{test2025, title = {Just a Test}, year = {2025}}"
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False)
    f.write(bib)
    path = f.name
    f.close()

    try:
        papers = import_bibtex_file(Path(path))
        assert len(papers) == 1
        assert papers[0]["title"] == "Just a Test"
    finally:
        Path(path).unlink()
