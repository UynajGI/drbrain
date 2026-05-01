"""Import papers from Zotero SQLite databases and BibTeX files."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_BIBTEX_ENTRY = re.compile(
    r"""@(\w+)\s*\{\s*([^,}]+)\s*,\s*((?:[^@]|(?:(?!^@\w+\{).))*)\}""",
    re.DOTALL | re.MULTILINE,
)
_BIB_FIELD = re.compile(r"""(\w+)\s*=\s*[{"]([^}"]+)[}"]""", re.DOTALL)


def _map_zotero_type(item_type: str) -> str:
    mapping = {
        "journalArticle": "paper",
        "conferencePaper": "paper",
        "preprint": "preprint",
        "thesis": "thesis",
        "book": "book",
        "bookSection": "book",
        "report": "document",
        "document": "document",
    }
    return mapping.get(item_type, "paper")


def import_zotero_db(conn: sqlite3.Connection) -> list[dict]:
    """Import papers from a Zotero SQLite connection."""
    field_map: dict[str, int] = {}
    for row in conn.execute("SELECT fieldID, fieldName FROM fields"):
        field_map[row[1]] = row[0]

    creator_map: dict[int, str] = {}
    try:
        for row in conn.execute("SELECT creatorID, lastName, firstName FROM creators"):
            creator_map[row[0]] = f"{row[2]} {row[1]}".strip()
    except sqlite3.OperationalError:
        pass

    items = conn.execute("SELECT itemID, itemType FROM items").fetchall()
    papers = []
    for item_id, item_type in items:
        fields = {}
        for row in conn.execute(
            "SELECT f.fieldName, d.value FROM itemData d "
            "JOIN fields f ON f.fieldID = d.fieldID WHERE d.itemID = ?",
            (item_id,),
        ):
            fields[row[0]] = row[1]

        authors = []
        try:
            for row in conn.execute(
                "SELECT c.creatorID FROM itemCreators c "
                "WHERE c.itemID = ? AND c.creatorType = 'author'",
                (item_id,),
            ):
                if row[0] in creator_map:
                    authors.append(creator_map[row[0]])
        except sqlite3.OperationalError:
            pass

        title = fields.get("title", "Untitled")
        year_str = fields.get("date", "")
        year = None
        if year_str:
            try:
                year = int(year_str[:4])
            except ValueError:
                pass

        papers.append(
            {
                "title": title,
                "year": year,
                "doi": fields.get("DOI", ""),
                "authors": " and ".join(authors),
                "paper_type": _map_zotero_type(item_type),
                "journal": fields.get("publicationTitle", ""),
                "volume": fields.get("volume", ""),
                "pages": fields.get("pages", ""),
                "url": fields.get("url", ""),
            }
        )
    return papers


def import_bibtex_file(path: Path) -> list[dict]:
    """Import papers from a BibTeX .bib file."""
    content = path.read_text(encoding="utf-8", errors="replace")
    papers = []

    for entry_match in _BIBTEX_ENTRY.finditer(content):
        entry_type = entry_match.group(1)
        body = entry_match.group(3)

        fields = {}
        for fm in _BIB_FIELD.finditer(body):
            fields[fm.group(1).lower()] = fm.group(2)

        title = fields.get("title", "Untitled")
        authors = fields.get("author", "")
        norm_authors = []
        for a in authors.split(" and "):
            a = a.strip()
            if "," in a:
                parts = a.split(",", 1)
                a = f"{parts[1].strip()} {parts[0].strip()}"
            norm_authors.append(a)

        year = None
        if fields.get("year"):
            try:
                year = int(fields["year"])
            except ValueError:
                pass

        type_map = {
            "article": "paper",
            "inproceedings": "paper",
            "phdthesis": "thesis",
            "mastersthesis": "thesis",
            "book": "book",
            "inbook": "book",
            "misc": "document",
            "techreport": "document",
        }

        papers.append(
            {
                "title": title,
                "year": year,
                "doi": fields.get("doi", ""),
                "authors": " and ".join(norm_authors),
                "paper_type": type_map.get(entry_type, "paper"),
                "journal": fields.get("journal", ""),
                "volume": fields.get("volume", ""),
                "pages": fields.get("pages", ""),
            }
        )
    return papers
