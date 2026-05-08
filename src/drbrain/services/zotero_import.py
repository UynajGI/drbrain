"""Import papers from Zotero SQLite databases, BibTeX files, Zotero Web API, and Endnote XML/RIS."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_BIBTEX_ENTRY = re.compile(
    r"""@(\w+)\s*\{\s*([^,}]+)\s*,\s*((?:[^@]|(?:(?!^@\w+\{).))*)\}""",
    re.DOTALL | re.MULTILINE,
)
_BIB_FIELD = re.compile(r"""(\w+)\s*=\s*[{"]([^}"]+)[}"]""", re.DOTALL)

_DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)

# RIS field tag -> dict key mapping
_RIS_TAG_MAP: dict[str, str] = {
    "TY": "ris_type",
    "TI": "title",
    "T1": "title",
    "AU": "authors",
    "A1": "authors",
    "PY": "date",
    "Y1": "date",
    "DO": "doi",
    "JO": "journal",
    "JF": "journal",
    "JA": "journal",
    "T2": "journal",
    "VL": "volume",
    "SP": "start_page",
    "EP": "end_page",
    "PB": "publisher",
    "SN": "isbn",
    "UR": "url",
    "L1": "file_attachments",
    "L2": "url",
    "AB": "abstract",
    "N2": "abstract",
    "IS": "issue",
    "ER": None,  # end-of-record marker
}

# RIS type -> internal paper_type
_RIS_TYPE_MAP: dict[str, str] = {
    "JOUR": "paper",
    "CONF": "paper",
    "CPAPER": "paper",
    "THES": "thesis",
    "BOOK": "book",
    "CHAP": "book",
    "RPRT": "document",
    "GEN": "document",
}

# SI / supplement PDF filtering pattern
_SI_PATTERN = re.compile(
    r"(?:^|[-_ ])(?:SI|[Ss]uppl(?:ement(?:ary)?)?|[Ss]upporting)"
    r"|[-_ ](?:S\d+|Table\s*S\d+|Figure\s*S\d+)\.pdf$",
)


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


def _clean_doi(raw: str) -> str:
    """Strip URL prefix from DOI, return bare DOI."""
    if not raw:
        return ""
    return _DOI_PREFIX_RE.sub("", raw).strip()


def _parse_year(date_str: str) -> int | None:
    """Extract 4-digit year from a date string."""
    if not date_str:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", date_str)
    return int(m.group(1)) if m else None


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """Check whether a table exists in the SQLite database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a column exists in a table."""
    try:
        conn.execute(f"SELECT {column} FROM {table} LIMIT 0")
        return True
    except sqlite3.OperationalError:
        return False


# ============================================================================
#  T1: Zotero local SQLite -- collection filter + creator + PDF
# ============================================================================


def import_zotero_db(
    conn: sqlite3.Connection,
    *,
    collection_key: str | None = None,
    storage_dir: Path | None = None,
) -> list[dict]:
    """Import papers from a Zotero SQLite connection.

    Auto-detects normalized vs simplified Zotero schema. Supports:
    - Normalized: items.itemTypeID -> itemTypes, itemData.valueID -> itemDataValues
    - Simplified: items.itemType TEXT, itemData.value TEXT

    Args:
        conn: SQLite3 connection to a zotero.sqlite database.
        collection_key: Optional collection key to filter by.
        storage_dir: Optional Zotero storage directory for PDF detection.
                     Defaults to ``conn``'s directory's ``storage/`` subdir.

    Returns:
        List of paper dicts with keys: title, year, doi, authors, paper_type,
        journal, volume, pages, url, pdf_path.
    """
    # Detect schema variant:
    # Normalized: items.itemTypeID INTEGER -> itemTypes table
    # Simplified: items.itemType TEXT (no foreign key)
    has_item_type_id = _has_column(conn, "items", "itemTypeID")
    has_deleted = _has_table(conn, "deletedItems")
    has_creator_types = _has_table(conn, "creatorTypes")
    has_order_index = _has_column(conn, "itemCreators", "orderIndex")

    # Build the base items query based on schema
    if has_item_type_id:
        query = (
            "SELECT i.itemID, i.key, it.typeName "
            "FROM items i "
            "JOIN itemTypes it ON i.itemTypeID = it.itemTypeID "
            "WHERE it.typeName NOT IN ('attachment', 'note')"
        )
        if has_deleted:
            query += " AND i.itemID NOT IN (SELECT itemID FROM deletedItems)"
    else:
        query = "SELECT itemID, key, itemType FROM items"
        if has_deleted:
            query += " WHERE itemID NOT IN (SELECT itemID FROM deletedItems)"

    params: list = []

    # Filter by collection if requested
    item_id_filter: set[int] | None = None
    if collection_key:
        try:
            collector_rows = conn.execute(
                "SELECT ci.itemID FROM collectionItems ci "
                "JOIN collections c ON ci.collectionID = c.collectionID "
                "WHERE c.key = ?",
                (collection_key,),
            ).fetchall()
            item_id_filter = {r[0] for r in collector_rows}
        except sqlite3.OperationalError:
            pass

    items_rows = conn.execute(query, params).fetchall()

    papers = []
    for item_id, item_key, item_type in items_rows:
        if item_id_filter is not None and item_id not in item_id_filter:
            continue

        # Skip attachment/note when using simplified schema (no WHERE filter)
        if not has_item_type_id and item_type in ("attachment", "note"):
            continue

        # Read field values
        fields: dict[str, str] = {}
        try:
            for row in conn.execute(
                "SELECT f.fieldName, d.value FROM itemData d "
                "JOIN fields f ON f.fieldID = d.fieldID WHERE d.itemID = ?",
                (item_id,),
            ):
                fields[row[0]] = row[1]
        except sqlite3.OperationalError:
            pass

        # Parse creators: try normalized schema first, fall back to simplified
        authors: list[str] = []
        try:
            if has_creator_types:
                order_clause = "ORDER BY ic.orderIndex" if has_order_index else ""
                creator_rows = conn.execute(
                    f"SELECT c.firstName, c.lastName, ct.creatorType "
                    f"FROM itemCreators ic "
                    f"JOIN creators c ON ic.creatorID = c.creatorID "
                    f"JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID "
                    f"WHERE ic.itemID = ? {order_clause}",
                    (item_id,),
                ).fetchall()
            else:
                order_clause = "ORDER BY ic.orderIndex" if has_order_index else ""
                creator_rows = conn.execute(
                    f"SELECT c.firstName, c.lastName, ic.creatorType "
                    f"FROM itemCreators ic "
                    f"JOIN creators c ON ic.creatorID = c.creatorID "
                    f"WHERE ic.itemID = ? {order_clause}",
                    (item_id,),
                ).fetchall()

            for first, last, ctype in creator_rows:
                if ctype == "author":
                    first = (first or "").strip()
                    last = (last or "").strip()
                    if first and last:
                        authors.append(f"{first} {last}")
                    elif last:
                        authors.append(last)
                    elif first:
                        authors.append(first)
        except sqlite3.OperationalError:
            # Fall back to old approach: creatorID-based lookup via creator_map
            try:
                creator_map: dict[int, str] = {}
                for row in conn.execute("SELECT creatorID, lastName, firstName FROM creators"):
                    creator_map[row[0]] = f"{row[2]} {row[1]}".strip()
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
        year = _parse_year(year_str)

        # Find PDF attachment
        pdf_path: str = ""
        if storage_dir:
            try:
                resolved = _find_local_pdf(conn, item_id, storage_dir)
                if resolved:
                    pdf_path = str(resolved)
            except sqlite3.OperationalError:
                pass

        papers.append(
            {
                "title": title,
                "year": year,
                "doi": _clean_doi(fields.get("DOI", "")),
                "authors": " and ".join(authors),
                "paper_type": _map_zotero_type(item_type),
                "journal": fields.get("publicationTitle", ""),
                "volume": fields.get("volume", ""),
                "pages": fields.get("pages", ""),
                "url": fields.get("url", ""),
                "pdf_path": pdf_path,
            }
        )
    return papers


def _find_local_pdf(conn: sqlite3.Connection, parent_id: int, storage_dir: Path) -> Path | None:
    """Find PDF attachment for a given item in local Zotero storage.

    Queries ``itemAttachments`` for PDFs with the given parent item ID,
    resolves ``storage:<filename>`` paths via ``storage_dir / item_key / filename``.
    """
    try:
        rows = conn.execute(
            "SELECT ia.path, i.key "
            "FROM itemAttachments ia "
            "JOIN items i ON ia.itemID = i.itemID "
            "WHERE ia.parentItemID = ? "
            "AND ia.contentType = 'application/pdf'",
            (parent_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None

    for raw_path, att_key in rows:
        raw_path = (raw_path or "").strip()
        att_key = (att_key or "").strip()
        if raw_path.startswith("storage:") and att_key:
            filename = raw_path[len("storage:") :]
            pdf_path = storage_dir / att_key / filename
            if pdf_path.exists():
                return pdf_path
        elif raw_path:
            p = Path(raw_path)
            if p.exists():
                return p
    return None


def list_collections_local(db_path: Path) -> list[dict]:
    """List all collections in a local Zotero SQLite database.

    Args:
        db_path: Path to ``zotero.sqlite``.

    Returns:
        List of dicts with keys: ``key``, ``name``, ``numItems``.
    """
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT c.key, c.collectionName, COUNT(ci.itemID) as numItems "
            "FROM collections c "
            "LEFT JOIN collectionItems ci ON c.collectionID = ci.collectionID "
            "GROUP BY c.collectionID "
            "ORDER BY c.collectionName",
        ).fetchall()
        return [
            {"key": r["key"], "name": r["collectionName"], "numItems": r["numItems"]} for r in rows
        ]
    finally:
        conn.close()


# ============================================================================
#  T2: Zotero Web API
# ============================================================================


def fetch_zotero_api(
    library_id: str,
    api_key: str,
    *,
    library_type: str = "user",
    collection_key: str | None = None,
) -> list[dict]:
    """Fetch paper metadata from Zotero Web API.

    Args:
        library_id: Zotero library ID (user ID or group ID).
        api_key: Zotero API key.
        library_type: ``"user"`` or ``"group"``.
        collection_key: Optional collection key to filter by.

    Returns:
        List of paper dicts (same format as ``import_zotero_db`` output,
        minus ``pdf_path``).
    """
    try:
        from pyzotero import zotero as pyzotero  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyzotero is required for Zotero Web API access. Install with: pip install pyzotero"
        ) from None

    zot = pyzotero.Zotero(library_id, library_type, api_key)

    if collection_key:
        items = zot.everything(zot.collection_items(collection_key))
    else:
        items = zot.everything(zot.items())

    # Filter out attachments and notes
    items = [
        it
        for it in items
        if it.get("data", {}).get("itemType") not in ("attachment", "note", "linkAttachment")
    ]

    papers: list[dict] = []
    for item in items:
        data = item.get("data", {})

        # Parse creators
        creators = data.get("creators", [])
        authors: list[str] = []
        for c in creators:
            if c.get("creatorType", "author") != "author":
                continue
            if "name" in c:
                authors.append(c["name"])
            else:
                first = (c.get("firstName") or "").strip()
                last = (c.get("lastName") or "").strip()
                if first and last:
                    authors.append(f"{first} {last}")
                elif last:
                    authors.append(last)
                elif first:
                    authors.append(first)

        item_type = data.get("itemType", "")
        title = data.get("title", "Untitled")
        year = _parse_year(data.get("date", ""))

        papers.append(
            {
                "title": title,
                "year": year,
                "doi": _clean_doi(data.get("DOI", "")),
                "authors": " and ".join(authors),
                "paper_type": _map_zotero_type(item_type),
                "journal": (
                    data.get("publicationTitle")
                    or data.get("proceedingsTitle")
                    or data.get("bookTitle")
                    or ""
                ),
                "volume": data.get("volume", "") or "",
                "pages": data.get("pages", "") or "",
                "url": data.get("url", "") or "",
                "publisher": data.get("publisher", "") or "",
                "abstract": data.get("abstractNote", "") or "",
            }
        )

    return papers


def list_collections_api(
    library_id: str,
    api_key: str,
    *,
    library_type: str = "user",
) -> list[dict]:
    """List all collections from Zotero Web API.

    Args:
        library_id: Zotero library ID.
        api_key: Zotero API key.
        library_type: ``"user"`` or ``"group"``.

    Returns:
        List of dicts with keys: ``key``, ``name``, ``numItems``.
    """
    try:
        from pyzotero import zotero as pyzotero  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyzotero is required for Zotero Web API access. Install with: pip install pyzotero"
        ) from None

    zot = pyzotero.Zotero(library_id, library_type, api_key)
    collections = zot.collections()
    return [
        {
            "key": c["data"]["key"],
            "name": c["data"]["name"],
            "numItems": c["meta"].get("numItems", 0),
        }
        for c in collections
    ]


# ============================================================================
#  T3: Endnote XML / RIS
# ============================================================================


def parse_endnote_xml(path: Path) -> list[dict]:
    """Parse an Endnote XML export file.

    Requires the optional ``endnote_utils`` package.
    Falls back gracefully if not installed.

    Args:
        path: Path to the Endnote XML file.

    Returns:
        List of paper dicts.
    """
    try:
        import importlib

        core = importlib.import_module("endnote_utils.core")
    except ImportError:
        raise ImportError(
            "endnote_utils is required for Endnote XML parsing. "
            "Install with: pip install endnote-utils"
        ) from None

    papers: list[dict] = []
    for elem in core.iter_records_xml(path):
        rec = core.process_record_xml(elem, "endnote")
        papers.append(_endnote_record_to_dict(rec))
    return papers


def _endnote_record_to_dict(record: dict) -> dict:
    """Convert an endnote-utils record dict to DrBrain paper dict."""
    raw_authors = record.get("authors", "")
    authors_list = [a.strip() for a in raw_authors.split("; ") if a.strip()] if raw_authors else []
    # Normalize "Last, First" -> "First Last"
    norm_authors: list[str] = []
    for a in authors_list:
        if "," in a:
            parts = a.split(",", 1)
            norm_authors.append(f"{parts[1].strip()} {parts[0].strip()}")
        else:
            norm_authors.append(a)

    year = None
    year_str = record.get("year", "")
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            pass

    ref_type = record.get("ref_type", "")
    type_map = {
        "Journal Article": "paper",
        "Conference Paper": "paper",
        "Conference Proceedings": "paper",
        "Book": "book",
        "Book Section": "book",
        "Thesis": "thesis",
        "Report": "document",
        "Generic": "document",
    }

    return {
        "title": record.get("title", "Untitled"),
        "year": year,
        "doi": _clean_doi(record.get("doi", "")),
        "authors": " and ".join(norm_authors),
        "paper_type": type_map.get(ref_type, "paper"),
        "journal": record.get("journal", ""),
        "volume": record.get("volume", ""),
        "pages": record.get("pages", ""),
        "publisher": record.get("publisher", ""),
        "abstract": record.get("abstract", ""),
        "url": record.get("url", ""),
    }


def parse_endnote_ris(path: Path) -> list[dict]:
    """Parse an Endnote/Reference Manager RIS file.

    Regex-based parser -- no external dependencies.
    Handles multi-line values and repeated tags (e.g., AU).

    Args:
        path: Path to the RIS file.

    Returns:
        List of paper dicts.
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    # Split records by ER tag (at start of line)
    records_text = re.split(r"^ER\s", content, flags=re.MULTILINE)

    papers: list[dict] = []
    for rec_text in records_text:
        entries = _parse_ris_record(rec_text)
        if not entries or "title" not in entries:
            continue

        paper = _ris_entries_to_dict(entries)
        papers.append(paper)
    return papers


def _parse_ris_record(record_text: str) -> dict[str, str]:
    """Parse a single RIS record into a flat dict.

    Multi-value fields (AU, L1) are joined with delimiter.
    """
    # Match "TAG  - value" lines (two spaces before dash)
    tag_re = re.compile(r"^([A-Z][A-Z0-9])\s{2}-\s+(.*)$", re.MULTILINE)

    fields: dict[str, list[str]] = {}
    for m in tag_re.finditer(record_text):
        tag = m.group(1)
        value = m.group(2).strip()
        if tag == "ER":
            continue
        mapped = _RIS_TAG_MAP.get(tag)
        if mapped is None:
            continue
        fields.setdefault(mapped, []).append(value)

    # Flatten single-value fields; join multi-value fields
    result: dict[str, str] = {}
    multi_value_keys = {"authors", "file_attachments"}
    for key, values in fields.items():
        if key in multi_value_keys:
            result[key] = " ; ".join(values)
        else:
            result[key] = values[0]  # last value wins for single-value fields
    return result


def _ris_entries_to_dict(entries: dict[str, str]) -> dict:
    """Convert parsed RIS entries to DrBrain paper dict."""
    # Normalize authors: RIS uses "Last, First" or "First Last"
    raw_authors = entries.get("authors", "")
    authors_list = [a.strip() for a in raw_authors.split(" ; ") if a.strip()]
    norm_authors: list[str] = []
    for a in authors_list:
        if "," in a:
            parts = a.split(",", 1)
            norm_authors.append(f"{parts[1].strip()} {parts[0].strip()}")
        else:
            norm_authors.append(a)

    year = None
    year_str = entries.get("date", "")
    if year_str:
        try:
            year = int(year_str[:4])
        except ValueError:
            pass

    # Assemble pages from start/end
    pages = ""
    sp = entries.get("start_page", "")
    ep = entries.get("end_page", "")
    if sp and ep:
        pages = f"{sp}-{ep}"
    elif sp:
        pages = sp

    ris_type = entries.get("ris_type", "GEN")
    paper_type = _RIS_TYPE_MAP.get(ris_type, "paper")

    return {
        "title": entries.get("title", "Untitled"),
        "year": year,
        "doi": _clean_doi(entries.get("doi", "")),
        "authors": " and ".join(norm_authors),
        "paper_type": paper_type,
        "journal": entries.get("journal", ""),
        "volume": entries.get("volume", ""),
        "pages": pages,
        "publisher": entries.get("publisher", ""),
        "abstract": entries.get("abstract", ""),
        "url": entries.get("url", ""),
    }


# ============================================================================
#  BibTeX import
# ============================================================================


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
