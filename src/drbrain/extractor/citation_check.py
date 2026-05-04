"""Extract citations from text and verify against local library."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Narrative: Author (Year), Author & Author (Year), Author et al. (Year)
_RE_NARRATIVE = re.compile(
    r"""
    (?P<author>
        [A-Z][a-zA-Z\-']+
        (?:\s+(?:and|&)\s+[A-Z][a-zA-Z\-']+)?
    )
    (?:\s+et\s+al\.)?
    \s*
    \((?P<year>\d{4})\)
    """,
    re.VERBOSE,
)

# Parenthetical: (...)
_RE_PARENTHETICAL = re.compile(
    r"""
    \(
    (?P<citations>[^)]*\d{4}[^)]*)
    \)
    """,
    re.VERBOSE,
)

# Single citation inside parenthetical group
_RE_PAREN_SINGLE = re.compile(
    r"""
    (?P<author>
        [A-Z][a-zA-Z\-']+
        (?:\s+(?:and|&)\s+[A-Z][a-zA-Z\-']+)?
    )
    (?:\s+et\s+al\.)?
    ,?\s*
    (?P<year>\d{4})
    """,
    re.VERBOSE,
)


@dataclass
class CitationMatch:
    author: str
    year: str
    raw: str
    found: bool = False
    matched_id: str | None = None
    matched_title: str | None = None


def extract_citations(text: str) -> list[CitationMatch]:
    """Extract author-year citations from text. Deduplicates by (author, year)."""
    seen: set[tuple[str, str]] = set()
    results: list[CitationMatch] = []

    # Narrative citations: Author (Year)
    for m in _RE_NARRATIVE.finditer(text):
        author = m.group("author").strip()
        year = m.group("year")
        key = (author.lower(), year)
        if key not in seen:
            seen.add(key)
            results.append(
                CitationMatch(
                    author=author,
                    year=year,
                    raw=m.group(0),
                )
            )

    # Parenthetical citations: (...)
    for pm in _RE_PARENTHETICAL.finditer(text):
        inner = pm.group("citations")
        for sm in _RE_PAREN_SINGLE.finditer(inner):
            author = sm.group("author").strip()
            year = sm.group("year")
            key = (author.lower(), year)
            if key not in seen:
                seen.add(key)
                results.append(
                    CitationMatch(
                        author=author,
                        year=year,
                        raw=f"({sm.group(0)})",
                    )
                )

    return results


def match_citations(
    citations: list[CitationMatch],
    db,
) -> list[CitationMatch]:
    """Match extracted citations against local paper library.

    Searches by author lastname in aliases + year in papers.
    """
    if db is None:
        return citations

    for c in citations:
        rows = db.conn.execute(
            "SELECT a.canonical_id, p.local_id, p.title, p.year "
            "FROM aliases a "
            "JOIN concepts c2 ON c2.label = a.canonical_id "
            "JOIN papers p ON p.local_id = c2.local_id "
            "WHERE a.variant LIKE ? AND p.year = ? "
            "LIMIT 1",
            (f"%{c.author}%", int(c.year)),
        ).fetchall()

        if rows:
            c.found = True
            c.matched_id = rows[0][1]
            c.matched_title = rows[0][2]

    return citations
