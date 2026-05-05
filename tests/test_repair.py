"""Tests for metadata repair."""

from unittest.mock import patch

from drbrain.services.repair import (
    REPAIR_SOURCES,
    normalize_title,
    repair_paper,
)


def test_normalize_title_all_caps():
    """All-caps titles are normalized to title case."""
    result = normalize_title("DEEP LEARNING FOR GRAPHS")
    assert result != "DEEP LEARNING FOR GRAPHS"
    assert "Deep" in result


def test_normalize_title_already_ok():
    """Well-formatted titles pass through unchanged."""
    assert normalize_title("Deep Learning for Graphs") == "Deep Learning for Graphs"


def test_normalize_title_strips_arxiv_id():
    """arXiv ID embedded in title is removed."""
    result = normalize_title("arxiv:2401.00001 A Novel Approach to GNNs")
    assert "arxiv:2401" not in result.lower()
    assert "A Novel Approach" in result


def test_repair_sources_enum():
    """REPAIR_SOURCES lists expected fields per source."""
    assert "doi" in REPAIR_SOURCES
    assert "arxiv" in REPAIR_SOURCES
    assert "title_year" in REPAIR_SOURCES


class FakeDB:
    """Fake database for testing repair_paper.

    Accepts an optional paper dict to customize get_paper() return value.
    Pass paper=None (or omit) to simulate paper-not-found.
    """

    def __init__(self, paper=None):
        self._committed = False
        self._executed = []
        self._paper = paper
        self.conn = self._FakeConn(self)

    class _FakeConn:
        def __init__(self, parent):
            self._parent = parent

        def execute(self, sql, params=()):
            self._parent._executed.append((sql, params))
            return self._FakeCursor()

        class _FakeCursor:
            def fetchone(self):
                return None

            def fetchall(self):
                return []

    def get_paper(self, lid):
        if self._paper is None:
            return None
        paper = dict(self._paper)
        paper.setdefault("local_id", lid)
        paper.setdefault("title", "TEST PAPER")
        paper.setdefault("year", 2024)
        paper.setdefault("doi", None)
        paper.setdefault("arxiv", None)
        return paper

    def commit(self):
        self._committed = True


def _make_paper(**overrides):
    """Helper to build a paper dict with sensible defaults."""
    defaults = {"title": "TEST PAPER", "year": 2024, "doi": None, "arxiv": None}
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Existing test
# ---------------------------------------------------------------------------


def test_repair_paper_dry_run():
    """dry_run does not modify DB but detects title issue."""
    db = FakeDB(paper=_make_paper())
    repairs = repair_paper(db, "p1", dry_run=True)
    assert isinstance(repairs, list)
    title_repairs = [r for r in repairs if r["field"] == "title"]
    assert len(title_repairs) >= 1
    assert not db._committed


# ---------------------------------------------------------------------------
# Tests: _repair_via_crossref (via repair_paper)
# ---------------------------------------------------------------------------


def test_repair_via_crossref_returns_repairs():
    """When CrossRef returns data with differing fields, repairs are emitted."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", title="OLD TITLE"))

    crossref_data = {
        "title": ["Better CrossRef Title"],
        "created": {"date-parts": [[2023, 5, 15]]},
        "author": [
            {"given": "Alice", "family": "Smith"},
            {"given": "Bob", "family": "Jones"},
        ],
        "container-title": ["Journal of Testing"],
    }

    with patch(
        "drbrain.extractor.crossref.fetch_work_by_doi",
        return_value=crossref_data,
        create=True,
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    crossref_repairs = [r for r in repairs if r["source"] == "CrossRef"]
    fields = {r["field"] for r in crossref_repairs}
    assert "title" in fields
    assert "year" in fields
    assert "authors" in fields
    assert "journal" in fields

    # Verify specific values
    title_r = next(r for r in crossref_repairs if r["field"] == "title")
    assert title_r["new"] == "Better CrossRef Title"

    year_r = next(r for r in crossref_repairs if r["field"] == "year")
    assert year_r["new"] == 2023


def test_repair_via_crossref_empty_data_no_repairs():
    """When CrossRef returns empty data, no CrossRef repairs are emitted."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi"))

    with patch(
        "drbrain.extractor.crossref.fetch_work_by_doi",
        return_value={},
        create=True,
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    crossref_repairs = [r for r in repairs if r["source"] == "CrossRef"]
    assert len(crossref_repairs) == 0


def test_repair_via_crossref_none_data_no_repairs():
    """When CrossRef returns None, no CrossRef repairs are emitted."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi"))

    with patch(
        "drbrain.extractor.crossref.fetch_work_by_doi",
        return_value=None,
        create=True,
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    crossref_repairs = [r for r in repairs if r["source"] == "CrossRef"]
    assert len(crossref_repairs) == 0


# ---------------------------------------------------------------------------
# Tests: _repair_via_arxiv (via repair_paper)
# ---------------------------------------------------------------------------


def test_repair_via_arxiv_returns_repairs():
    """When arXiv returns a different title/year, repairs are emitted."""
    db = FakeDB(paper=_make_paper(arxiv="2401.00001", title="OLD TITLE", year=2023))

    with patch(
        "drbrain.parser.mineru_parser._fetch_arxiv_metadata",
        return_value=("ArXiv Provided Title", 2024),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    arxiv_repairs = [r for r in repairs if r["source"] == "arXiv"]
    fields = {r["field"] for r in arxiv_repairs}
    assert "title" in fields
    assert "year" in fields

    title_r = next(r for r in arxiv_repairs if r["field"] == "title")
    assert title_r["new"] == "ArXiv Provided Title"

    year_r = next(r for r in arxiv_repairs if r["field"] == "year")
    assert year_r["new"] == 2024


def test_repair_via_arxiv_same_title_no_title_repair():
    """When arXiv title matches normalized title, only year repair emits."""
    db = FakeDB(paper=_make_paper(arxiv="2401.00001", title="Test Paper", year=2023))

    with patch(
        "drbrain.parser.mineru_parser._fetch_arxiv_metadata",
        return_value=("Test Paper", 2024),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    arxiv_repairs = [r for r in repairs if r["source"] == "arXiv"]
    fields = {r["field"] for r in arxiv_repairs}
    # Title matches normalized → no title repair; year differs → year repair
    assert "title" not in fields
    assert "year" in fields


# ---------------------------------------------------------------------------
# Tests: _repair_via_title_year (via repair_paper)
# ---------------------------------------------------------------------------


def test_repair_via_title_year_returns_doi():
    """When fetch_doi_by_title returns a DOI, a doi repair is emitted."""
    db = FakeDB(paper=_make_paper(title="A Specific Research Title", doi=None))

    with patch(
        "drbrain.extractor.crossref.fetch_doi_by_title",
        return_value={"doi": "10.5678/founddoi", "title": "A Specific Research Title"},
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    doi_repairs = [r for r in repairs if r["field"] == "doi"]
    assert len(doi_repairs) == 1
    assert doi_repairs[0]["new"] == "10.5678/founddoi"
    assert doi_repairs[0]["source"] == "CrossRef"


def test_repair_via_title_year_no_doi_when_paper_has_doi():
    """When paper already has a DOI, title-year lookup is skipped."""
    db = FakeDB(paper=_make_paper(title="A Specific Research Title", doi="10.existing/doi"))

    with patch(
        "drbrain.extractor.crossref.fetch_doi_by_title",
    ) as mock_fetch:
        repair_paper(db, "p1", dry_run=True)

    # The mock should NOT have been called because paper already has DOI
    mock_fetch.assert_not_called()


def test_repair_via_title_year_empty_title_no_call():
    """When paper has no title, title-year lookup returns early."""
    db = FakeDB(paper=_make_paper(title=""))

    with patch(
        "drbrain.extractor.crossref.fetch_doi_by_title",
    ) as mock_fetch:
        repair_paper(db, "p1", dry_run=True)

    # Should not call fetch_doi_by_title when title is empty
    mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: repair_paper with dry_run=False
# ---------------------------------------------------------------------------


def test_repair_paper_dry_run_false_applies_updates():
    """With dry_run=False, DB updates are executed and commit is called."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", title="OLD TITLE", year=2022))

    crossref_data = {
        "title": ["Updated CrossRef Title"],
        "created": {"date-parts": [[2025]]},
        "author": [
            {"given": "Alice", "family": "Smith"},
        ],
        "container-title": ["Test Journal"],
    }

    with patch(
        "drbrain.extractor.crossref.fetch_work_by_doi",
        return_value=crossref_data,
        create=True,
    ):
        repairs = repair_paper(db, "p1", dry_run=False)

    # Verify repairs were still returned
    assert len(repairs) > 0

    # Verify commit was called
    assert db._committed

    # Verify SQL UPDATE statements were issued for title and year
    title_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE papers SET title" in sql
    ]
    year_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE papers SET year" in sql
    ]
    assert len(title_updates) >= 1  # normalization + CrossRef title
    assert len(year_updates) >= 1


def test_repair_paper_dry_run_false_applies_doi_update():
    """With dry_run=False, doi repair triggers paper_ids UPDATE."""
    db = FakeDB(paper=_make_paper(title="A Specific Research Title", doi=None))

    with patch(
        "drbrain.extractor.crossref.fetch_doi_by_title",
        return_value={"doi": "10.5678/founddoi2"},
    ):
        repair_paper(db, "p1", dry_run=False)

    assert db._committed
    doi_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE paper_ids SET doi" in sql
    ]
    assert len(doi_updates) == 1
    assert doi_updates[0][1] == ("10.5678/founddoi2", "p1")


# ---------------------------------------------------------------------------
# Tests: repair_paper edge cases
# ---------------------------------------------------------------------------


def test_repair_paper_none_paper_returns_error():
    """When get_paper returns None, an error dict is returned."""
    db = FakeDB(paper=None)  # paper=None signals not-found
    repairs = repair_paper(db, "nonexistent", dry_run=True)
    assert len(repairs) == 1
    assert repairs[0]["field"] == "error"
    assert "not found" in repairs[0]["reason"].lower()


def test_repair_paper_no_doi_no_arxiv_only_normalization():
    """Paper with no DOI and no arXiv: only title normalization, no API calls."""
    db = FakeDB(paper=_make_paper(title="ALL CAPS TITLE", doi=None, arxiv=None))

    with (
        patch(
            "drbrain.extractor.crossref.fetch_work_by_doi",
            create=True,
        ) as mock_crossref,
        patch("drbrain.parser.mineru_parser._fetch_arxiv_metadata") as mock_arxiv,
        patch(
            "drbrain.services.repair._enrich_via_openalex",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.crossref.fetch_doi_by_title",
            return_value=None,
        ),
    ):
        result = repair_paper(db, "p1", dry_run=True)

    # Only normalization repair, no API calls
    sources = {r["source"] for r in result}
    assert sources == {"normalization"}

    # _repair_via_crossref and _repair_via_arxiv return early
    # because doi/arxiv are None → no API calls
    mock_crossref.assert_not_called()
    mock_arxiv.assert_not_called()
    # _repair_via_title_year probes CrossRef when title exists;
    # our mock returns None so no repair is emitted (but the call is expected)


# ---------------------------------------------------------------------------
# Tests: exception handling in repair functions
# ---------------------------------------------------------------------------


def test_repair_via_crossref_handles_api_exception():
    """When fetch_work_by_doi raises, no CrossRef repairs are emitted."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi"))

    with patch(
        "drbrain.extractor.crossref.fetch_work_by_doi",
        side_effect=RuntimeError("API down"),
        create=True,
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    crossref_repairs = [r for r in repairs if r["source"] == "CrossRef"]
    assert len(crossref_repairs) == 0


def test_repair_via_arxiv_handles_api_exception():
    """When _fetch_arxiv_metadata raises, no arXiv repairs are emitted."""
    db = FakeDB(paper=_make_paper(arxiv="2401.00001"))

    with patch(
        "drbrain.parser.mineru_parser._fetch_arxiv_metadata",
        side_effect=RuntimeError("arXiv API down"),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    arxiv_repairs = [r for r in repairs if r["source"] == "arXiv"]
    assert len(arxiv_repairs) == 0


def test_repair_via_title_year_handles_api_exception():
    """When fetch_doi_by_title raises, no title-year repairs are emitted."""
    db = FakeDB(paper=_make_paper(title="Some Title", doi=None))

    with patch(
        "drbrain.extractor.crossref.fetch_doi_by_title",
        side_effect=RuntimeError("CrossRef API down"),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    doi_repairs = [r for r in repairs if r["field"] == "doi"]
    assert len(doi_repairs) == 0


def test_repair_paper_handles_repair_fn_exception():
    """When a repair function raises unexpectedly, others still run."""
    db = FakeDB(paper=_make_paper(arxiv="2401.00001"))

    # Mock _repair_via_crossref to raise (covering the except in
    # repair_paper's for-loop), while _repair_via_arxiv still succeeds.
    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            side_effect=RuntimeError("unexpected crash"),
        ),
        patch(
            "drbrain.parser.mineru_parser._fetch_arxiv_metadata",
            return_value=("ArXiv Title", 2025),
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    # _repair_via_arxiv should still have produced repairs
    arxiv_repairs = [r for r in repairs if r["source"] == "arXiv"]
    assert len(arxiv_repairs) >= 1


# ---------------------------------------------------------------------------
# Tests: _enrich_via_openalex (via repair_paper)
# ---------------------------------------------------------------------------


def test_enrich_via_openalex_doi_path_returns_abstract_and_citation():
    """DOI path: get_work_enriched returns abstract + cited_by_count, both applied."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", abstract="", citation_count=0))

    enriched = {
        "doi": "10.1234/testdoi",
        "title": "Test Paper",
        "year": 2024,
        "openalex_id": "https://openalex.org/W123",
        "abstract": "This is an abstract from OpenAlex.",
        "cited_by_count": 42,
        "journal": "Test Journal",
        "authors": "Alice Smith and Bob Jones",
        "volume": "15",
        "pages": "100-120",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    fields = {r["field"] for r in oa_repairs}
    assert "abstract" in fields
    assert "citation_count" in fields
    assert "journal" in fields
    assert "authors" in fields
    assert "volume" in fields
    assert "pages" in fields

    abstract_r = next(r for r in oa_repairs if r["field"] == "abstract")
    assert "abstract from OpenAlex" in abstract_r["new"]

    citation_r = next(r for r in oa_repairs if r["field"] == "citation_count")
    assert citation_r["new"] == 42


def test_enrich_via_openalex_title_path_returns_abstract_and_citation():
    """Title path (no DOI): search_work_by_title + get_work_enriched returns enriched data."""
    db = FakeDB(paper=_make_paper(doi=None, title="Some Title", abstract=""))

    enriched = {
        "doi": "10.5678/found",
        "title": "Some Title",
        "year": 2023,
        "openalex_id": "https://openalex.org/W456",
        "abstract": "Abstract via title search.",
        "cited_by_count": 17,
        "journal": "Another Journal",
        "authors": "Carol Davis",
        "volume": "",
        "pages": "",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.search_work_by_title",
            return_value={"doi": "10.5678/found", "title": "Some Title"},
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    fields = {r["field"] for r in oa_repairs}
    assert "abstract" in fields
    assert "citation_count" in fields
    assert "authors" in fields


def test_enrich_via_openalex_authors_only():
    """When enriched has no authors but search_authors_by_work returns them."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", authors=""))

    enriched = {
        "doi": "10.1234/testdoi",
        "title": "Test Paper",
        "year": 2024,
        "openalex_id": "https://openalex.org/W123",
        "abstract": "",
        "cited_by_count": 0,
        "journal": "",
        "authors": "",
        "volume": "",
        "pages": "",
    }

    authors_list = [
        {
            "author_id": "A1234567890",
            "display_name": "Eve Green",
            "orcid": None,
            "raw_affiliation": [],
        },
        {
            "author_id": "A0987654321",
            "display_name": "Frank Blue",
            "orcid": None,
            "raw_affiliation": [],
        },
    ]

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=authors_list,
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    fields = {r["field"] for r in oa_repairs}
    assert "authors" in fields

    authors_r = next(r for r in oa_repairs if r["field"] == "authors")
    assert "Eve Green" in authors_r["new"]
    assert "Frank Blue" in authors_r["new"]


def test_enrich_via_openalex_no_doi_no_title_returns_empty():
    """When paper has neither DOI nor title, enrich returns empty."""
    db = FakeDB(paper=_make_paper(title="", doi=None))

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    assert len(oa_repairs) == 0


def test_enrich_via_openalex_handles_api_exception():
    """When OpenAlex API raises, no OpenAlex repairs are emitted."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi"))

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            side_effect=RuntimeError("OpenAlex API down"),
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    assert len(oa_repairs) == 0


def test_repair_paper_applies_authors_to_db():
    """With dry_run=False, authors repair triggers papers UPDATE."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", authors=""))

    enriched = {
        "doi": "10.1234/testdoi",
        "title": "Test Paper",
        "year": 2024,
        "openalex_id": "https://openalex.org/W123",
        "abstract": "",
        "cited_by_count": 0,
        "journal": "",
        "authors": "Alice Smith",
        "volume": "",
        "pages": "",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repair_paper(db, "p1", dry_run=False)

    assert db._committed
    authors_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE papers SET authors" in sql
    ]
    assert len(authors_updates) == 1
    assert authors_updates[0][1] == ("Alice Smith", "p1")


def test_repair_paper_applies_volume_pages_to_db():
    """With dry_run=False, volume and pages repairs trigger papers UPDATE."""
    db = FakeDB(paper=_make_paper(doi="10.1234/testdoi", volume="", pages=""))

    enriched = {
        "doi": "10.1234/testdoi",
        "title": "Test Paper",
        "year": 2024,
        "openalex_id": "https://openalex.org/W123",
        "abstract": "",
        "cited_by_count": 0,
        "journal": "",
        "authors": "",
        "volume": "42",
        "pages": "200-250",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repair_paper(db, "p1", dry_run=False)

    assert db._committed
    volume_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE papers SET volume" in sql
    ]
    pages_updates = [
        (sql, params) for sql, params in db._executed if "UPDATE papers SET pages" in sql
    ]
    assert len(volume_updates) == 1
    assert len(pages_updates) == 1
    assert volume_updates[0][1] == ("42", "p1")
    assert pages_updates[0][1] == ("200-250", "p1")


def test_enrich_via_openalex_paper_already_has_data_skips():
    """When paper already has abstract/cited_by_count/authors, no OpenAlex repair emitted."""
    db = FakeDB(
        paper=_make_paper(
            doi="10.1234/testdoi",
            abstract="Existing abstract",
            citation_count=10,
            authors="Existing Author",
            volume="1",
            pages="1-10",
            journal="Existing Journal",
        )
    )

    enriched = {
        "doi": "10.1234/testdoi",
        "title": "Test Paper",
        "year": 2024,
        "openalex_id": "https://openalex.org/W123",
        "abstract": "New abstract",
        "cited_by_count": 42,
        "journal": "New Journal",
        "authors": "New Author",
        "volume": "2",
        "pages": "11-20",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            return_value=enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    assert len(oa_repairs) == 0


def test_enrich_via_openalex_doi_path_fallback_to_title():
    """When get_work_enriched returns None for DOI, falls back to title search."""
    db = FakeDB(paper=_make_paper(doi="10.broken/doi", title="Real Title", abstract=""))

    enriched = {
        "doi": "10.5678/title-found",
        "title": "Real Title",
        "year": 2023,
        "openalex_id": "https://openalex.org/W789",
        "abstract": "Abstract via title fallback.",
        "cited_by_count": 5,
        "journal": "",
        "authors": "",
        "volume": "",
        "pages": "",
    }

    with (
        patch(
            "drbrain.services.repair._repair_via_crossref",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_arxiv",
            return_value=[],
        ),
        patch(
            "drbrain.services.repair._repair_via_title_year",
            return_value=[],
        ),
        patch(
            "drbrain.extractor.openalex.get_work_enriched",
            side_effect=lambda doi: None if doi == "10.broken/doi" else enriched,
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_work_by_title",
            return_value={"doi": "10.5678/title-found"},
            create=True,
        ),
        patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            return_value=None,
            create=True,
        ),
    ):
        repairs = repair_paper(db, "p1", dry_run=True)

    oa_repairs = [r for r in repairs if r["source"] == "OpenAlex"]
    assert len(oa_repairs) > 0
    fields = {r["field"] for r in oa_repairs}
    assert "abstract" in fields
    assert "citation_count" in fields
