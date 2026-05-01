"""Tests for metadata repair."""

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
    def __init__(self):
        self._committed = False
        self._executed = []
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
        return {"local_id": lid, "title": "TEST PAPER", "year": 2024, "doi": None, "arxiv": None}

    def commit(self):
        self._committed = True


def test_repair_paper_dry_run():
    """dry_run does not modify DB but detects title issue."""
    db = FakeDB()
    repairs = repair_paper(db, "p1", dry_run=True)
    assert isinstance(repairs, list)
    title_repairs = [r for r in repairs if r["field"] == "title"]
    assert len(title_repairs) >= 1
    assert not db._committed
