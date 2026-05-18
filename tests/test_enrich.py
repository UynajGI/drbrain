"""Tests for metadata enrichment service.

TDD: tests written before implementation.
"""

from __future__ import annotations


class TestMetadataCheck:
    """Test metadata completeness checking."""

    def test_check_complete(self):
        from drbrain.services.enrich import check_metadata_completeness

        paper = {
            "title": "Test Paper",
            "year": 2023,
            "authors": "Smith, John",
            "journal": "Nature",
            "doi": "10.1234/test",
        }
        missing = check_metadata_completeness(paper)
        assert missing == []

    def test_check_missing_fields(self):
        from drbrain.services.enrich import check_metadata_completeness

        paper = {
            "title": "Test",
            "year": None,
            "authors": "",
            "journal": "",
        }
        missing = check_metadata_completeness(paper)
        assert "year" in missing
        assert "authors" in missing
        assert "journal" in missing

    def test_check_partial(self):
        from drbrain.services.enrich import check_metadata_completeness

        paper = {"title": "Test", "year": 2023}
        missing = check_metadata_completeness(paper)
        assert "authors" in missing
        assert "year" not in missing

    def test_scrub_suspect_detection(self):
        from drbrain.services.enrich import detect_scrub_suspects

        # Paper missing title - highly suspect
        paper = {"title": "", "authors": "X", "year": 2020}
        issues = detect_scrub_suspects(paper)
        assert any("title" in i.lower() for i in issues)

    def test_scrub_suspect_clean_paper(self):
        from drbrain.services.enrich import detect_scrub_suspects

        paper = {
            "title": "A Real Title Of A Paper",
            "authors": "Smith, John and Jones, Bob",
            "year": 2023,
            "journal": "Nature",
            "doi": "10.1234/real",
        }
        issues = detect_scrub_suspects(paper)
        assert len(issues) == 0


class TestCrossRefBackfill:
    """Test CrossRef metadata backfill (unit tests, no API calls)."""

    def test_build_crossref_url(self):
        from drbrain.services.enrich import _build_crossref_url

        url = _build_crossref_url("10.1234/test")
        assert "10.1234/test" in url
        assert "api.crossref.org" in url

    def test_parse_crossref_response(self):
        from drbrain.services.enrich import _parse_crossref_response

        response = {
            "message": {
                "title": ["Test Paper"],
                "author": [
                    {"given": "John", "family": "Smith"},
                    {"given": "Bob", "family": "Jones"},
                ],
                "published-print": {"date-parts": [[2023]]},
                "container-title": ["Nature"],
                "volume": "580",
                "page": "123-130",
            }
        }
        meta = _parse_crossref_response(response)
        assert meta["title"] == "Test Paper"
        assert meta["authors"] == "Smith, John and Jones, Bob"
        assert meta["year"] == 2023
        assert meta["journal"] == "Nature"
        assert meta["volume"] == "580"


class TestMergeEnrichment:
    """Test merging enriched metadata into paper dict."""

    def test_merge_fills_missing(self):
        from drbrain.services.enrich import merge_enrichment

        paper = {"title": "Old", "year": None, "authors": ""}
        enriched = {"title": "New Title", "year": 2023, "authors": "Smith, J"}
        result = merge_enrichment(paper, enriched)
        assert result["year"] == 2023
        assert result["authors"] == "Smith, J"
        # Title should not be overwritten if already present
        assert result["title"] == "Old"

    def test_merge_does_not_overwrite(self):
        from drbrain.services.enrich import merge_enrichment

        paper = {"title": "Original", "year": 2020, "authors": "Jones, B"}
        enriched = {"title": "Different", "year": 2023}
        result = merge_enrichment(paper, enriched)
        assert result["title"] == "Original"
        assert result["year"] == 2020
