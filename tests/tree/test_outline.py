"""T20/T21: transient structure hints — real locators, no model, no network.

The vendored Markdown entry point and the classic PDF entry point are the
boundaries under test; the PDF building itself is a boundary test of the
vendored page extraction.  Every test runs with all outbound sockets denied
(where it matters) so a hint can never depend on a model call.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import fitz
import pytest

from drbrain.tree.outline import (
    TEXT_HINT_LIMIT,
    StructureHint,
    extract_md_outline,
    extract_pdf_outline,
    structure_coverage,
)
from drbrain.tree.upstream import load_pageindex_module

MD_DOC = (
    "# Doc Title\n"
    "\n"
    "Abstract paragraph.\n"
    "\n"
    "## Methods\n"
    "\n"
    "Body of the methods section.\n"
    "\n"
    "```python\n"
    "# not a heading\n"
    "## also not a heading\n"
    "```\n"
    "\n"
    "~~~\n"
    "### still not a heading\n"
    "~~~\n"
    "\n"
    "### Deep Analysis\n"
    "\n"
    "Deep body.\n"
    "\n"
    "#### Level Four\n"
    "\n"
    "##### Level Five\n"
    "\n"
    "###### Level Six\n"
    "\n"
    "Tail paragraph.\n"
)

TEX_DOC = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "\\section{Introduction}\n"
    "Text with an inline formula $x=1$.\n"
    "\\begin{verbatim}\n"
    "\\section{Not A Section}\n"
    "\\end{verbatim}\n"
    "\\subsection{Prior Work}\n"
    "More prose.\n"
    "\\section{Methods}\n"
    "Method body.\n"
    "\\end{document}\n"
)

PDF_PAGES = (
    "Introduction\n\nThis paper studies structure hints.\nMore text on page one.",
    "Methods\n\nWe reuse the vendored two-page extraction.",
    "Results\n\nNumbers and a conclusion.",
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Deny every outbound connection (and DNS) while recording the attempt."""
    attempts: list[tuple] = []

    def deny(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    return attempts


@pytest.fixture
def pdf_document(tmp_path: Path) -> tuple[Path, int]:
    """A tiny real PDF: three pages, each with a distinct leading heading."""
    document = fitz.open()
    for body in PDF_PAGES:
        page = document.new_page()
        page.insert_text((72, 72), body)
    path = tmp_path / "tiny.pdf"
    document.save(path)
    document.close()
    return path, len(PDF_PAGES)


def _hint(**overrides) -> StructureHint:
    payload = {
        "kind": "page",
        "title": "Page",
        "heading_path": (),
        "anchor": "",
        "level": 0,
        "page_start": 1,
        "page_end": 1,
    }
    payload.update(overrides)
    return StructureHint(**payload)


def _section(**overrides) -> dict:
    payload = {
        "kind": "section",
        "title": "Intro",
        "heading_path": ("Intro",),
        "anchor": "Intro",
        "level": 1,
        "line_start": 1,
        "line_end": 4,
    }
    payload.update(overrides)
    return payload


class TestMarkdownOutline:
    def test_heading_levels_paths_and_line_ranges(self):
        hints = extract_md_outline(MD_DOC)
        assert [hint.title for hint in hints] == [
            "Doc Title",
            "Methods",
            "Deep Analysis",
            "Level Four",
            "Level Five",
            "Level Six",
        ]
        assert [hint.level for hint in hints] == [1, 2, 3, 4, 5, 6]
        assert hints[-1].heading_path == (
            "Doc Title",
            "Methods",
            "Deep Analysis",
            "Level Four",
            "Level Five",
            "Level Six",
        )
        assert all(hint.origin == "pageindex-md" for hint in hints)
        assert all(hint.kind == "section" for hint in hints)
        assert all(hint.anchor == hint.title for hint in hints)
        # Text material: lines only, never fabricated PDF pages.
        assert all(hint.page_start is None and hint.page_end is None for hint in hints)
        # 1-based inclusive, contiguous, and ending at the last real line.
        assert hints[0].line_start == 1
        for previous, current in zip(hints, hints[1:]):
            assert previous.line_end + 1 == current.line_start
        assert hints[-1].line_end == MD_DOC.count("\n") + 1

    def test_code_fences_never_become_headings(self):
        hints = extract_md_outline(MD_DOC)
        titles = [hint.title for hint in hints]
        assert "not a heading" not in titles
        assert "also not a heading" not in titles
        assert "still not a heading" not in titles
        assert all("not a heading" not in " > ".join(hint.heading_path) for hint in hints)

    def test_headingless_text_has_no_invented_sections(self):
        assert extract_md_outline("Only prose here.\n\nAnd more prose.\n") == []
        fenced_only = "```\n# looks like a heading\n```\n"
        assert extract_md_outline(fenced_only) == []
        assert extract_md_outline("") == []
        assert extract_md_outline("   \n\n") == []

    def test_max_nodes_truncates_in_document_order(self):
        assert [hint.title for hint in extract_md_outline(MD_DOC, max_nodes=2)] == [
            "Doc Title",
            "Methods",
        ]
        with pytest.raises(ValueError):
            extract_md_outline(MD_DOC, max_nodes=0)

    def test_repeated_calls_are_identical(self):
        first = extract_md_outline(MD_DOC)
        second = extract_md_outline(MD_DOC)
        assert first == second
        assert [(h.line_start, h.line_end, h.title) for h in first] == [
            (h.line_start, h.line_end, h.title) for h in second
        ]

    def test_text_hint_is_capped_and_budget_only(self):
        hints = extract_md_outline(MD_DOC)
        assert all(len(hint.text_hint) <= TEXT_HINT_LIMIT for hint in hints)
        assert "Abstract paragraph." in hints[0].text_hint

    def test_temporary_adapter_file_is_deleted(self, tmp_path, monkeypatch):
        """The upstream path-based entry gets a temp file with a real lifetime."""
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        hints = extract_md_outline(MD_DOC)
        assert hints  # the adapter ran and produced hints
        assert list(tmp_path.iterdir()) == []


class TestTexOutline:
    def test_tex_sectioning_is_detected_with_depth(self):
        hints = extract_md_outline(TEX_DOC)
        assert [hint.title for hint in hints] == ["Introduction", "Prior Work", "Methods"]
        assert [hint.level for hint in hints] == [1, 2, 1]
        assert hints[1].heading_path == ("Introduction", "Prior Work")
        assert all(hint.origin == "headings" for hint in hints)
        # 1-based inclusive line locators over the real TeX lines.
        assert hints[0].line_start == 3
        assert hints[0].line_end == hints[1].line_start - 1
        assert hints[-1].line_end == TEX_DOC.count("\n") + 1

    def test_tex_verbatim_sections_are_ignored(self):
        hints = extract_md_outline(TEX_DOC)
        assert all("Not A Section" not in hint.heading_path for hint in hints)

    def test_tex_without_sections_is_empty(self):
        assert (
            extract_md_outline("\\documentclass{article}\n\\begin{document}\n\\end{document}\n")
            == []
        )


class TestHintContract:
    def test_locator_families_never_mix(self):
        with pytest.raises(ValueError, match="one locator family"):
            _hint(line_start=1, line_end=2)
        with pytest.raises(ValueError, match="one locator family"):
            StructureHint(**_section(page_start=1, page_end=2))

    def test_locators_are_required(self):
        with pytest.raises(ValueError, match="at least one real locator"):
            StructureHint(kind="page", title="", heading_path=(), anchor="", level=0)

    def test_both_ends_of_a_locator_pair_are_required(self):
        with pytest.raises(ValueError, match="both ends are required"):
            _hint(page_end=None)
        with pytest.raises(ValueError, match="both ends are required"):
            StructureHint(**{**_section(), "line_end": None})

    def test_ranges_are_1_based_and_ordered(self):
        with pytest.raises(ValueError, match="invalid page range"):
            _hint(page_start=2, page_end=1)
        with pytest.raises(ValueError, match="invalid page range"):
            _hint(page_start=0, page_end=1)
        with pytest.raises(ValueError, match="invalid line range"):
            StructureHint(**_section(line_start=5, line_end=4))

    def test_section_hints_must_be_self_consistent(self):
        with pytest.raises(ValueError, match="outline depth"):
            StructureHint(**_section(level=2))
        with pytest.raises(ValueError, match="ends with its own title"):
            StructureHint(**_section(heading_path=("Other",)))
        with pytest.raises(ValueError, match="anchor is the heading text"):
            StructureHint(**_section(anchor="other"))

    def test_page_hints_carry_no_heading_path(self):
        with pytest.raises(ValueError, match="no heading path"):
            _hint(heading_path=("Intro",))
        with pytest.raises(ValueError, match="level 0"):
            _hint(level=1)

    def test_unknown_kind_and_origin_are_rejected(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            _hint(kind="chapter")
        with pytest.raises(ValueError, match="origin must be one of"):
            _hint(origin="pageindex-flash")

    def test_hint_size_caps_are_enforced(self):
        with pytest.raises(ValueError, match="text_hint exceeds"):
            _hint(text_hint="x" * (TEXT_HINT_LIMIT + 1))
        with pytest.raises(ValueError, match="title exceeds"):
            _hint(title="t" * 200)


class TestPdfOutline:
    def test_page_cover_uses_real_pdf_pages(self, pdf_document, no_network):
        path, npages = pdf_document
        hints = extract_pdf_outline(path)
        assert [hint.kind for hint in hints] == ["page"] * npages
        assert [(hint.page_start, hint.page_end) for hint in hints] == [(1, 1), (2, 2), (3, 3)]
        assert all(hint.origin == "pageindex-classic" for hint in hints)
        assert all(hint.line_start is None and hint.line_end is None for hint in hints)
        assert all(1 <= hint.page_start <= hint.page_end <= npages for hint in hints)
        # The page's own leading text, never a synthesized title.
        assert [hint.title for hint in hints] == ["Introduction", "Methods", "Results"]
        assert no_network == []

    def test_injected_page_list_is_used_as_is(self, pdf_document, no_network, monkeypatch):
        path, npages = pdf_document
        utils_module = load_pageindex_module("utils")
        injected = utils_module.get_page_tokens(str(path))
        assert len(injected) == npages

        def forbidden(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("page extraction must be skipped when page_list is injected")

        monkeypatch.setattr(utils_module, "get_page_tokens", forbidden)
        hints = extract_pdf_outline(path, page_list=injected)
        assert [(hint.page_start, hint.page_end) for hint in hints] == [(1, 1), (2, 2), (3, 3)]
        assert hints == extract_pdf_outline(path, page_list=injected)
        assert no_network == []

    def test_injected_page_list_must_match_the_pdf(self, pdf_document):
        path, _npages = pdf_document
        with pytest.raises(ValueError, match="must describe the real PDF"):
            extract_pdf_outline(path, page_list=[("only", 1), ("two", 1)])

    @pytest.mark.parametrize(
        "page_list",
        [
            [],
            [("text only",)],
            [(1, 2)],
            [("text", "not a token count")],
            ["ab"],  # a two-character string is not a page pair
            [("text", True)],
        ],
    )
    def test_malformed_page_list_is_rejected(self, pdf_document, page_list):
        path, _npages = pdf_document
        with pytest.raises(ValueError):
            extract_pdf_outline(path, page_list=page_list)

    def test_repeated_calls_are_identical(self, pdf_document, no_network):
        path, _npages = pdf_document
        assert extract_pdf_outline(path) == extract_pdf_outline(path)

    def test_max_nodes_truncates(self, pdf_document, no_network):
        path, _npages = pdf_document
        assert len(extract_pdf_outline(path, max_nodes=2)) == 2

    def test_no_model_entry_point_without_an_index_model(self, pdf_document, monkeypatch):
        path, _npages = pdf_document
        classic = load_pageindex_module("page_index_classic")
        utils_module = load_pageindex_module("utils")

        def forbidden(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("no model may be called without an explicit index model")

        monkeypatch.setattr(classic, "page_index_main", forbidden)
        monkeypatch.setattr(utils_module, "llm_completion", forbidden)
        monkeypatch.setattr(utils_module, "llm_acompletion", forbidden)
        assert len(extract_pdf_outline(path)) == 3

    def test_classic_tree_is_reused_with_an_explicit_index_model(
        self, pdf_document, tmp_path, monkeypatch, no_network
    ):
        path, npages = pdf_document
        classic = load_pageindex_module("page_index_classic")
        utils_module = load_pageindex_module("utils")
        captured: dict = {}
        backend = {"api_base": "http://127.0.0.1:9/v1", "api_key": "local"}
        monkeypatch.chdir(tmp_path)

        def fake_main(doc, opt, logger, page_list=None):  # noqa: N803 - upstream shape
            captured["doc"] = doc
            captured["opt"] = opt
            captured["logger"] = logger
            captured["page_list"] = page_list
            captured["backend"] = utils_module._llm_backend.get()
            # Exercise the injected logger: upstream's JsonLogger would drop
            # ``logs/`` and a JSON file into the current directory here.
            logger.info({"total_page_number": len(page_list)})
            logger.error("pretend upstream failure")
            return {
                "doc_name": "tiny",
                "structure": [
                    {
                        "title": "Introduction",
                        "node_id": "0000",
                        "start_index": 1,
                        "end_index": 2,
                        "nodes": [
                            {
                                "title": "Setup",
                                "node_id": "0001",
                                "start_index": 1,
                                "end_index": 1,
                                "nodes": [],
                            }
                        ],
                    },
                    {"title": "Results", "node_id": "0002", "start_index": 3, "end_index": 3},
                ],
            }

        monkeypatch.setattr(classic, "page_index_main", fake_main)
        hints = extract_pdf_outline(path, index_model="local/test", backend=backend)

        # The real PDF is passed through: no temporary pseudo-PDF, no SDK store.
        assert Path(captured["doc"]) == path
        assert captured["opt"].if_add_node_summary == "no"
        assert captured["opt"].if_add_doc_description == "no"
        assert captured["opt"].if_add_node_text == "no"
        assert captured["opt"].model == "local/test"
        # Upstream's JsonLogger writes ./logs files; our adapter must be used.
        assert not isinstance(captured["logger"], utils_module.JsonLogger)
        # The injected page map is the PDF's own extracted pages, in order.
        assert captured["page_list"] == list(utils_module.get_page_tokens(str(path)))
        assert len(captured["page_list"]) == npages
        assert captured["backend"] == backend
        assert utils_module._llm_backend.get() is None  # reset after the call
        # Hints are transient: the call left no log file, tree or store behind.
        assert list(tmp_path.iterdir()) == [path]
        assert no_network == []

        assert [hint.title for hint in hints] == ["Introduction", "Setup", "Results"]
        assert [(hint.page_start, hint.page_end) for hint in hints] == [(1, 2), (1, 1), (3, 3)]
        assert hints[1].heading_path == ("Introduction", "Setup")
        assert all(hint.origin == "pageindex-classic" for hint in hints)
        assert all(hint.kind == "section" for hint in hints)
        assert all(hint.text_hint for hint in hints)

    def test_classic_hints_report_overlapping_parent_ranges(self, pdf_document, monkeypatch):
        path, _npages = pdf_document
        classic = load_pageindex_module("page_index_classic")

        def fake_main(doc, opt, logger, page_list=None):  # noqa: N803 - upstream shape
            return {
                "structure": [
                    {
                        "title": "Chunk",
                        "start_index": 1,
                        "end_index": 3,
                        "nodes": [{"title": "Part", "start_index": 1, "end_index": 2, "nodes": []}],
                    }
                ]
            }

        monkeypatch.setattr(classic, "page_index_main", fake_main)
        hints = extract_pdf_outline(path, index_model="local/test")
        coverage = structure_coverage(hints, total_pages=3)["pages"]
        assert coverage["covered"] == 3
        assert coverage["residual_ranges"] == []
        assert coverage["overlap_units"] == 2  # parent/child ranges may overlap

    def test_out_of_document_pages_are_dropped_not_clamped_into_hints(
        self, pdf_document, monkeypatch
    ):
        path, _npages = pdf_document
        classic = load_pageindex_module("page_index_classic")

        def fake_main(doc, opt, logger, page_list=None):  # noqa: N803 - upstream shape
            return {
                "structure": [
                    {"title": "Real", "start_index": 1, "end_index": 3, "nodes": []},
                    {"title": "Ghost", "start_index": 9, "end_index": 11, "nodes": []},
                    {"title": "Untitled", "start_index": None, "end_index": None, "nodes": []},
                ]
            }

        monkeypatch.setattr(classic, "page_index_main", fake_main)
        hints = extract_pdf_outline(path, index_model="local/test")
        assert [hint.title for hint in hints] == ["Real"]
        assert hints[0].page_start == 1 and hints[0].page_end == 3


class TestCoverage:
    def test_reports_residuals_when_totals_are_known(self, pdf_document, no_network):
        path, _npages = pdf_document
        hints = extract_pdf_outline(path)
        complete = structure_coverage(hints, total_pages=3)
        assert complete["family"] == "page"
        assert complete["hint_count"] == 3
        assert complete["pages"]["covered"] == 3
        assert complete["pages"]["uncovered"] == 0
        assert complete["pages"]["residual_ranges"] == []
        assert complete["pages"]["ratio"] == 1.0

        partial = structure_coverage(hints, total_pages=5)["pages"]
        assert partial["uncovered"] == 2
        assert partial["residual_ranges"] == [[4, 5]]
        assert partial["ratio"] == pytest.approx(0.6)

    def test_line_coverage_for_markdown_text(self):
        hints = extract_md_outline(MD_DOC)
        lines = MD_DOC.count("\n") + 1
        coverage = structure_coverage(hints, total_lines=lines)
        assert coverage["family"] == "line"
        assert coverage["lines"]["covered"] == lines
        assert coverage["lines"]["residual_ranges"] == []
        assert structure_coverage(hints)["lines"]["uncovered"] is None

    def test_unknown_totals_are_not_guessed(self):
        coverage = structure_coverage([_hint(page_start=2, page_end=4)])
        pages = coverage["pages"]
        assert pages["total"] is None
        assert pages["uncovered"] is None
        assert pages["ratio"] is None
        assert pages["residual_ranges"] == []
        assert pages["covered"] == 3

    def test_gaps_and_overlaps_are_counted(self):
        hints = [
            _hint(page_start=1, page_end=3),
            _hint(page_start=2, page_end=4),
            _hint(page_start=6, page_end=6),
        ]
        pages = structure_coverage(hints, total_pages=8)["pages"]
        assert pages["covered"] == 5  # 1..4 plus 6
        assert pages["overlap_units"] == 2
        assert pages["residual_ranges"] == [[5, 5], [7, 8]]

    def test_out_of_range_claims_are_flagged(self):
        hints = [_hint(page_start=1, page_end=6)]
        pages = structure_coverage(hints, total_pages=3)["pages"]
        assert pages["out_of_range_units"] == 3
        assert pages["uncovered"] == 0

    def test_empty_hint_list_is_an_honest_full_residual(self):
        coverage = structure_coverage([], total_pages=2, total_lines=4)
        assert coverage["family"] == "none"
        assert coverage["hint_count"] == 0
        assert coverage["pages"]["residual_ranges"] == [[1, 2]]
        assert coverage["lines"]["residual_ranges"] == [[1, 4]]

    def test_mixed_families_are_reported_separately(self):
        hints = [_hint(page_start=1, page_end=1), StructureHint(**_section())]
        coverage = structure_coverage(hints, total_pages=2, total_lines=10)
        assert coverage["family"] == "mixed"
        assert coverage["pages"]["covered"] == 1
        assert coverage["lines"]["covered"] == 4
