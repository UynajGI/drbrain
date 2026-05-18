"""Tests for src/drbrain/services/citation_styles.py."""

from __future__ import annotations

import textwrap

import pytest

from drbrain.services.citation_styles import (
    BUILTIN_STYLES,
    format_refs,
    get_formatter,
    list_styles,
    show_style,
)

# ── Sample metadata ────────────────────────────────────────────────

SINGLE_AUTHOR = {
    "title": "Deep Learning",
    "authors": ["Goodfellow, Ian"],
    "year": "2016",
    "journal": "MIT Press",
    "doi": "",
}

TWO_AUTHORS = {
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
    "year": "2017",
    "journal": "NeurIPS",
    "volume": "30",
    "pages": "5998-6008",
    "doi": "",
}

THREE_AUTHORS = {
    "title": "Graph Neural Networks",
    "authors": ["Smith, John", "Jones, Bob", "Lee, Alice"],
    "year": "2020",
    "journal": "Nature",
    "volume": "580",
    "pages": "123-130",
    "doi": "10.1038/s41586-020-1234-5",
}

MANY_AUTHORS = {
    "title": "Large-Scale Study",
    "authors": [f"Author{i}, First{i}" for i in range(10)],
    "year": "2023",
    "journal": "Science",
    "volume": "381",
    "pages": "456-462",
    "doi": "10.1126/science.abc1234",
}

UNKNOWN_AUTHOR = {
    "title": "Anonymous Work",
    "authors": [],
    "year": "",
    "journal": "",
}


# ── Built-in styles ────────────────────────────────────────────────


class TestBuiltinStylesExist:
    def test_all_four_present(self):
        for name in ("apa", "vancouver", "chicago-author-date", "mla"):
            assert name in BUILTIN_STYLES

    def test_get_formatter_returns_callable(self):
        for name in BUILTIN_STYLES:
            fmt = get_formatter(name)
            assert callable(fmt)

    def test_get_formatter_produces_non_empty_string(self):
        for name in BUILTIN_STYLES:
            fmt = get_formatter(name)
            result = fmt(SINGLE_AUTHOR, None)
            assert isinstance(result, str)
            assert len(result) > 0


class TestApaStyle:
    def test_single_author(self):
        fmt = get_formatter("apa")
        r = fmt(SINGLE_AUTHOR, None)
        assert "Goodfellow, Ian" in r
        assert "(2016)" in r
        assert "Deep Learning" in r
        assert "*MIT Press*" in r

    def test_two_authors_ampersand(self):
        fmt = get_formatter("apa")
        r = fmt(TWO_AUTHORS, None)
        assert "Vaswani, Ashish, & Shazeer, Noam" in r

    def test_three_authors(self):
        fmt = get_formatter("apa")
        r = fmt(THREE_AUTHORS, None)
        assert "Smith, John, Jones, Bob, & Lee, Alice" in r

    def test_many_authors_et_al(self):
        fmt = get_formatter("apa")
        r = fmt(MANY_AUTHORS, None)
        assert "Author0, First0 et al." in r

    def test_unknown_author(self):
        fmt = get_formatter("apa")
        r = fmt(UNKNOWN_AUTHOR, None)
        assert "Unknown" in r
        assert "n.d." in r

    def test_numbered_list(self):
        fmt = get_formatter("apa")
        r = fmt(SINGLE_AUTHOR, 1)
        assert r.startswith("1. ")

    def test_bullet_list(self):
        fmt = get_formatter("apa")
        r = fmt(SINGLE_AUTHOR, None)
        assert r.startswith("- ")

    def test_with_doi(self):
        fmt = get_formatter("apa")
        r = fmt(THREE_AUTHORS, None)
        assert "https://doi.org/10.1038/s41586-020-1234-5" in r


class TestVancouverStyle:
    def test_numbered_output(self):
        fmt = get_formatter("vancouver")
        r = fmt(SINGLE_AUTHOR, 1)
        assert r.startswith("1. ")

    def test_initials_format(self):
        fmt = get_formatter("vancouver")
        r = fmt(SINGLE_AUTHOR, None)
        assert "Goodfellow I" in r

    def test_six_authors_full(self):
        meta = {
            "title": "Test",
            "authors": [f"Last{i}, First{i}" for i in range(6)],
            "year": "2020",
            "journal": "Test Journal",
            "doi": "",
        }
        fmt = get_formatter("vancouver")
        r = fmt(meta, None)
        assert "Last0 F" in r
        assert "Last5 F" in r
        assert "et al" not in r

    def test_many_authors_et_al(self):
        fmt = get_formatter("vancouver")
        r = fmt(MANY_AUTHORS, None)
        assert "et al" in r

    def test_doi_format(self):
        meta = {**SINGLE_AUTHOR, "doi": "10.1234/test.1"}
        fmt = get_formatter("vancouver")
        r = fmt(meta, None)
        assert "doi:10.1234/test.1" in r


class TestChicagoStyle:
    def test_full_name_reversal(self):
        fmt = get_formatter("chicago-author-date")
        r = fmt(SINGLE_AUTHOR, None)
        assert "Goodfellow, Ian" in r

    def test_title_quoted(self):
        fmt = get_formatter("chicago-author-date")
        r = fmt(SINGLE_AUTHOR, None)
        assert '"Deep Learning."' in r

    def test_many_authors(self):
        fmt = get_formatter("chicago-author-date")
        r = fmt(MANY_AUTHORS, None)
        assert "Author0, First0 et al." in r


class TestMlaStyle:
    def test_name_reversal(self):
        fmt = get_formatter("mla")
        r = fmt(SINGLE_AUTHOR, None)
        assert "Goodfellow, Ian" in r

    def test_title_quoted(self):
        fmt = get_formatter("mla")
        r = fmt(SINGLE_AUTHOR, None)
        assert '"Deep Learning."' in r

    def test_volume_and_number(self):
        fmt = get_formatter("mla")
        r = fmt(TWO_AUTHORS, None)
        assert "vol. 30" in r
        assert "pp. 5998-6008" in r


# ── format_refs helper ─────────────────────────────────────────────


class TestFormatRefs:
    def test_apa_bullet_list(self):
        metas = [SINGLE_AUTHOR, TWO_AUTHORS]
        result = format_refs(metas, style="apa")
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("- ")
        assert lines[1].startswith("- ")

    def test_vancouver_numbered(self):
        metas = [SINGLE_AUTHOR, TWO_AUTHORS]
        result = format_refs(metas, style="vancouver")
        lines = result.split("\n")
        assert lines[0].startswith("1. ")
        assert lines[1].startswith("2. ")

    def test_default_style_is_apa(self):
        result = format_refs([SINGLE_AUTHOR])
        assert "- Goodfellow" in result


# ── Custom styles ──────────────────────────────────────────────────


class TestCustomStyles:
    def test_custom_style_loaded(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "nature.py").write_text(
            textwrap.dedent("""\
            def format_ref(meta, idx=None):
                prefix = f"{idx}. " if idx else "- "
                return prefix + (meta.get("title") or "Untitled").upper()
            """)
        )

        fmt = get_formatter("nature", styles_dir=custom_dir)
        r = fmt(SINGLE_AUTHOR, None)
        assert r == "- DEEP LEARNING"

    def test_custom_style_missing_format_ref(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "bad.py").write_text("# no format_ref defined\n")

        with pytest.raises(AttributeError, match="format_ref"):
            get_formatter("bad", styles_dir=custom_dir)

    def test_custom_style_invalid_name(self):
        with pytest.raises(ValueError, match="Invalid citation style name"):
            get_formatter("../../etc/passwd")

    def test_custom_style_not_found(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_formatter("nonexistent", styles_dir=custom_dir)

    def test_custom_style_syntax_error(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "broken.py").write_text(
            "def format_ref(meta idx):  # missing comma\n    pass\n"
        )

        with pytest.raises(ImportError, match="Failed to load"):
            get_formatter("broken", styles_dir=custom_dir)


# ── list_styles ────────────────────────────────────────────────────


class TestListStyles:
    def test_builtins_always_present(self):
        styles = list_styles()
        names = {s["name"] for s in styles}
        assert names >= {"apa", "vancouver", "chicago-author-date", "mla"}

    def test_builtins_have_source(self):
        styles = list_styles()
        for s in styles:
            if s["name"] in BUILTIN_STYLES:
                assert s["source"] == "built-in"

    def test_custom_styles_in_list(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "nature.py").write_text(
            "def format_ref(meta, idx=None):\n    return meta.get('title', '')"
        )
        (custom_dir / "nature.json").write_text(
            '{"description": "Nature journal style", "source": "custom"}'
        )

        styles = list_styles(custom_dir)
        names = {s["name"] for s in styles}
        assert "nature" in names

    def test_custom_does_not_shadow_builtin(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "apa.py").write_text("def format_ref(meta, idx=None):\n    return 'shadow'")
        styles = list_styles(custom_dir)
        apa_entries = [s for s in styles if s["name"] == "apa"]
        assert len(apa_entries) == 1
        assert apa_entries[0]["source"] == "built-in"


# ── show_style ─────────────────────────────────────────────────────


class TestShowStyle:
    def test_builtin_returns_comment(self):
        result = show_style("apa")
        assert "Built-in style: apa" in result
        assert "APA 7th" in result

    def test_custom_returns_source(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        (custom_dir / "nature.py").write_text(
            "def format_ref(meta, idx=None):\n    return meta['title']\n"
        )

        result = show_style("nature", styles_dir=custom_dir)
        assert "def format_ref" in result

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Invalid citation style name"):
            show_style("../../../etc/shadow")

    def test_not_found(self, tmp_path):
        custom_dir = tmp_path / "styles"
        custom_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            show_style("ghost", styles_dir=custom_dir)
