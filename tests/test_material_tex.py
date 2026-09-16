"""Tests for the LaTeX material adapter (wrapped arXiv sources)."""

from __future__ import annotations

from pathlib import Path

from drbrain.parser.material import extract_material

BANNER = "=" * 48

WRAPPED = (
    "% DrBrain source_id: hep-lat/9107001\n"
    "% Title: How to Put a Heavier Higgs on the Lattice\n"
    "% DOI: 10.1016/0370-2693(92)90028-3\n"
    "% Categories: hep-lat\n"
    f"{BANNER}\n"
    "FILE: 9107001.tex\n"
    f"{BANNER}\n"
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "\\section{Introduction}\n"
    "We study the lattice \\cite{ref1}. See also arXiv:1412.6980 for methods.\n"
    "\\section{Results}\n"
    "The result holds.\n"
    "\\end{document}\n"
)


def test_wrapped_tex_header_metadata_and_markdown(tmp_path: Path):
    """Wrapper header fields feed title/DOI/arXiv/year; sections become headings."""
    source = tmp_path / "tex_00001_hep-lat_9107001.tex"
    source.write_text(WRAPPED, encoding="utf-8")
    parsed = extract_material(source, {})

    assert parsed.title == "How to Put a Heavier Higgs on the Lattice"
    assert parsed.doi == "10.1016/0370-2693(92)90028-3"
    assert parsed.arxiv == "hep-lat/9107001"
    assert parsed.year == 1991
    # Section commands become markdown headings so the tree builder sees them.
    assert "## Introduction" in parsed.raw_md
    assert "## Results" in parsed.raw_md
    # The wrapper header and banner never leak into raw.md.
    assert "DrBrain source_id" not in parsed.raw_md
    assert "FILE: 9107001.tex" not in parsed.raw_md
    # A reference-side arXiv mention is never adopted as the paper's own id.
    assert parsed.arxiv != "1412.6980"


def test_wrapped_tex_citations_become_atoms(tmp_path: Path):
    """``\\cite`` keys are extracted into ``[CITE:...]`` atoms in raw.md."""
    source = tmp_path / "tex_cite.tex"
    source.write_text(WRAPPED, encoding="utf-8")
    parsed = extract_material(source, {})
    assert "[CITE:ref1]" in parsed.raw_md


def test_tex_without_wrapper_still_converts(tmp_path: Path):
    """A bare .tex without wrapper metadata still converts and falls back to the stem."""
    source = tmp_path / "bare.tex"
    source.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Solo}\n"
        "Only text.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    parsed = extract_material(source, {})
    assert "## Solo" in parsed.raw_md
    assert parsed.arxiv is None
    assert parsed.title == "bare"


def test_new_style_source_id_keeps_versionless_id_and_year(tmp_path: Path):
    """New-style ids strip the version suffix and derive the submission year."""
    source = tmp_path / "tex_new.tex"
    source.write_text(
        "% DrBrain source_id: 2301.03216v2\n"
        "% Title: A New Study\n"
        f"{BANNER}\n"
        "FILE: 2301.03216.tex\n"
        f"{BANNER}\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "Hi.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    parsed = extract_material(source, {})
    assert parsed.arxiv == "2301.03216"
    assert parsed.year == 2023
