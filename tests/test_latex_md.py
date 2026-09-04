"""LaTeX → raw.md conversion tests (review §6.1: physics corpus path)."""

from __future__ import annotations

from drbrain.parser.latex_md import (
    latex_to_document,
    latex_to_markdown,
    markdown_to_tree,
)

SAMPLE = r"""
\documentclass{revtex4-2}
\begin{document}
\title{Kagome Metals}

% hidden comment %\cite{hidden}
\begin{abstract}
We study the flat band in kagome metals with \textbf{ARPES}.
\end{abstract}
\section{Introduction}
Kagome lattices host flat bands \cite{kane2011,ohgushi2000, kane2011}.
As shown in Eq.~\ref{eq:band},
\begin{equation}
E(k) = -2t\cos(ka)
\label{eq:band}
\end{equation}
\subsection{Model}
The Hamiltonian is $H = \sum_k t_{ij}$ in our notation.
\section{Results}
\begin{figure}
\includegraphics{fig1}
\caption{ARPES intensity map.}
\end{figure}
See \citep{ohgushi2000} for details.
\end{document}
"""


def test_citations_extracted_deduped_and_atomized():
    doc = latex_to_document(SAMPLE)
    assert doc.citations == ["kane2011", "ohgushi2000"]
    assert "[CITE:kane2011,ohgushi2000,kane2011]" in doc.markdown
    assert "hidden" not in doc.citations


def test_comments_stripped():
    doc = latex_to_document(SAMPLE)
    assert "hidden comment" not in doc.markdown


def test_abstract_hoisted_and_removed_from_body():
    doc = latex_to_document(SAMPLE)
    assert doc.markdown.startswith("## Abstract")
    assert doc.markdown.count("We study the flat band") == 1
    assert "ARPES" in doc.abstract


def test_display_math_stays_an_atom():
    doc = latex_to_document(SAMPLE)
    assert "```math" in doc.markdown
    # the equation body must survive simplification verbatim
    assert "E(k) = -2t\\cos(ka)" in doc.markdown


def test_inline_math_verbatim():
    doc = latex_to_document(SAMPLE)
    assert "$H = \\sum_k t_{ij}$" in doc.markdown


def test_headings_become_markdown():
    markdown = latex_to_markdown(SAMPLE)
    assert "## Introduction" in markdown
    assert "### Model" in markdown
    assert "## Results" in markdown


def test_floats_reduced_to_captions():
    doc = latex_to_document(SAMPLE)
    assert "includegraphics" not in doc.markdown
    assert "*ARPES intensity map.*" in doc.markdown


def test_refs_atomized():
    doc = latex_to_document(SAMPLE)
    assert "[REF:eq:band]" in doc.markdown


def test_markdown_tree_offsets_are_exact():
    doc = latex_to_document(SAMPLE)
    tree = markdown_to_tree(doc.markdown)
    lines = doc.markdown.split("\n")
    flat = tree["structure"]
    assert [n["title"] for n in flat] == ["Abstract", "Introduction", "Results"]
    intro = flat[1]
    assert intro["node_id"] == "sec-2"
    assert intro["nodes"][0]["title"] == "Model"
    assert intro["nodes"][0]["node_id"] == "sec-2-1"
    # line ranges are 0-based, end-exclusive, and slice back to real content
    start, end = intro["line_start"], intro["line_end"]
    assert lines[start] == "## Introduction"
    assert "Kagome lattices" in "\n".join(lines[start:end])
    # model subsection must fall inside the introduction's range
    assert intro["line_start"] <= intro["nodes"][0]["line_start"] < intro["line_end"]


# ── round-4 OCR findings ─────────────────────────────────────────────────────


def test_headings_with_nested_braces_become_markdown():
    """OCR r4: nested braces (\\texorpdfstring, subscripts) must still yield a heading."""
    nested = r"""
\begin{document}
\section{The \texorpdfstring{$X$}{X} model}
body one
\section{Fe$_3$Sn$_2$ \textemdash{} results}
body two
\end{document}
"""
    markdown = latex_to_markdown(nested)
    assert "## The X model" in markdown
    assert "## Fe$_3$Sn$_2$ results" in markdown
    assert "body one" in markdown
    assert "body two" in markdown
    # the heading must be a real tree node, not LaTeX residue in prose
    tree = markdown_to_tree(markdown)
    titles = [n["title"] for n in tree["structure"]]
    assert "The X model" in titles
    assert "Fe$_3$Sn$_2$ results" in titles


def test_abstract_display_math_is_protected():
    """OCR r4: $$...$$ / \\[..\\] inside the abstract must survive as atoms."""
    doc = latex_to_document(
        r"""
\begin{document}
\begin{abstract}
The gap opens as $$\Delta = 2\sqrt{m^2 - b^2}$$ which is the main result.
\end{abstract}
\section{Intro}
text
\end{document}
"""
    )
    assert r"\Delta = 2\sqrt{m^2 - b^2}" in doc.markdown
    assert "```math" in doc.markdown


def test_markdown_to_tree_has_no_min_node_lines_param():
    """OCR r4: the unused min_node_lines parameter was removed from the API."""
    import inspect

    sig = inspect.signature(markdown_to_tree)
    assert "min_node_lines" not in sig.parameters
