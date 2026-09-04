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
