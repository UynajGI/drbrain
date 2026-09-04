"""LaTeX → raw.md conversion for arXiv-scale corpora (review §6).

0.8M-paper physics corpora never pass through PDF/MinerU: the raw arXiv
LaTeX source is the input. This module turns a LaTeX body into the same
``raw.md`` + ``tree.json`` shape the ingest pipeline produces for PDFs, with
three corpus-level invariants (review §6.1):

- math stays an **atom**: display math becomes a fenced block so chunking
  never splits an equation across retrieval units;
- ``\\cite`` keys are extracted into ``[CITE:...]`` atoms (and returned as a
  list) so citation edges are parseable without re-reading LaTeX;
- ``\\section``-style headings become markdown headings, which
  :func:`drbrain.parser.latex_md.markdown_to_tree` turns into a PageIndex
  tree with exact ``line_start``/``line_end`` offsets into the produced
  raw.md — cheaper and more accurate than LLM tree building.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# ── LaTeX → markdown ─────────────────────────────────────────────────────────

_HEADING_COMMANDS = {
    "section": "##",
    "subsection": "###",
    "subsubsection": "####",
    # paragraph/subparagraph 各降一级，避免与 subsubsection 在树里同层折叠
    # （OCR r6：四级同层会丢一档文档层级）。
    "paragraph": "#####",
    "subparagraph": "######",
}

_VERB_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "math",
}

_CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear)\*?\s*(\[[^]]*\])*\s*\{([^}]*)\}"
)
_REF_RE = re.compile(r"\\(?:ref|autoref|eqref|pageref|Cref|cref)\s*\{([^}]*)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
_COMMENT_RE = re.compile(r"(?m)^((?:[^%\\]|\\.)*)%.*$")
_GROUP_RE = re.compile(
    r"\\(?:text|textbf|textit|texttt|textrm|mathrm|mathbf|mathit|emph)\{([^{}]*)\}"
)
_BRACE_CMD_RE = re.compile(r"\\(?:title|author|date|thanks)\s*(\[[^]]*\])?\s*\{")
_RESIDUAL_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?(\[[^]]*\])?")
_DOLLAR_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def _strip_comments(latex: str) -> str:
    """Drop ``%`` comments (an unescaped percent to end of line)."""
    return _COMMENT_RE.sub(r"\1", latex)


def _extract_body(latex: str) -> str:
    """Return the document body (after ``\\begin{document}`` when present)."""
    match = re.search(r"\\begin\{document\}", latex)
    if match:
        latex = latex[match.end() :]
    end = re.search(r"\\end\{document\}", latex)
    if end:
        latex = latex[: end.start()]
    return latex


def _extract_abstract(latex: str) -> tuple[str, str]:
    """Return ``(abstract, body-with-abstract-removed)``."""
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex, re.DOTALL)
    if not match:
        return "", latex
    return match.group(1).strip(), latex[: match.start()] + latex[match.end() :]


def _fence(body: str, lang: str = "math") -> str:
    return f"\n\n```{lang}\n{body.strip()}\n```\n\n"


def _protect_display_math(latex: str) -> str:
    """Wrap display math environments and ``$$...$$`` in fenced blocks."""

    def _env_repl(match: re.Match) -> str:
        return _fence(f"\\begin{{{match.group(1)}}}\n{match.group(2)}\n\\end{{{match.group(1)}}}")

    latex = re.sub(
        r"\\begin\{(" + "|".join(re.escape(e) for e in _VERB_ENVIRONMENTS) + r")\}(.*?)\\end\{\1\}",
        _env_repl,
        latex,
        flags=re.DOTALL,
    )
    latex = _DOLLAR_DISPLAY_RE.sub(lambda m: _fence(m.group(1)), latex)
    latex = re.sub(
        r"(?<!\\)\\\[(.*?)\\\]",
        lambda m: _fence(m.group(1)),
        latex,
        flags=re.DOTALL,
    )
    return latex


_MATH_ATOM_SPLIT_RE = re.compile(
    r"(```math\n.*?\n```|(?<!\\)\$(?!\$)(?:\\.|[^$\\])+?(?<!\\)\$)",
    re.DOTALL,
)


def _simplify_inline(latex: str) -> str:
    """Normalize the LaTeX subset that carries no markdown-relevant meaning.

    Math atoms (fenced display blocks and ``$...$`` spans) pass through
    verbatim — simplification must never eat a command inside an equation.
    """
    out: list[str] = []
    for i, part in enumerate(_MATH_ATOM_SPLIT_RE.split(latex)):
        if i % 2 == 1:
            out.append(part)
            continue
        part = _GROUP_RE.sub(r"\1", part)
        part = _BRACE_CMD_RE.sub("", part)
        part = part.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
        part = part.replace("~", " ")
        part = re.sub(r"\\newline|\\\\", "\n", part)
        part = _RESIDUAL_CMD_RE.sub(" ", part)
        part = part.replace("{", "").replace("}", "")
        out.append(part)
    return "".join(out)


def _clean_heading_text(text: str) -> str:
    """Reduce a LaTeX heading argument to plain markdown-safe text.

    Tolerates one level of brace nesting (real headings nest:
    ``\\section{The \\texorpdfstring{$X$}{X} model}``), keeps ``\\texorpdfstring``
    's second (plain-text) argument, unwraps font commands and drops residual
    markup so the title survives as a readable markdown heading.
    """
    text = re.sub(
        r"\\texorpdfstring\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
        r"\2",
        text,
    )
    text = _GROUP_RE.sub(r"\1", text)
    text = _RESIDUAL_CMD_RE.sub(" ", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _headings_and_lists(latex: str) -> str:
    # 允许一层嵌套花括号：\section{Fe$_3$Sn$_2$ \textemdash{} results} 这类
    # 真实标题用 \{([^{}]*)\} 匹配不到，会整段漏进散文管线被撕碎。
    latex = re.sub(
        r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(\[[^]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
        lambda m: f"\n\n{_HEADING_COMMANDS[m.group(1)]} {_clean_heading_text(m.group(3))}\n\n",
        latex,
    )
    latex = re.sub(r"\\item\s?", "\n- ", latex)
    latex = re.sub(
        r"\\begin\{(itemize|enumerate)\}|\s*\\end\{(itemize|enumerate)\}",
        "\n",
        latex,
    )
    return latex


def _extract_figure_captions(latex: str) -> str:
    """Drop float bodies, keep captions as plain paragraphs."""

    def _float_repl(match: re.Match) -> str:
        # 容忍一层嵌套花括号（OCR r6）：\caption{Intensity of Fe$_3$Sn$_2$ ...}
        # 用 \{([^{}]*)\} 匹配不到时整条 float 连同 caption 一起被丢。
        caption = re.search(
            r"\\caption\s*(\[[^]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", match.group(0)
        )
        if not caption:
            return "\n"
        return f"\n\n*{_clean_heading_text(caption.group(2))}*\n\n"

    latex = re.sub(
        r"\\begin\{(figure|table|figure\*|table\*|wraptable|wrapfigure|sidewaysfigure|sidewaystable)\}.*?\\end\{\1\}",
        _float_repl,
        latex,
        flags=re.DOTALL,
    )
    return latex


@dataclass
class LatexDocument:
    """Converted paper plus the metadata needed for corpus ingestion."""

    markdown: str
    citations: list[str] = field(default_factory=list)
    abstract: str = ""

    @property
    def content_fingerprint(self) -> str:
        """Stable per-document fingerprint (fallback content hash)."""
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()[:16]


def latex_to_document(latex: str) -> LatexDocument:
    """Convert one arXiv LaTeX source into pipeline-ready markdown."""
    latex = _strip_comments(latex)
    latex = _extract_body(latex)
    abstract, latex = _extract_abstract(latex)
    latex = _protect_display_math(latex)
    latex = _extract_figure_captions(latex)

    citations: list[str] = []
    for match in _CITE_RE.finditer(latex):
        for key in match.group(2).split(","):
            # 与 [CITE:...] 原子同一归一化（OCR r6）：两侧表示必须能 join。
            key = key.strip().replace(" ", "")
            if key:
                citations.append(key)
    latex = _CITE_RE.sub(lambda m: f" [CITE:{m.group(2).replace(' ', '')}] ", latex)
    latex = _REF_RE.sub(lambda m: f" [REF:{m.group(1)}] ", latex)
    latex = _LABEL_RE.sub(" ", latex)

    latex = _headings_and_lists(latex)
    latex = _simplify_inline(latex)
    markdown = re.sub(r"\n{3,}", "\n\n", latex).strip()

    abstract_md = ""
    if abstract:
        # 摘要与正文走同一套数学原子保护：摘要里的 $$...$$ / \[..\] 不先
        # protect 就进 _simplify_inline 会被原子拆分正则撕碎。
        abstract_md = _simplify_inline(_protect_display_math(abstract)).strip()
        markdown = f"## Abstract\n\n{abstract_md}\n\n{markdown}"
    return LatexDocument(
        markdown=markdown,
        citations=list(dict.fromkeys(citations)),
        # 字段存转换后的文本（OCR r6）：raw LaTeX 的残留（\textbf 等）一旦
        # 被下游 embed/展示就会漏出去。
        abstract=abstract_md,
    )


def latex_to_markdown(latex: str) -> str:
    """Convenience wrapper returning only the converted markdown."""
    return latex_to_document(latex).markdown


# ── markdown → PageIndex tree ────────────────────────────────────────────────


def markdown_to_tree(markdown: str) -> dict:
    """Build a ``tree.json`` structure from markdown headings.

    Every node carries exact ``line_start``/``line_end`` offsets into the
    markdown (0-based, end-exclusive — matching
    ``drbrain.services.embedding._collect_tree_nodes``), so RAG-DB filling
    reproduces pipeline text construction without an LLM pass.
    """
    lines = markdown.split("\n")
    root: dict = {"structure": []}
    stack: list[tuple[int, dict]] = []  # (level, node)

    def _close(depth: int, end_line: int) -> None:
        while stack and stack[-1][0] >= depth:
            level, node = stack.pop()
            node["line_end"] = end_line
            parent = stack[-1][1]["nodes"] if stack else root["structure"]
            parent.append(node)

    heading_re = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
    for idx, line in enumerate(lines):
        match = heading_re.match(line)
        if not match:
            continue
        level = len(match.group(1))
        _close(level, idx)
        node: dict = {
            "node_id": "",
            "title": match.group(2).strip(),
            "line_start": idx,
            "nodes": [],
        }
        stack.append((level, node))
    _close(0, len(lines))

    def _assign_ids(nodes: list[dict], prefix: str = "sec") -> None:
        for i, node in enumerate(nodes, start=1):
            node["node_id"] = f"{prefix}-{i}"
            if node["nodes"]:
                _assign_ids(node["nodes"], node["node_id"])

    _assign_ids(root["structure"])
    return root
