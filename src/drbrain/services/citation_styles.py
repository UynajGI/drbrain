"""Citation style management for Markdown reference export.

Built-in styles: apa, vancouver, chicago-author-date, mla
Custom styles: loaded dynamically from the configured citation styles directory.

Formatter interface (every custom style file must implement):

    def format_ref(meta: dict, idx: int | None = None) -> str:
        '''Return a single formatted reference line (Markdown).

        Args:
            meta: Paper metadata dict (title, authors, year, journal,
                  volume, issue, pages, doi, publisher, paper_type, ...)
            idx: 1-based index for numbered lists; None for bullet lists.
        Returns:
            Formatted reference string, e.g. "1. Smith et al. (2023). ..."
        '''
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

FormatterFn = Callable[[dict, int | None], str]

# ── Built-in styles ─────────────────────────────────────────────────


def _fmt_apa(meta: dict, idx: int | None = None) -> str:
    """APA 7th edition (author-date, ampersand, italicised journal+volume)."""
    authors = meta.get("authors") or []
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) <= 3:
        author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        author_str = f"{authors[0]} et al."

    year = meta.get("year") or "n.d."
    title = meta.get("title") or "Untitled"
    journal = meta.get("journal") or ""
    volume = meta.get("volume") or ""
    issue = meta.get("issue") or ""
    pages = meta.get("pages") or ""
    doi = meta.get("doi") or ""

    journal_part = ""
    if journal:
        journal_part = f"*{journal}*"
        if volume:
            journal_part += f", *{volume}*"
            if issue:
                journal_part += f"({issue})"
        if pages:
            journal_part += f", {pages}"

    ref = f"{author_str} ({year}). {title}."
    if journal_part:
        ref += f" {journal_part}."
    if doi:
        ref += f" https://doi.org/{doi}"

    prefix = f"{idx}. " if idx is not None else "- "
    return prefix + ref


def _fmt_vancouver(meta: dict, idx: int | None = None) -> str:
    """Vancouver / ICMJE numbered style (used by most biomedical journals)."""
    authors = meta.get("authors") or []

    def _initials(name: str) -> str:
        parts = name.split(",", 1)
        if len(parts) == 2:
            last, first = parts[0].strip(), parts[1].strip()
            initials = "".join(w[0] for w in first.split() if w)
            return f"{last} {initials}"
        return name

    if not authors:
        author_str = "Unknown"
    elif len(authors) <= 6:
        author_str = ", ".join(_initials(a) for a in authors)
    else:
        author_str = ", ".join(_initials(a) for a in authors[:6]) + ", et al"

    year = meta.get("year") or "n.d."
    title = meta.get("title") or "Untitled"
    journal = meta.get("journal") or ""
    volume = meta.get("volume") or ""
    issue = meta.get("issue") or ""
    pages = meta.get("pages") or ""
    doi = meta.get("doi") or ""

    ref = f"{author_str}. {title}."
    if journal:
        ref += f" {journal}."
    if year:
        ref += f" {year}"
    if volume:
        ref += f";{volume}"
    if issue:
        ref += f"({issue})"
    if pages:
        ref += f":{pages}"
    ref = ref.rstrip(";:") + "."
    if doi:
        ref += f" doi:{doi}"

    prefix = f"{idx}. " if idx is not None else "- "
    return prefix + ref


def _fmt_chicago_author_date(meta: dict, idx: int | None = None) -> str:
    """Chicago 17th ed. Author-Date (common in humanities/social sciences)."""
    authors = meta.get("authors") or []

    def _chicago_author(name: str, first: bool) -> str:
        parts = name.split(",", 1)
        if len(parts) == 2:
            last, given = parts[0].strip(), parts[1].strip()
            return f"{last}, {given}" if first else f"{given} {last}"
        return name

    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = _chicago_author(authors[0], first=True)
    elif len(authors) <= 3:
        formatted = [_chicago_author(authors[0], first=True)]
        formatted += [_chicago_author(a, first=False) for a in authors[1:]]
        author_str = ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
    else:
        author_str = _chicago_author(authors[0], first=True).rstrip(".") + " et al."

    year = meta.get("year") or "n.d."
    title = meta.get("title") or "Untitled"
    journal = meta.get("journal") or ""
    volume = meta.get("volume") or ""
    issue = meta.get("issue") or ""
    pages = meta.get("pages") or ""
    doi = meta.get("doi") or ""

    ref = f'{author_str.rstrip(".")}. {year}. "{title}."'
    if journal:
        ref += f" *{journal}*"
    if volume:
        ref += f" {volume}"
    if issue:
        ref += f" ({issue})"
    if pages:
        ref += f": {pages}"
    ref = ref.rstrip(":") + "."
    if doi:
        ref += f" https://doi.org/{doi}"

    prefix = f"{idx}. " if idx is not None else "- "
    return prefix + ref


def _fmt_mla(meta: dict, idx: int | None = None) -> str:
    """MLA 9th edition (humanities, container model)."""
    authors = meta.get("authors") or []

    def _mla_author(name: str, reverse: bool) -> str:
        parts = name.split(",", 1)
        if len(parts) == 2:
            last, given = parts[0].strip(), parts[1].strip()
            return f"{last}, {given}" if reverse else f"{given} {last}"
        return name

    if len(authors) == 1:
        author_str = _mla_author(authors[0], reverse=True)
    elif len(authors) == 2:
        author_str = (
            f"{_mla_author(authors[0], reverse=True)}, and {_mla_author(authors[1], reverse=False)}"
        )
    elif authors:
        author_str = _mla_author(authors[0], reverse=True).rstrip(".") + ", et al."
    else:
        author_str = "Unknown"

    title = meta.get("title") or "Untitled"
    journal = meta.get("journal") or ""
    volume = meta.get("volume") or ""
    issue = meta.get("issue") or ""
    year = meta.get("year") or "n.d."
    pages = meta.get("pages") or ""
    doi = meta.get("doi") or ""

    ref = f'{author_str.rstrip(".")}. "{title}."'
    if journal:
        ref += f" *{journal}*"
    if volume:
        ref += f", vol. {volume}"
    if issue:
        ref += f", no. {issue}"
    if year:
        ref += f", {year}"
    if pages:
        ref += f", pp. {pages}"
    ref = ref.rstrip(",") + "."
    if doi:
        ref += f" https://doi.org/{doi}"

    prefix = f"{idx}. " if idx is not None else "- "
    return prefix + ref


BUILTIN_STYLES: dict[str, FormatterFn] = {
    "apa": _fmt_apa,
    "vancouver": _fmt_vancouver,
    "chicago-author-date": _fmt_chicago_author_date,
    "mla": _fmt_mla,
}

BUILTIN_DESCRIPTIONS: dict[str, str] = {
    "apa": "APA 7th edition (author-year, default)",
    "vancouver": "Vancouver / ICMJE numeric style (biomedical journals)",
    "chicago-author-date": "Chicago 17th edition author-date style (humanities and social sciences)",
    "mla": "MLA 9th edition (humanities, container model)",
}

DEFAULT_STYLES_DIR = Path("data/citation_styles")


# ── Public API ──────────────────────────────────────────────────────


def list_styles(styles_dir: Path | None = None) -> list[dict]:
    """Return all available styles with name, source, and description.

    Args:
        styles_dir: Directory for custom style files. Defaults to data/citation_styles/.

    Returns:
        List of dicts with keys ``name``, ``source``, ``description``.
    """
    d = Path(styles_dir) if styles_dir else DEFAULT_STYLES_DIR
    results: list[dict] = []
    for name, desc in BUILTIN_DESCRIPTIONS.items():
        results.append({"name": name, "source": "built-in", "description": desc})

    if d.exists():
        for py_file in sorted(d.glob("*.py")):
            name = py_file.stem
            if name in BUILTIN_STYLES:
                continue
            meta_file = d / f"{name}.json"
            desc = ""
            source = "custom"
            if meta_file.exists():
                try:
                    m = json.loads(meta_file.read_text(encoding="utf-8"))
                    desc = m.get("description", "")
                    source = m.get("source", "custom")
                except Exception:
                    pass
            results.append({"name": name, "source": source, "description": desc})

    return results


def get_formatter(name: str, styles_dir: Path | None = None) -> FormatterFn:
    """Load a formatter by style name.

    Checks built-in styles first, then the custom style directory.

    Args:
        name: Citation style name (e.g. ``"apa"``, ``"vancouver"``).
        styles_dir: Directory for custom style files. Defaults to data/citation_styles/.

    Returns:
        A callable ``(meta: dict, idx: int | None) -> str`` that formats one reference.

    Raises:
        ValueError: Invalid style name or path traversal.
        FileNotFoundError: Style not found.
        ImportError: Custom style file cannot be loaded.
        AttributeError: Style file does not define ``format_ref``.
    """
    if name in BUILTIN_STYLES:
        return BUILTIN_STYLES[name]

    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Invalid citation style name '{name}': "
            "only letters, digits, hyphens, and underscores are allowed."
        )

    d = Path(styles_dir) if styles_dir else DEFAULT_STYLES_DIR
    style_file = (d / f"{name}.py").resolve()
    if not style_file.is_relative_to(d.resolve()):
        raise ValueError(f"Invalid citation style name '{name}': path traversal detected.")
    if not style_file.exists():
        available = ", ".join(s["name"] for s in list_styles(d))
        raise FileNotFoundError(
            f"Citation style '{name}' does not exist.\n"
            f"Available styles: {available}\n"
            f"To add a new style, place a Python file at {style_file}"
            f" with a `format_ref(meta, idx)` function."
        )

    spec = importlib.util.spec_from_file_location(f"_csl_{name}", style_file)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load citation style '{name}': "
            f"import system returned no valid spec/loader ({style_file})"
        )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        raise ImportError(f"Failed to load citation style '{name}' ({style_file}): {exc}") from exc
    if not hasattr(mod, "format_ref"):
        raise AttributeError(
            f"Style file {style_file} must define a `format_ref(meta, idx)` function."
        )
    return mod.format_ref


def show_style(name: str, styles_dir: Path | None = None) -> str:
    """Return source code of a custom style, or a description for built-ins.

    Args:
        name: Citation style name.
        styles_dir: Directory for custom style files.

    Returns:
        Source code string of the style file, or a comment block for built-in styles.

    Raises:
        ValueError: Invalid name or path traversal.
        FileNotFoundError: Custom style file does not exist.
    """
    if name in BUILTIN_STYLES:
        desc = BUILTIN_DESCRIPTIONS.get(name, "")
        return (
            f"# Built-in style: {name}\n"
            f"# {desc}\n"
            f"# (implemented in drbrain/services/citation_styles.py)"
        )

    import re as _re

    if not _re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Invalid citation style name '{name}': "
            "only letters, digits, hyphens, and underscores are allowed."
        )

    d = Path(styles_dir) if styles_dir else DEFAULT_STYLES_DIR
    style_file = (d / f"{name}.py").resolve()
    if not style_file.is_relative_to(d.resolve()):
        raise ValueError(f"Invalid citation style name '{name}': path traversal detected.")
    if not style_file.exists():
        raise FileNotFoundError(f"Citation style '{name}' does not exist.")
    return style_file.read_text(encoding="utf-8")


def format_refs(
    metas: list[dict],
    style: str = "apa",
    styles_dir: Path | None = None,
) -> str:
    """Format a list of paper metadata dicts into styled Markdown references.

    Args:
        metas: List of paper metadata dicts.
        style: Style name (default: "apa").
        styles_dir: Custom styles directory.

    Returns:
        Newline-joined formatted references.
    """
    fmt = get_formatter(style, styles_dir)
    if style in ("vancouver",):
        # Numbered list
        return "\n".join(fmt(m, i) for i, m in enumerate(metas, 1))
    else:
        # Bullet list
        return "\n".join(fmt(m, None) for m in metas)
