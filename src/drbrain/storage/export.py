"""Export papers to BibTeX, RIS, and Markdown formats."""

from __future__ import annotations

import re


def _bibtex_escape(text: str) -> str:
    for ch in ("&", "%", "#", "_"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _extract_lastname(full_name: str) -> str:
    """Extract last name from a full author name. Handles:
    - Chinese names: first character is surname
    - Particles: de, van, von, del, della, di, le, la
    - Initials: "J. K. Eaton" → "Eaton"
    - Simple: last word is surname
    """
    if not full_name:
        return ""
    name = full_name.strip()
    # Chinese name detection
    if re.search(r"[一-鿿]", name):
        return name[0]
    parts = name.split()
    if not parts:
        return ""
    particles = {"de", "van", "von", "del", "della", "di", "le", "la"}
    # If all but last are initials (e.g. "J. K. Rowling"), take last
    if len(parts) >= 2 and all(len(p.rstrip(".")) <= 2 for p in parts[:-1]):
        return parts[-1]
    # Collect trailing particles + surname (e.g. "Vincent van Gogh" → "van Gogh")
    i = len(parts) - 1
    while i > 0 and parts[i - 1].lower() in particles:
        i -= 1
    return parts[-1] if i == len(parts) - 1 else " ".join(parts[i:])


def _make_cite_key(meta: dict) -> str:
    last = meta.get("first_author_lastname") or "Unknown"
    last = re.sub(r"[^a-zA-Z]", "", last)
    year = str(meta.get("year") or "")
    title = meta.get("title") or ""
    word = ""
    for w in title.split():
        cleaned = re.sub(r"[^a-zA-Z]", "", w)
        if len(cleaned) > 3:
            word = cleaned.capitalize()
            break
    return f"{last}{year}{word}"


def _bibtex_entry_type(meta: dict) -> str:
    paper_type = meta.get("paper_type", "paper")
    mapping = {
        "paper": "article",
        "review": "article",
        "thesis": "phdthesis",
        "preprint": "misc",
        "book": "book",
        "document": "misc",
    }
    return mapping.get(paper_type, "article")


def meta_to_bibtex(meta: dict) -> str:
    key = _make_cite_key(meta)
    entry_type = _bibtex_entry_type(meta)
    title = _bibtex_escape(meta.get("title", "Untitled"))

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  title = {{{title}}},")

    year = meta.get("year")
    if year:
        lines.append(f"  year = {{{year}}},")

    authors = meta.get("authors", "")
    if authors:
        lines.append(f"  author = {{{_bibtex_escape(authors)}}},")

    journal = meta.get("journal", "")
    if journal:
        lines.append(f"  journal = {{{_bibtex_escape(journal)}}},")

    volume = meta.get("volume", "")
    if volume:
        lines.append(f"  volume = {{{volume}}},")

    pages = meta.get("pages", "")
    if pages:
        lines.append(f"  pages = {{{pages}}},")

    doi = meta.get("doi", "")
    if doi:
        lines.append(f"  doi = {{{doi}}},")

    arxiv = meta.get("arxiv", "")
    if arxiv:
        lines.append(f"  note = {{arXiv:{arxiv}}},")

    lines.append("}")
    return "\n".join(lines)


def meta_to_ris(meta: dict) -> str:
    paper_type = meta.get("paper_type", "paper")
    ty_mapping = {
        "paper": "JOUR",
        "review": "JOUR",
        "thesis": "THES",
        "preprint": "GEN",
        "book": "BOOK",
        "document": "GEN",
    }
    lines = [f"TY  - {ty_mapping.get(paper_type, 'JOUR')}"]
    lines.append(f"TI  - {meta.get('title', 'Untitled')}")

    year = meta.get("year")
    if year:
        lines.append(f"PY  - {year}")

    authors = meta.get("authors", "")
    if authors:
        for author in authors.split(" and "):
            lines.append(f"AU  - {author.strip()}")

    journal = meta.get("journal", "")
    if journal:
        lines.append(f"JF  - {journal}")

    volume = meta.get("volume", "")
    if volume:
        lines.append(f"VL  - {volume}")

    pages = meta.get("pages", "")
    if pages:
        lines.append(f"SP  - {pages.split('-')[0]}")
        if "-" in pages:
            lines.append(f"EP  - {pages.split('-')[1]}")

    doi = meta.get("doi", "")
    if doi:
        lines.append(f"DO  - {doi}")

    lines.append("ER  -")
    return "\n".join(lines)


def meta_to_markdown(meta: dict, style: str = "apa", styles_dir: str | None = None) -> str:
    title = meta.get("title", "Untitled")
    year = meta.get("year", "")
    year_str = f" ({year})" if year else ""
    authors = meta.get("authors") or "Unknown"
    parts = [f"- **{title}**{year_str}, {authors}"]

    journal = meta.get("journal", "")
    if journal:
        parts[0] += f", *{journal}*"

    doi = meta.get("doi", "")
    if doi:
        parts.append(f"  DOI: [{doi}](https://doi.org/{doi})")

    return "\n".join(parts)


def batch_export(metas: list[dict], fmt: str, style: str = "apa") -> str:
    if fmt == "bib":
        return "\n\n".join(meta_to_bibtex(m) for m in metas)
    elif fmt == "ris":
        return "\n\n".join(meta_to_ris(m) for m in metas)
    elif fmt == "md":
        from drbrain.services.citation_styles import format_refs

        return format_refs(metas, style=style)
    return ""
