"""Heuristic section detection and fallback parsing for MinerU output."""

from __future__ import annotations

import re

MAX_CHARS = 12_000

# Markdown heading sections
HEADING_SECTIONS = re.compile(
    r"^(abstract|introduction|related\s*work|method(ology)?|"
    r"conclusion|limitations?|future\s*work|discussion|results|"
    r"supplementary\s*material)$",
    re.IGNORECASE,
)

# Inline section markers: "Introduction.—", "Conclusion. —", etc.
INLINE_SECTION = re.compile(
    r"^(Abstract|Introduction|Related\s*(Work|Work).{0,5}|Methods?|"
    r"Methodology|Conclusion|Limitations?|Future\s*Work|Discussion|"
    r"Results|GME and reflected entropy|A new measure)"
    r"[.\s:—–-]",
    re.IGNORECASE,
)

ID_PATTERN = re.compile(r"(10\.\d{4,}/[\S]+|arxiv[:\s]+(\d{4}\.\d{4,5}))", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_FILENAME = re.compile(r"(\d{4}\.\d{4,5})v\d*\.pdf$", re.IGNORECASE)


def filter_sections(raw_md: str) -> list[str]:
    """Extract target academic sections from mineru output.

    Handles both markdown headings (# Introduction) and inline markers (Introduction.—).
    Falls back to returning all text if no sections detected.
    """
    lines = raw_md.splitlines()
    blocks: list[str] = []
    current_section = ""
    current_body: list[str] = []
    found_any_target = False

    for line in lines:
        # Check for markdown heading sections
        if line.startswith("#"):
            heading_text = line.lstrip("# ").strip()
            if current_body and HEADING_SECTIONS.match(current_section):
                joined = "\n".join(current_body)
                if joined.strip():
                    blocks.append(joined[:MAX_CHARS])
                    found_any_target = True
            current_section = heading_text
            current_body = []
        else:
            # Check for inline section marker at start of a line/paragraph
            if INLINE_SECTION.match(line.strip()):
                if current_body and HEADING_SECTIONS.match(current_section):
                    joined = "\n".join(current_body)
                    if joined.strip():
                        blocks.append(joined[:MAX_CHARS])
                        found_any_target = True
                # Extract section name from inline marker
                m = INLINE_SECTION.match(line.strip())
                if m:
                    current_section = m.group(1)
                    # The rest of the line after the marker is content
                    rest = line.strip()[m.end() :]
                    current_body = [rest] if rest.strip() else []
                else:
                    current_body.append(line)
            else:
                current_body.append(line)

    # Handle last section
    if current_body and HEADING_SECTIONS.match(current_section):
        joined = "\n".join(current_body)
        if joined.strip():
            blocks.append(joined[:MAX_CHARS])
            found_any_target = True

    # If no target sections found, return all text (excluding title line and thinking line)
    if not found_any_target:
        filtered_lines = [line for line in lines if not line.startswith("Thinking...")]
        text = "\n".join(filtered_lines).strip()
        if text:
            # Split into chunks if too large
            if len(text) > MAX_CHARS:
                return [text[:MAX_CHARS]]
            return [text]
        return []

    return blocks
