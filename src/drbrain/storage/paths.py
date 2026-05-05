"""Centralized path accessors for paper directories."""

from pathlib import Path


def paper_dir(papers_root: Path, local_id: str) -> Path:
    """Return the per-paper directory path."""
    return papers_root / local_id


def raw_md_path(paper_dir: Path) -> Path:
    """Return path to the MinerU markdown file."""
    return paper_dir / "raw.md"


def tree_json_path(paper_dir: Path) -> Path:
    """Return path to the PageIndex tree JSON file."""
    return paper_dir / "tree.json"


def source_pdf_path(paper_dir: Path) -> Path:
    """Return path to the source PDF copy."""
    return paper_dir / "source.pdf"


def images_dir(paper_dir: Path) -> Path:
    """Return path to the extracted images directory."""
    return paper_dir / "images"
