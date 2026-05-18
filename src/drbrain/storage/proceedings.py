"""Proceedings management — lightweight JSON-backed store.

Conference proceedings are stored as a JSON array in a file
(default: data/proceedings.json). Each proceeding has an id,
name, year, venue, and a list of associated paper local_ids.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

DEFAULT_PATH = Path("data/proceedings.json")


def _read_store(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_store(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_proceeding(
    path: Path,
    name: str,
    year: int,
    venue: str = "",
) -> dict:
    """Create a new proceedings entry or return existing.

    Args:
        path: Path to proceedings.json store file.
        name: Conference name (e.g. "NeurIPS").
        year: Conference year.
        venue: Location string (e.g. "Vancouver").

    Returns:
        Proceeding dict with ``id``, ``name``, ``year``, ``venue``, ``papers``.
    """
    data = _read_store(path)

    # Check for duplicate
    for p in data:
        if p["name"] == name and p["year"] == year:
            return dict(p)

    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "year": year,
        "venue": venue,
        "papers": [],
    }
    data.append(entry)
    _write_store(path, data)
    return dict(entry)


def add_paper(path: Path, proceeding_id: str, paper_id: str) -> None:
    """Add a paper local_id to a proceeding.

    Args:
        path: Path to proceedings.json.
        proceeding_id: Proceeding ID.
        paper_id: Paper local_id to add.

    Raises:
        ValueError: If proceeding_id not found.
    """
    data = _read_store(path)
    for p in data:
        if p["id"] == proceeding_id:
            papers: list[str] = p.get("papers", [])
            if paper_id not in papers:
                papers.append(paper_id)
                p["papers"] = papers
                _write_store(path, data)
            return
    raise ValueError(f"Proceeding '{proceeding_id}' not found")


def list_proceedings(path: Path) -> list[dict]:
    """List all proceedings.

    Args:
        path: Path to proceedings.json.

    Returns:
        List of proceeding dicts sorted by year desc, name.
    """
    data = _read_store(path)
    return sorted(data, key=lambda p: (-p.get("year", 0), p.get("name", "")))


def get_proceeding(path: Path, proceeding_id: str) -> dict | None:
    """Get a single proceeding by ID.

    Args:
        path: Path to proceedings.json.
        proceeding_id: Proceeding ID.

    Returns:
        Proceeding dict or None.
    """
    for p in _read_store(path):
        if p["id"] == proceeding_id:
            return dict(p)
    return None


def load_proceedings(path: Path) -> list[dict]:
    """Load all proceedings (alias for list_proceedings with no sorting)."""
    return _read_store(path)
