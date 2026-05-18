"""Explore silos — lightweight literature discovery collections.

Each silo is a directory containing a ``silo.json`` metadata file and
a ``papers.jsonl`` file with one paper dict per line. Silos are separate
from the main library and from workspaces — designed for exploratory
literature search.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

_SILO_NAME_RE = re.compile(r"^[a-zA-Z0-9][-a-zA-Z0-9_.]{0,63}$")


def _validate_silo_name(name: str) -> None:
    if not name or not _SILO_NAME_RE.match(name):
        raise ValueError(
            f"Invalid silo name: '{name}'. "
            "Must be 1-64 chars, letters/digits/hyphens/underscores/dots only."
        )


def _silo_dir(root: Path, name: str) -> Path:
    _validate_silo_name(name)
    return (root / name).resolve()


def _silo_json(root: Path, name: str) -> Path:
    return _silo_dir(root, name) / "silo.json"


def _papers_jsonl(root: Path, name: str) -> Path:
    return _silo_dir(root, name) / "papers.jsonl"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_explore_silo(
    root: Path,
    name: str,
    description: str = "",
) -> dict:
    """Create a new explore silo or return existing.

    Args:
        root: Base directory for explore silos (e.g. ``data/explore/``).
        name: Silo name (alphanumeric + hyphens/underscores/dots, 1-64 chars).
        description: Optional human-readable description.

    Returns:
        Silo metadata dict with ``name``, ``description``, ``created_at``,
        ``paper_count``.

    Raises:
        ValueError: If name is invalid.
    """
    _validate_silo_name(name)
    d = _silo_dir(root, name)
    d.mkdir(parents=True, exist_ok=True)

    json_path = _silo_json(root, name)
    if json_path.exists():
        return _read_json(json_path)

    meta = {
        "name": name,
        "description": description,
        "created_at": datetime.now(UTC).isoformat(),
        "paper_count": 0,
    }
    _write_json(json_path, meta)

    # Initialize empty papers JSONL
    _papers_jsonl(root, name).touch()
    return meta


def add_paper_to_silo(root: Path, name: str, paper: dict) -> None:
    """Append a paper dict to a silo's papers.jsonl.

    Args:
        root: Explore base directory.
        name: Silo name.
        paper: Dict with ``title``, ``authors``, ``year``, ``doi`` (optional).

    Raises:
        ValueError: If silo doesn't exist.
    """
    json_path = _silo_json(root, name)
    if not json_path.exists():
        raise ValueError(f"Silo not found: {name}")

    papers_path = _papers_jsonl(root, name)
    line = json.dumps(paper, ensure_ascii=False)
    with open(papers_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # Update paper count
    meta = _read_json(json_path)
    meta["paper_count"] = meta.get("paper_count", 0) + 1
    _write_json(json_path, meta)


def get_silo_papers(root: Path, name: str) -> list[dict]:
    """Read all papers from a silo.

    Args:
        root: Explore base directory.
        name: Silo name.

    Returns:
        List of paper dicts.

    Raises:
        ValueError: If silo doesn't exist.
    """
    json_path = _silo_json(root, name)
    if not json_path.exists():
        raise ValueError(f"Silo not found: {name}")

    papers_path = _papers_jsonl(root, name)
    if not papers_path.exists():
        return []

    papers: list[dict] = []
    for line in papers_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def search_silo(root: Path, name: str, query: str) -> list[dict]:
    """Case-insensitive keyword search over silo papers.

    Matches against ``title``, ``authors``, and ``doi`` fields.

    Args:
        root: Explore base directory.
        name: Silo name.
        query: Search query string.

    Returns:
        List of matching paper dicts, ordered by recency (newest papers last in
        file = most recent at end, so reversed).
    """
    papers = get_silo_papers(root, name)
    q = query.lower()
    results: list[dict] = []
    for p in papers:
        title = (p.get("title") or "").lower()
        authors = " ".join(p.get("authors") or []).lower()
        doi = (p.get("doi") or "").lower()
        if q in title or q in authors or q in doi:
            results.append(p)
    return results


def list_explore_silos(root: Path) -> list[dict]:
    """List all explore silos.

    Args:
        root: Explore base directory.

    Returns:
        List of silo metadata dicts.
    """
    if not root.exists():
        return []
    silos: list[dict] = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            jp = d / "silo.json"
            if jp.exists():
                silos.append(_read_json(jp))
    return silos


def delete_explore_silo(root: Path, name: str) -> None:
    """Delete an explore silo and all its data.

    Args:
        root: Explore base directory.
        name: Silo name.
    """
    d = _silo_dir(root, name)
    if d.exists():
        shutil.rmtree(d)
