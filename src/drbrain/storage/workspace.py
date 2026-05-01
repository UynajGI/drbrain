"""Workspace management — paper subsets for focused analysis."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml


class WorkspaceError(Exception):
    """Raised when a workspace operation fails."""


def _ws_dir(name: str, root: Path | None = None) -> Path:
    return (root or Path("workspace")) / name


def _papers_json(name: str, root: Path | None = None) -> Path:
    return _ws_dir(name, root) / "refs" / "papers.json"


def _yaml_path(name: str, root: Path | None = None) -> Path:
    return _ws_dir(name, root) / "workspace.yaml"


def _read_papers(name: str, root: Path | None = None) -> list[dict]:
    p = _papers_json(name, root)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _write_papers(name: str, papers: list[dict], root: Path | None = None) -> None:
    p = _papers_json(name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def create_workspace(
    name: str,
    root: Path | None = None,
    description: str = "",
) -> Path:
    """Create a new workspace directory with metadata.

    Raises WorkspaceError if workspace already exists.
    """
    ws = _ws_dir(name, root)
    if ws.exists():
        raise WorkspaceError(f"Workspace already exists: {name}")

    refs = ws / "refs"
    refs.mkdir(parents=True)

    now = datetime.now(UTC).isoformat()
    yaml_data = {
        "name": name,
        "description": description or "",
        "created": now,
    }
    _yaml_path(name, root).write_text(
        yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )

    _write_papers(name, [], root)
    return ws


def add_papers(name: str, local_ids: list[str], root: Path | None = None) -> None:
    """Add papers to a workspace. Ignores duplicates.

    Raises WorkspaceError if workspace does not exist.
    """
    ws = _ws_dir(name, root)
    if not ws.exists():
        raise WorkspaceError(f"Workspace not found: {name}")

    papers = _read_papers(name, root)
    existing = {p["local_id"] for p in papers}
    now = datetime.now(UTC).isoformat()
    for lid in local_ids:
        if lid not in existing:
            papers.append({"local_id": lid, "added_at": now})
            existing.add(lid)
    _write_papers(name, papers, root)


def remove_papers(name: str, local_ids: list[str], root: Path | None = None) -> None:
    """Remove papers from a workspace."""
    papers = _read_papers(name, root)
    remove = set(local_ids)
    papers = [p for p in papers if p["local_id"] not in remove]
    _write_papers(name, papers, root)


def list_workspaces(root: Path | None = None) -> list[str]:
    """Return sorted list of workspace names."""
    root = root or Path("workspace")
    if not root.exists():
        return []
    names = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "workspace.yaml").exists():
            names.append(d.name)
    return names


def get_workspace(name: str, root: Path | None = None) -> dict | None:
    """Get workspace info including paper list."""
    ws = _ws_dir(name, root)
    if not ws.exists():
        return None
    yaml_data = yaml.safe_load(_yaml_path(name, root).read_text(encoding="utf-8"))
    papers = _read_papers(name, root)
    return {
        "name": yaml_data.get("name", name),
        "description": yaml_data.get("description", ""),
        "created": yaml_data.get("created", ""),
        "paper_count": len(papers),
        "papers": [p["local_id"] for p in papers],
    }


def delete_workspace(name: str, root: Path | None = None) -> None:
    """Delete a workspace directory entirely."""
    ws = _ws_dir(name, root)
    if ws.exists():
        shutil.rmtree(ws)


def load_workspace_papers(name: str, root: Path | None = None) -> list[str]:
    """Load paper IDs from a workspace. Returns empty list if not found."""
    papers = _read_papers(name, root)
    return [p["local_id"] for p in papers]
