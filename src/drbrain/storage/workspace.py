"""Workspace management — paper subsets for focused analysis."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from drbrain.exceptions import DrBrainError


class WorkspaceError(DrBrainError):
    """Raised when a workspace operation fails."""


_INVALID_NAME_CHARS = {"/", "\\", ":"}


def validate_workspace_name(name: str) -> bool:
    """Return True if *name* is a safe workspace name.

    Rejects: empty, ".", "..", absolute paths, path separators,
    ".." anywhere, ":" (Windows drive), leading/trailing whitespace.
    """
    if not name:
        return False
    if name != name.strip():
        return False
    if name in (".", ".."):
        return False
    if name.startswith("/"):
        return False
    if ".." in name:
        return False
    if any(c in _INVALID_NAME_CHARS for c in name):
        return False
    return True


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
    data = json.dumps(papers, indent=2, ensure_ascii=False) + "\n"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(p)


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
        "schema_version": 1,
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


def rename_workspace(
    old_name: str,
    new_name: str,
    root: Path | None = None,
) -> Path:
    """Rename a workspace directory.

    Validates both names against ``validate_workspace_name``.

    Raises:
        ValueError: If either name fails validation.
        FileNotFoundError: If *old_name* workspace does not exist.
        FileExistsError: If *new_name* workspace already exists.
    """
    if not validate_workspace_name(old_name):
        raise ValueError(f"Invalid workspace name: {old_name!r}")
    if not validate_workspace_name(new_name):
        raise ValueError(f"Invalid workspace name: {new_name!r}")

    old_dir = _ws_dir(old_name, root)
    new_dir = _ws_dir(new_name, root)

    if not old_dir.exists():
        raise FileNotFoundError(f"Workspace not found: {old_name!r}")
    if new_dir.exists():
        raise FileExistsError(f"Workspace already exists: {new_name!r}")

    # Update the name field in workspace.yaml before renaming
    yaml_p = new_dir / "workspace.yaml"  # won't exist yet, but we write after rename
    old_dir.rename(new_dir)

    # Update the name field in the renamed workspace.yaml
    if yaml_p.exists():
        yaml_data = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
        yaml_data["name"] = new_name
        yaml_p.write_text(
            yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    return new_dir
