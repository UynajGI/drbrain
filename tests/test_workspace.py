"""Tests for workspace CRUD operations."""

import json

import pytest
import yaml

from drbrain.storage.workspace import (
    WorkspaceError,
    add_papers,
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
    load_workspace_papers,
    remove_papers,
    rename_workspace,
    validate_workspace_name,
)


def test_create_workspace(tmp_path):
    """create_workspace creates directory, yaml, and papers.json."""
    ws_dir = tmp_path / "test-proj"
    create_workspace("test-proj", root=tmp_path, description="A test workspace")

    assert ws_dir.exists()
    assert (ws_dir / "workspace.yaml").exists()
    assert (ws_dir / "refs" / "papers.json").exists()

    yaml_content = (ws_dir / "workspace.yaml").read_text()
    assert "name: test-proj" in yaml_content


def test_create_workspace_duplicate(tmp_path):
    """Creating duplicate workspace raises WorkspaceError."""
    create_workspace("dup", root=tmp_path)
    try:
        create_workspace("dup", root=tmp_path)
        assert False, "expected WorkspaceError"
    except WorkspaceError as e:
        assert "dup" in str(e)


def test_add_papers(tmp_path):
    """add_papers appends local_ids to papers.json."""
    ws_dir = tmp_path / "ws"
    create_workspace("ws", root=tmp_path)
    add_papers("ws", ["p1a2b3c4", "p5d6e7f8"], root=tmp_path)

    papers = json.loads((ws_dir / "refs" / "papers.json").read_text())
    ids = {p["local_id"] for p in papers}
    assert "p1a2b3c4" in ids
    assert "p5d6e7f8" in ids

    # No duplicates
    add_papers("ws", ["p1a2b3c4"], root=tmp_path)
    papers = json.loads((ws_dir / "refs" / "papers.json").read_text())
    ids_list = [p["local_id"] for p in papers]
    assert ids_list.count("p1a2b3c4") == 1


def test_add_papers_nonexistent_workspace(tmp_path):
    """Adding to non-existent workspace raises WorkspaceError."""
    try:
        add_papers("nope", ["p1"], root=tmp_path)
        assert False, "expected WorkspaceError"
    except WorkspaceError:
        pass


def test_remove_papers(tmp_path):
    """remove_papers removes entries by local_id."""
    ws_dir = tmp_path / "ws"
    create_workspace("ws", root=tmp_path)
    add_papers("ws", ["p1", "p2", "p3"], root=tmp_path)
    remove_papers("ws", ["p2"], root=tmp_path)

    papers = json.loads((ws_dir / "refs" / "papers.json").read_text())
    ids = {p["local_id"] for p in papers}
    assert ids == {"p1", "p3"}


def test_list_workspaces(tmp_path):
    """list_workspaces returns sorted names."""
    create_workspace("alpha", root=tmp_path)
    create_workspace("beta", root=tmp_path)
    names = list_workspaces(root=tmp_path)
    assert names == ["alpha", "beta"]


def test_list_workspaces_empty(tmp_path):
    """Empty workspace root returns empty list."""
    assert list_workspaces(root=tmp_path) == []


def test_get_workspace(tmp_path):
    """get_workspace returns name, description, paper IDs."""
    create_workspace("ws", root=tmp_path, description="desc")
    add_papers("ws", ["p1", "p2"], root=tmp_path)
    ws = get_workspace("ws", root=tmp_path)
    assert ws["name"] == "ws"
    assert ws["description"] == "desc"
    assert ws["paper_count"] == 2
    assert ws["papers"] == ["p1", "p2"]


def test_delete_workspace(tmp_path):
    """delete_workspace removes the workspace directory."""
    create_workspace("ws", root=tmp_path)
    delete_workspace("ws", root=tmp_path)
    assert not (tmp_path / "ws").exists()


def test_load_workspace_papers(tmp_path):
    """load_workspace_papers returns list of local_ids."""
    create_workspace("ws", root=tmp_path)
    add_papers("ws", ["p1", "p2"], root=tmp_path)
    ids = load_workspace_papers("ws", root=tmp_path)
    assert ids == ["p1", "p2"]


def test_load_workspace_papers_nonexistent(tmp_path):
    """Non-existent workspace returns empty list."""
    assert load_workspace_papers("nope", root=tmp_path) == []


# ---------------------------------------------------------------------------
# validate_workspace_name
# ---------------------------------------------------------------------------


def test_validate_name_rejects_empty():
    """Empty string is invalid."""
    assert validate_workspace_name("") is False


def test_validate_name_rejects_dot():
    """'.' alone is invalid."""
    assert validate_workspace_name(".") is False


def test_validate_name_rejects_dot_dot():
    """'..' alone is invalid."""
    assert validate_workspace_name("..") is False


def test_validate_name_rejects_absolute_path():
    """Absolute paths are rejected."""
    assert validate_workspace_name("/etc/passwd") is False


def test_validate_name_rejects_path_separator():
    """Names containing '/' or '\\' are rejected."""
    assert validate_workspace_name("foo/bar") is False
    assert validate_workspace_name("foo\\bar") is False


def test_validate_name_rejects_colon():
    """Names containing ':' (Windows drive) are rejected."""
    assert validate_workspace_name("C:") is False
    assert validate_workspace_name("foo:bar") is False


def test_validate_name_rejects_leading_whitespace():
    """Leading whitespace is rejected."""
    assert validate_workspace_name("  foo") is False


def test_validate_name_rejects_trailing_whitespace():
    """Trailing whitespace is rejected."""
    assert validate_workspace_name("foo  ") is False


def test_validate_name_rejects_dot_dot_anywhere():
    """'..' anywhere in the name is rejected."""
    assert validate_workspace_name("foo..bar") is False
    assert validate_workspace_name("..foo") is False
    assert validate_workspace_name("foo..") is False


def test_validate_name_accepts_valid():
    """Valid names pass validation."""
    assert validate_workspace_name("my-project") is True
    assert validate_workspace_name("paper_review_2024") is True
    assert validate_workspace_name("deep.learning.survey") is True
    assert validate_workspace_name("a") is True
    assert validate_workspace_name("test 123") is True


# ---------------------------------------------------------------------------
# schema_version
# ---------------------------------------------------------------------------


def test_create_has_schema_version(tmp_path):
    """create_workspace writes schema_version: 1 in workspace.yaml."""
    create_workspace("ws", root=tmp_path)
    yaml_data = yaml.safe_load((tmp_path / "ws" / "workspace.yaml").read_text(encoding="utf-8"))
    assert yaml_data["schema_version"] == 1


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_uses_tmp(tmp_path):
    """_write_papers writes atomically — no .tmp file left behind."""
    create_workspace("ws", root=tmp_path)

    papers = [{"local_id": "x", "added_at": "2024-01-01T00:00:00Z"}]
    from drbrain.storage import workspace as ws_mod

    ws_mod._write_papers("ws", papers, root=tmp_path)

    papers_json = tmp_path / "ws" / "refs" / "papers.json"
    tmp_json = tmp_path / "ws" / "refs" / "papers.json.tmp"

    # Data is written correctly
    data = json.loads(papers_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["local_id"] == "x"

    # No stale .tmp file
    assert not tmp_json.exists()


# ---------------------------------------------------------------------------
# rename_workspace
# ---------------------------------------------------------------------------


def test_rename_workspace_success(tmp_path):
    """rename_workspace renames the directory and returns new path."""
    create_workspace("old-name", root=tmp_path, description="test")
    add_papers("old-name", ["p1", "p2"], root=tmp_path)

    new_path = rename_workspace("old-name", "new-name", root=tmp_path)

    assert new_path == tmp_path / "new-name"
    assert not (tmp_path / "old-name").exists()
    assert (tmp_path / "new-name" / "workspace.yaml").exists()
    assert (tmp_path / "new-name" / "refs" / "papers.json").exists()

    # Content preserved
    ws = get_workspace("new-name", root=tmp_path)
    assert ws["name"] == "new-name"
    assert ws["description"] == "test"
    assert ws["papers"] == ["p1", "p2"]


def test_rename_source_not_found(tmp_path):
    """Raising non-existent workspace raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="old-name"):
        rename_workspace("old-name", "new-name", root=tmp_path)


def test_rename_target_exists(tmp_path):
    """Raising into existing workspace raises FileExistsError."""
    create_workspace("old-name", root=tmp_path)
    create_workspace("new-name", root=tmp_path)

    with pytest.raises(FileExistsError, match="new-name"):
        rename_workspace("old-name", "new-name", root=tmp_path)


def test_rename_validates_names(tmp_path):
    """rename_workspace validates both old and new names."""
    create_workspace("valid", root=tmp_path)
    # Bad old name
    with pytest.raises(ValueError):
        rename_workspace("..", "valid2", root=tmp_path)
    # Bad new name
    with pytest.raises(ValueError):
        rename_workspace("valid", "../evil", root=tmp_path)
