"""Tests for workspace CRUD operations."""

import json

from drbrain.storage.workspace import (
    WorkspaceError,
    add_papers,
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
    load_workspace_papers,
    remove_papers,
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
