"""Tests for drbrain.cli.ws_commands — workspace CLI subcommands.

All workspace storage functions accept an optional ``root`` param. The CLI
calls them without ``root`` (default ``Path("workspace")``), so we patch the
storage module's default root via a context-manager that wraps each function
to inject a temp root. This avoids touching the real on-disk workspace dir
and avoids network/DB access.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from drbrain.cli.ws_commands import ws_app
from drbrain.storage import workspace as ws_mod

runner = CliRunner()


@pytest.fixture
def ws_root(tmp_path, monkeypatch):
    """Redirect all workspace storage calls to a temp root directory."""
    root = tmp_path / "workspaces"
    root.mkdir()

    real_funcs = {
        "create_workspace": ws_mod.create_workspace,
        "add_papers": ws_mod.add_papers,
        "remove_papers": ws_mod.remove_papers,
        "list_workspaces": ws_mod.list_workspaces,
        "get_workspace": ws_mod.get_workspace,
        "delete_workspace": ws_mod.delete_workspace,
        "rename_workspace": ws_mod.rename_workspace,
    }
    # Wrap each so root= is always injected
    for name, fn in real_funcs.items():

        def _wrap(f=fn):
            def wrapper(*args, **kwargs):
                kwargs.setdefault("root", root)
                return f(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(ws_mod, name, _wrap())
    return root


# -- create --


def test_ws_create_basic(ws_root):
    result = runner.invoke(ws_app, ["create", "alpha"])
    assert result.exit_code == 0, result.output
    assert "Workspace created: alpha" in result.output
    assert (ws_root / "alpha" / "workspace.yaml").exists()


def test_ws_create_with_description(ws_root):
    result = runner.invoke(ws_app, ["create", "beta", "--description", "A test ws"])
    assert result.exit_code == 0
    assert (ws_root / "beta" / "workspace.yaml").exists()


def test_ws_create_json(ws_root):
    result = runner.invoke(ws_app, ["create", "gamma", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data == {"created": "gamma", "description": ""}


def test_ws_create_duplicate_fails(ws_root):
    runner.invoke(ws_app, ["create", "dup"])
    result = runner.invoke(ws_app, ["create", "dup"])
    assert result.exit_code == 1
    assert "already exists" in result.output


# -- list --


def test_ws_list_empty(ws_root):
    result = runner.invoke(ws_app, ["list"])
    assert result.exit_code == 0
    assert "No workspaces" in result.output


def test_ws_list_with_entries(ws_root):
    runner.invoke(ws_app, ["create", "one"])
    runner.invoke(ws_app, ["create", "two"])
    result = runner.invoke(ws_app, ["list"])
    assert result.exit_code == 0
    assert "Workspaces (2)" in result.output
    assert "one" in result.output
    assert "two" in result.output


def test_ws_list_json(ws_root):
    runner.invoke(ws_app, ["create", "x"])
    result = runner.invoke(ws_app, ["list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data == {"workspaces": ["x"]}


# -- add / remove --


def test_ws_add_papers(ws_root):
    runner.invoke(ws_app, ["create", "ws1"])
    result = runner.invoke(ws_app, ["add", "ws1", "p001", "p002"])
    assert result.exit_code == 0, result.output
    assert "Added 2 paper(s)" in result.output
    assert "2 total" in result.output


def test_ws_add_json(ws_root):
    runner.invoke(ws_app, ["create", "wsj"])
    result = runner.invoke(ws_app, ["add", "wsj", "p1", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["name"] == "wsj"
    assert data["paper_count"] == 1
    assert data["papers"] == ["p1"]


def test_ws_add_to_missing_fails(ws_root):
    result = runner.invoke(ws_app, ["add", "nope", "p1"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_ws_add_duplicate_ignored(ws_root):
    runner.invoke(ws_app, ["create", "wd"])
    runner.invoke(ws_app, ["add", "wd", "p1"])
    result = runner.invoke(ws_app, ["add", "wd", "p1"])
    assert result.exit_code == 0
    assert "1 total" in result.output  # still 1, dedup'd


def test_ws_remove_papers(ws_root):
    runner.invoke(ws_app, ["create", "wr"])
    runner.invoke(ws_app, ["add", "wr", "a", "b"])
    result = runner.invoke(ws_app, ["remove", "wr", "a"])
    assert result.exit_code == 0
    assert "Removed 1 paper(s)" in result.output
    assert "1 total" in result.output


def test_ws_remove_json(ws_root):
    runner.invoke(ws_app, ["create", "wrj"])
    runner.invoke(ws_app, ["add", "wrj", "x", "y"])
    result = runner.invoke(ws_app, ["remove", "wrj", "x", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["paper_count"] == 1


# -- show --


def test_ws_show_basic(ws_root):
    runner.invoke(ws_app, ["create", "show1", "-d", "desc here"])
    runner.invoke(ws_app, ["add", "show1", "p9"])
    result = runner.invoke(ws_app, ["show", "show1"])
    assert result.exit_code == 0
    assert "Workspace: show1" in result.output
    assert "desc here" in result.output
    assert "Papers: 1" in result.output
    assert "- p9" in result.output


def test_ws_show_json(ws_root):
    runner.invoke(ws_app, ["create", "sj"])
    result = runner.invoke(ws_app, ["show", "sj", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["name"] == "sj"
    assert data["paper_count"] == 0


def test_ws_show_missing_fails(ws_root):
    result = runner.invoke(ws_app, ["show", "ghost"])
    assert result.exit_code == 1
    assert "not found" in result.output


# -- delete --


def test_ws_delete(ws_root):
    runner.invoke(ws_app, ["create", "todelete"])
    result = runner.invoke(ws_app, ["delete", "todelete"])
    assert result.exit_code == 0
    assert "Workspace deleted" in result.output
    assert not (ws_root / "todelete").exists()


def test_ws_delete_json(ws_root):
    runner.invoke(ws_app, ["create", "dj"])
    result = runner.invoke(ws_app, ["delete", "dj", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data == {"deleted": "dj"}


def test_ws_delete_missing_fails(ws_root):
    result = runner.invoke(ws_app, ["delete", "missing"])
    assert result.exit_code == 1
    assert "not found" in result.output


# -- rename --


def test_ws_rename(ws_root):
    runner.invoke(ws_app, ["create", "oldname"])
    result = runner.invoke(ws_app, ["rename", "oldname", "newname"])
    assert result.exit_code == 0
    assert "renamed: oldname -> newname" in result.output
    assert (ws_root / "newname" / "workspace.yaml").exists()


def test_ws_rename_json(ws_root):
    runner.invoke(ws_app, ["create", "a1"])
    result = runner.invoke(ws_app, ["rename", "a1", "a2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["renamed"] == "a1"
    assert data["to"] == "a2"
