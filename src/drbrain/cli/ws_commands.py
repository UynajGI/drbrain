"""Workspace management commands."""

from __future__ import annotations

import json

import typer

ws_app = typer.Typer(help="Manage paper workspaces")


@ws_app.command("create")
def ws_create_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    description: str = typer.Option("", "--description", "-d", help="Description"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Create a new workspace."""
    from drbrain.storage.workspace import WorkspaceError, create_workspace

    try:
        create_workspace(name, description=description)
        if json_output:
            typer.echo(json.dumps({"created": name, "description": description}))
        else:
            typer.echo(f"Workspace created: {name}")
    except WorkspaceError as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


@ws_app.command("add")
def ws_add_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    local_ids: list[str] = typer.Argument(..., help="Paper local_id(s) to add"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Add papers to a workspace."""
    from drbrain.storage.workspace import WorkspaceError, add_papers, get_workspace

    try:
        add_papers(name, local_ids)
        ws = get_workspace(name)
        if json_output:
            typer.echo(json.dumps(ws, indent=2))
        else:
            typer.echo(f"Added {len(local_ids)} paper(s) to '{name}' ({ws['paper_count']} total)")
    except WorkspaceError as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


@ws_app.command("remove")
def ws_remove_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    local_ids: list[str] = typer.Argument(..., help="Paper local_id(s) to remove"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Remove papers from a workspace."""
    from drbrain.storage.workspace import get_workspace, remove_papers

    remove_papers(name, local_ids)
    ws = get_workspace(name)
    if json_output:
        typer.echo(json.dumps(ws, indent=2))
    else:
        typer.echo(f"Removed {len(local_ids)} paper(s) from '{name}' ({ws['paper_count']} total)")


@ws_app.command("list")
def ws_list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """List all workspaces."""
    from drbrain.storage.workspace import list_workspaces

    names = list_workspaces()
    if json_output:
        typer.echo(json.dumps({"workspaces": names}))
    elif not names:
        typer.echo("No workspaces. Create one with: drbrain ws create <name>")
    else:
        typer.echo(f"Workspaces ({len(names)}):")
        for n in names:
            typer.echo(f"  {n}")


@ws_app.command("show")
def ws_show_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show workspace details and paper list."""
    from drbrain.storage.workspace import get_workspace

    ws = get_workspace(name)
    if ws is None:
        msg = f"Workspace not found: {name}"
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(ws, indent=2, default=str))
        return

    typer.echo(f"Workspace: {ws['name']}")
    typer.echo(f"  Description: {ws['description']}")
    typer.echo(f"  Created: {ws['created']}")
    typer.echo(f"  Papers: {ws['paper_count']}")
    for pid in ws["papers"]:
        typer.echo(f"    - {pid}")


@ws_app.command("delete")
def ws_delete_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Delete a workspace."""
    from drbrain.storage.workspace import delete_workspace, get_workspace

    ws = get_workspace(name)
    if ws is None:
        msg = f"Workspace not found: {name}"
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    delete_workspace(name)
    if json_output:
        typer.echo(json.dumps({"deleted": name}))
    else:
        typer.echo(f"Workspace deleted: {name}")


@ws_app.command("rename")
def ws_rename_cmd(
    old_name: str = typer.Argument(..., help="Current workspace name"),
    new_name: str = typer.Argument(..., help="New workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Rename a workspace."""
    from drbrain.storage.workspace import rename_workspace

    try:
        new_path = rename_workspace(old_name, new_name)
        if json_output:
            typer.echo(json.dumps({"renamed": old_name, "to": new_name, "path": str(new_path)}))
        else:
            typer.echo(f"Workspace renamed: {old_name} -> {new_name}")
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


# -- repair + import commands --
