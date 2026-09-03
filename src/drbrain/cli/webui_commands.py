"""`drbrain webui` — serve the local research workbench."""

from __future__ import annotations

import typer


def webui_cmd(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8765, "--port", "-p", help="Port"),
    open_browser: bool = typer.Option(False, "--open", help="Open the page in the default browser"),
) -> None:
    """Start the WebUI: search / ask / autoresearch loop / compute jobs / assets.

    The UI wraps the same database, ledger and plugins the CLI uses; nothing is
    preloaded. Stop with Ctrl-C.
    """
    cfg = ctx.obj["config"]
    from drbrain.webui.server import serve

    url = f"http://{host}:{port}/"
    typer.echo(f"DrBrain WebUI → {url}  (Ctrl-C to stop)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        serve(cfg, host=host, port=port)
    except KeyboardInterrupt:
        typer.echo("\nWebUI stopped")
