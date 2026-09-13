"""`drbrain webui` — serve the local research workbench."""

from __future__ import annotations

import typer


def webui_cmd(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8765, "--port", "-p", help="Port"),
    open_browser: bool = typer.Option(False, "--open", help="Open the page in the default browser"),
    show_token: bool = typer.Option(False, "--show-token", help="Print the access token and exit"),
) -> None:
    """Start the WebUI: projects / literature / sessions / runs / plugins / settings.

    A local access token is generated on first start, printed here and stored
    at config/webui_token (0600). Stop with Ctrl-C.
    """
    cfg = ctx.obj["config"]
    from drbrain.app import auth

    token = auth.ensure_token(cfg)
    if show_token:
        typer.echo(token)
        return

    from drbrain.app.web import create_app

    app = create_app(cfg)
    url = f"http://{host}:{port}/"
    typer.echo(f"DrBrain WebUI → {url}")
    typer.echo(f"访问令牌: {token}")
    typer.echo("（同样保存于 config/webui_token，0600；登录后可在设置页重置）")
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo(
            "警告：正在监听非本机地址。当前版本是本地单人模式，"
            "请通过 HTTPS 反向代理终止 TLS 后再暴露。",
            err=True,
        )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    import uvicorn

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        typer.echo("\nWebUI stopped")
