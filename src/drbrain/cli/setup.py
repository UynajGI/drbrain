"""Setup wizard — generates config.local.yaml, initializes data directories, validates environment."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml


def _check_python_package(module: str) -> bool:
    """Check if a Python module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _check_external_tool(name: str, test_cmd: str) -> tuple[bool, str | None]:
    """Check if an external tool is on PATH. Returns (found, path)."""
    path = shutil.which(test_cmd)
    return (path is not None, path)


def generate_local_config(
    output_path: str | Path,
    llm_primary: dict,
    mineru_mode: str = "flash",
    mineru_token: str = "",
    mineru_model: str = "vlm",
    mineru_is_ocr: bool = False,
    mineru_enable_formula: bool = True,
    mineru_enable_table: bool = True,
    db_path: str = "data/drbrain.db",
    s2_rate_limit: int = 100,
    crossref_email: str = "",
    openalex_token: str = "",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> Path:
    """Write config.local.yaml with user choices."""
    config: dict = {
        "llm": {
            "models": [llm_primary],
        },
        "mineru": {
            "token": mineru_token if mineru_mode == "token" else "",
            "model": mineru_model,
            "is_ocr": mineru_is_ocr,
            "enable_formula": mineru_enable_formula,
            "enable_table": mineru_enable_table,
        },
        "db": {"path": db_path},
        "api": {
            "s2_rate_limit": s2_rate_limit,
        },
        "bm25": {"k1": bm25_k1, "b": bm25_b},
    }
    if crossref_email:
        config["api"]["crossref_email"] = crossref_email
    if openalex_token:
        config["api"]["openalex_token"] = openalex_token

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return out


def _ensure_directories(cfg: dict) -> int:
    """Create all required data directories. Returns count of newly created dirs."""
    dirs_config = cfg.get("dirs", {})
    dir_paths: list[str] = []
    if dirs_config:
        dir_paths = list(dirs_config.values())
    else:
        dir_paths = [
            "data/spool/inbox",
            "data/spool/pending",
            "data/papers",
            "data/reports",
            "data/cache",
            "data/logs",
        ]

    created = 0
    for d in dir_paths:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created += 1
    return created


def _brief_validation(cfg: dict) -> tuple[list[str], list[str]]:
    """Quick environment validation. Returns (ok_list, warn_list)."""
    ok: list[str] = []
    warn: list[str] = []

    # Python deps
    deps = {
        "pymupdf": "PyMuPDF",
        "litellm": "LiteLLM",
        "typer": "Typer",
        "rich": "Rich",
        "yaml": "PyYAML",
        "pydantic": "Pydantic",
        "streamlit": "Streamlit",
    }
    missing_deps = [display for mod, display in deps.items() if not _check_python_package(mod)]
    if missing_deps:
        warn.append(f"Missing packages: {', '.join(missing_deps)}")
    else:
        ok.append("Python packages: all present")

    # External tools
    mineru_found, _ = _check_external_tool("mineru", "mineru-open-api")
    if mineru_found:
        ok.append("MinerU CLI: found")
    else:
        warn.append("MinerU CLI not found — PDF parsing will use PyMuPDF fallback")

    # Config
    if Path("config.yaml").exists():
        ok.append("config.yaml: found")
    else:
        warn.append("config.yaml not found")

    if Path("config.local.yaml").exists():
        ok.append("config.local.yaml: found")
    else:
        warn.append("config.local.yaml not found — run `drbrain setup` first")

    # Directories from config
    dirs_config = cfg.get("dirs", {})
    dir_paths = (
        list(dirs_config.values())
        if dirs_config
        else [
            "data/spool/inbox",
            "data/spool/pending",
            "data/papers",
            "data/reports",
            "data/cache",
            "data/logs",
        ]
    )
    missing_dirs = [d for d in dir_paths if not Path(d).exists()]
    if missing_dirs:
        warn.append(f"Missing directories: {', '.join(missing_dirs)}")
    else:
        ok.append("Data directories: all present")

    return ok, warn


def setup_cmd(
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Skip interactive prompts, use defaults"
    ),
):
    """Initialize DrBrain — generate config, create directories, validate environment."""
    # If config.local.yaml already exists, offer to re-run or validate only
    if Path("config.local.yaml").exists() and not quick:
        typer.echo("config.local.yaml already exists.\n")
        choice = typer.prompt(
            "  [r]e-run setup  [v]alidate environment only  [q]uit",
            default="v",
        )
        if choice == "q":
            typer.echo("Cancelled.")
            return
        if choice == "v":
            from drbrain.config import load_config

            cfg = load_config()
            _ensure_directories(cfg)
            ok, warn = _brief_validation(cfg)
            typer.echo()
            for line in ok:
                typer.echo(f"  [OK] {line}")
            for line in warn:
                typer.echo(f"  [!]  {line}")
            typer.echo()
            if not warn:
                typer.echo("Environment ready. Next: drbrain ingest")
            return

    # ── Quick mode: skip prompts, use defaults ──
    if quick:
        config = {
            "llm": {
                "models": [
                    {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "${OPENAI_API_KEY}",
                        "base_url": None,
                    }
                ],
            },
            "mineru": {"token": ""},
            "api": {"deepxiv_token": "", "s2_api_key": "", "crossref_email": "", "openalex_token": ""},
        }
        out = Path("config.local.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        typer.echo(f"Config written to {out} (defaults)")

        from drbrain.config import load_config

        cfg = load_config()
        created = _ensure_directories(cfg)
        if created:
            typer.echo(f"Created {created} director{'y' if created == 1 else 'ies'}")
        ok, warn = _brief_validation(cfg)
        for line in ok:
            typer.echo(f"  [OK] {line}")
        for line in warn:
            typer.echo(f"  [!]  {line}")
        typer.echo()
        typer.echo("Ready. Next step: drbrain ingest")
        typer.echo("Edit config.local.yaml to set your API keys and model.")
        return

    typer.echo("=" * 60)
    typer.echo("DrBrain Setup")
    typer.echo("=" * 60)
    typer.echo("  Settings in config.yaml (db, dirs, bm25, queue) are already configured.")
    typer.echo("  This wizard only writes secrets to config.local.yaml.")
    typer.echo()

    # ── LLM ──
    typer.echo("── LLM ──")
    provider = typer.prompt("  Provider (openai/anthropic/ollama)", default="openai")
    model = typer.prompt("  Model", default="gpt-4o")
    api_key = typer.prompt("  API key (leave empty for ollama)", default="", hide_input=True)
    base_url = typer.prompt("  Base URL (empty for default)", default="", show_default=False)
    base_url = base_url if base_url else None
    models: list[dict] = [
        {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
    ]

    has_fallback = typer.confirm("  Add fallback model?", default=False)
    if has_fallback:
        fb_provider = typer.prompt("    Fallback provider", default="ollama")
        fb_model = typer.prompt("    Fallback model", default="qwen2.5:7b")
        fb_key = typer.prompt("    Fallback API key", default="", hide_input=True)
        fb_url = typer.prompt("    Fallback base URL", default="http://localhost:11434")
        models.append(
            {
                "provider": fb_provider,
                "model": fb_model,
                "api_key": fb_key,
                "base_url": fb_url if fb_url else None,
            }
        )

    # ── MinerU ──
    typer.echo()
    typer.echo("── MinerU (PDF Parser) ──")
    typer.echo("  Settings in config.yaml. Only the token is secret.")
    mineru_mode = typer.prompt("  Mode (flash/token)", default="flash")
    mineru_token = ""
    if mineru_mode == "token":
        typer.echo("  Get token: https://mineru.net/apiManage/token")
        mineru_token = typer.prompt("  Token", hide_input=True)

    # ── API Keys ──
    typer.echo()
    typer.echo("── API Keys ──")
    deepxiv_token = typer.prompt(
        "  DeepXiv token (https://data.rag.ac.cn/register)", default="", hide_input=True
    )
    s2_api_key = typer.prompt(
        "  Semantic Scholar API key (https://semanticscholar.org/product/api)", default="", hide_input=True
    )
    crossref_email = typer.prompt("  CrossRef email (optional)", default="", show_default=False)
    openalex_token = typer.prompt("  OpenAlex token (optional)", default="", hide_input=True)

    # ── Review ──
    typer.echo()
    typer.echo("── Review ──")
    typer.echo(
        f"  LLM:      {provider}/{model}" + (f" + {fb_provider}/{fb_model}" if has_fallback else "")
    )
    typer.echo(f"  MinerU:   {mineru_mode}" + (" (token set)" if mineru_token else " (free tier)"))
    typer.echo(
        f"  APIs:     DeepXiv({'key set' if deepxiv_token else 'not set'})  "
        f"S2({'key set' if s2_api_key else 'anonymous'})  "
        f"CrossRef({crossref_email or 'none'})  "
        f"OpenAlex({'key set' if openalex_token else 'anonymous'})"
    )

    if not typer.confirm("\n  Write config.local.yaml?", default=True):
        typer.echo("Cancelled.")
        return

    # ── Write config.local.yaml (secrets only) ──
    api_cfg: dict = {}
    if deepxiv_token:
        api_cfg["deepxiv_token"] = deepxiv_token
    if s2_api_key:
        api_cfg["s2_api_key"] = s2_api_key
    if crossref_email:
        api_cfg["crossref_email"] = crossref_email
    if openalex_token:
        api_cfg["openalex_token"] = openalex_token

    config = {
        "llm": {"models": models},
        "mineru": {"token": mineru_token},
    }
    if api_cfg:
        config["api"] = api_cfg

    out = Path("config.local.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    typer.echo(f"\n  Config written to {out}")

    # ── Step 5: Initialize environment ──
    typer.echo("\n── Environment ──")
    from drbrain.config import load_config

    cfg = load_config()
    created = _ensure_directories(cfg)
    if created:
        typer.echo(f"  Created {created} director{'y' if created == 1 else 'ies'}")
    else:
        typer.echo("  Directories: all present")

    ok, warn = _brief_validation(cfg)
    for line in ok:
        typer.echo(f"  [OK] {line}")
    for line in warn:
        typer.echo(f"  [!]  {line}")

    typer.echo()
    if not warn:
        typer.echo("Ready. Next step: drbrain ingest")
    else:
        typer.echo(f"Setup complete with {len(warn)} warning(s).")
        typer.echo("Run `drbrain check` for detailed diagnostics.")
