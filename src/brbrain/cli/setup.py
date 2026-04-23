"""Setup wizard — generates config.local.yaml from interactive prompts."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml


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


def setup_cmd():
    """Interactive 16-step setup wizard."""
    typer.echo("=" * 60)
    typer.echo("DrBrain Setup — Configuration Wizard")
    typer.echo("=" * 60)

    # --- LLM Primary ---
    typer.echo("\n[1/16] LLM Primary Model")
    provider = typer.prompt("  Provider (openai/anthropic/ollama)", default="openai")
    model = typer.prompt("  Model name", default="gpt-4o")
    api_key = typer.prompt("  API key (leave empty for ollama)", default="", hide_input=True)
    base_url = typer.prompt("  Base URL (empty for default)", default="", show_default=False)
    base_url = base_url if base_url else None
    llm_primary = {
        "provider": provider, "model": model,
        "api_key": api_key, "base_url": base_url,
    }

    # --- LLM Fallback ---
    typer.echo("\n[2/16] LLM Fallback Model (optional)")
    has_fallback = typer.confirm("  Add fallback model?", default=False)
    if has_fallback:
        fb_provider = typer.prompt("  Fallback provider", default="ollama")
        fb_model = typer.prompt("  Fallback model", default="qwen2.5:7b")
        fb_key = typer.prompt("  Fallback API key", default="", hide_input=True)
        fb_url = typer.prompt("  Fallback base URL", default="http://localhost:11434")
        llm_primary.setdefault("_fallback", {
            "provider": fb_provider, "model": fb_model,
            "api_key": fb_key, "base_url": fb_url if fb_url else None,
        })

    # --- MinerU Mode ---
    typer.echo("\n[4/16] MinerU Mode")
    typer.echo("  Flash: free, no token required")
    typer.echo("  Token: higher quality, requires token from https://mineru.net/apiManage/token")
    mineru_mode = typer.prompt("  Mode (flash/token)", default="flash")

    # --- MinerU Token ---
    mineru_token = ""
    if mineru_mode == "token":
        typer.echo("\n[5/16] MinerU Token")
        typer.echo("  Get your token at: https://mineru.net/apiManage/token")
        mineru_token = typer.prompt("  Token", hide_input=True)

    # --- MinerU Model ---
    typer.echo("\n[6/16] MinerU Extraction Model")
    mineru_model = typer.prompt("  Model (pipeline/vlm/MinerU-HTML)", default="vlm")

    # --- MinerU Options ---
    typer.echo("\n[7/16] MinerU Options")
    mineru_is_ocr = typer.confirm("  Enable OCR extraction?", default=False)
    mineru_enable_formula = typer.confirm("  Enable formula parsing?", default=True)
    mineru_enable_table = typer.confirm("  Enable table parsing?", default=True)

    # --- Paths ---
    typer.echo("\n[10/16] Database Path")
    db_path = typer.prompt("  DB path", default="data/drbrain.db")

    # --- API & BM25 ---
    typer.echo("\n[14/16] Semantic Scholar API")
    s2_rate_limit = typer.prompt("  Rate limit (req/min)", default=100)

    typer.echo("\n[15/16] External APIs")
    crossref_email = typer.prompt("  CrossRef email (for polite pool)", default="", show_default=False)
    openalex_token = typer.prompt("  OpenAlex token (empty for anonymous)", default="", hide_input=True)

    typer.echo("\n[16/16] BM25 Parameters")
    bm25_k1 = typer.prompt("  k1 (term frequency saturation)", default=1.5)
    bm25_b = typer.prompt("  b (document length normalization)", default=0.75)

    typer.echo("\n[16/16] Review")
    typer.echo(f"  LLM: {provider}/{model}")
    typer.echo(f"  MinerU: {mineru_mode} mode, {mineru_model}")
    typer.echo(f"  DB: {db_path}")
    if typer.confirm("  Write config.local.yaml?", default=True):
        path = generate_local_config(
            output_path="config.local.yaml",
            llm_primary=llm_primary,
            mineru_mode=mineru_mode,
            mineru_token=mineru_token,
            mineru_model=mineru_model,
            mineru_is_ocr=mineru_is_ocr,
            mineru_enable_formula=mineru_enable_formula,
            mineru_enable_table=mineru_enable_table,
            db_path=db_path,
            s2_rate_limit=s2_rate_limit,
            crossref_email=crossref_email,
            openalex_token=openalex_token,
            bm25_k1=float(bm25_k1),
            bm25_b=float(bm25_b),
        )
        typer.echo(f"\nConfig written to {path}")
    else:
        typer.echo("Setup cancelled, no file written.")
