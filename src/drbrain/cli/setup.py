"""Setup wizard — generates config.local.yaml, initializes data directories, validates environment."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml
from loguru import logger

from drbrain.cli._setup_i18n import t as _t

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "agents"
_SKILLS_SRC = Path(__file__).parent.parent / "skills"
_PROJECT_SKILLS = Path("skills")

# Agent-specific skill directories (symlinked to skills/ when possible)
SKILL_DIR_MAP: dict[str, str] = {
    "claude_code": ".claude/skills",
    "codex": ".codex/skills",
    "qwen": ".qwen/skills",
    "cursor": ".cursor/skills",
    "cline": ".claude/skills",  # Cline reuses Claude Code conventions
    "windsurf": ".windsurf/skills",
    "copilot": ".github/skills",
}

INJECTION_MAP: dict[str, list[tuple[str, str]]] = {
    "claude_code": [
        ("CLAUDE.md.j2", "CLAUDE.md"),
    ],
    "claude_plugin": [
        ("plugin.json.j2", ".claude-plugin/plugin.json"),
        ("marketplace.json.j2", ".claude-plugin/marketplace.json"),
        ("mcp.json.j2", ".mcp.json"),
    ],
    "codex": [("AGENTS.md.j2", "AGENTS.md")],
    "qwen": [("QWEN.md.j2", ".qwen/QWEN.md")],
    "cursor": [
        ("drbrain.mdc.j2", ".cursor/rules/drbrain.mdc"),
        (".cursorrules.j2", ".cursorrules"),
    ],
    "cline": [(".clinerules.j2", ".clinerules")],
    "windsurf": [(".windsurfrules.j2", ".windsurfrules")],
    "copilot": [("copilot-instructions.md.j2", ".github/copilot-instructions.md")],
}


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
    embed_provider: str = "local",
    embed_model: str = "",
    embed_api_base: str = "",
    embed_api_key: str = "",
    embed_device: str = "auto",
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

    embed_cfg: dict = {"provider": embed_provider, "device": embed_device}
    if embed_provider == "openai-compat":
        if embed_model:
            embed_cfg["model"] = embed_model
        if embed_api_base:
            embed_cfg["api_base"] = embed_api_base
        if embed_api_key:
            embed_cfg["api_key"] = embed_api_key
    elif embed_provider == "local" and embed_model:
        embed_cfg["model"] = embed_model
    config["embed"] = embed_cfg

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

    # Python deps — use shared install hints for consistency
    from drbrain.cli.dependencies import _INSTALL_HINTS

    # Map Python module names to display names from install hints
    _module_to_hint_key: dict[str, str] = {
        "pymupdf": "pymupdf",
        "litellm": "litellm",
        "typer": "typer",
        "rich": "rich",
        "yaml": "pyyaml",
        "pydantic": "pydantic",
        "pyalex": "pyalex",
        "arxiv": "arxiv",
        "pymupdf4llm": "pymupdf4llm",
    }
    deps = {
        mod: _INSTALL_HINTS[hint_key].replace("pip install ", "")
        for mod, hint_key in _module_to_hint_key.items()
        if hint_key in _INSTALL_HINTS
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


def _detect_platforms() -> dict[str, bool]:
    """Detect installed AI platforms. Returns {platform_name: detected}."""
    home = Path.home()
    detected: dict[str, bool] = {}

    # Claude Code
    detected["claude_code"] = shutil.which("claude") is not None or (home / ".claude").exists()

    # Codex / OpenClaw
    detected["codex"] = shutil.which("codex") is not None or (home / ".codex").exists()

    # Qwen
    detected["qwen"] = (
        (home / ".qwen").exists()
        or (home / ".config" / "Qwen").exists()
        or (Path(".") / ".qwen").exists()
        or shutil.which("qwen") is not None
    )

    # Cursor
    detected["cursor"] = (
        (Path(".") / ".cursor").exists()
        or (home / ".cursor").exists()
        or shutil.which("cursor") is not None
    )

    # Cline
    detected["cline"] = (
        (home / ".cline").exists()
        or (Path(".") / ".clinerules").exists()
        or shutil.which("cline") is not None
    )

    # Windsurf
    detected["windsurf"] = (
        (home / ".windsurf").exists()
        or (Path(".") / ".windsurfrules").exists()
        or shutil.which("windsurf") is not None
    )

    # GitHub Copilot
    detected["copilot"] = (Path(".") / ".github").exists()

    return detected


def _inject_agent_entries(lang: str = "en") -> None:
    """Interactive agent entry injection. User must select at least one platform."""
    detected = _detect_platforms()
    available = {k for k, v in detected.items() if v}

    if not available:
        typer.echo(f"\n{_t('agent_none_detected', lang)}")
        return

    # Build display: Claude Code entry includes claude_plugin group
    _labels: dict[str, str] = {
        "claude_code": "Claude Code (includes .claude-plugin + .mcp.json)",
        "codex": "Codex / OpenClaw",
        "qwen": "Qwen",
        "cursor": "Cursor",
        "cline": "Cline",
        "windsurf": "Windsurf",
        "copilot": "GitHub Copilot",
    }

    platforms = sorted(available, key=lambda p: list(_labels).index(p) if p in _labels else 99)

    typer.echo(f"\n{_t('agent_detected_title', lang)}")
    for i, p in enumerate(platforms, 1):
        typer.echo(f"  [{i}] {_labels.get(p, p)}")
    typer.echo(f"  [a] {_t('agent_all', lang)}")
    typer.echo(f"  [n] {_t('agent_none', lang)}")

    choice_str = typer.prompt(_t("agent_select", lang), default="a").strip()

    if choice_str.lower() == "n":
        typer.echo(_t("agent_skipped", lang))
        return

    selected: set[str] = set()
    if choice_str.lower() == "a":
        selected = set(platforms)
    else:
        for part in choice_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part) - 1
                if 0 <= idx < len(platforms):
                    selected.add(platforms[idx])
            except ValueError:
                continue

    if not selected:
        typer.echo(_t("agent_no_selection", lang))
        return

    # Inject templates for selected platforms
    for platform in sorted(selected):
        entries = INJECTION_MAP.get(platform, [])
        # Claude Code also includes claude_plugin entries
        if platform == "claude_code" and "claude_code" in selected:
            entries = list(entries) + INJECTION_MAP.get("claude_plugin", [])

        for template_name, target_name in entries:
            template_path = _TEMPLATES_DIR / template_name
            target_path = Path(".") / target_name

            if not template_path.exists():
                typer.echo(f"  [!]  {_t('agent_template_missing', lang)}: {template_name}")
                continue

            if target_path.exists():
                typer.echo(f"  [!]  {target_name} {_t('agent_file_exists', lang)}")
                continue

            content = template_path.read_text(encoding="utf-8")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            typer.echo(f"  [+]  {target_name}")

    typer.echo()

    # Inject skills for selected platforms
    _install_skills(selected, lang=lang)


def _install_skills(selected_platforms: set[str], quiet: bool = False, lang: str = "en") -> None:
    """Copy skills to project skills/ and create agent-specific symlinks."""
    if _SKILLS_SRC.exists() and _SKILLS_SRC.is_dir():
        src = _SKILLS_SRC
    elif _PROJECT_SKILLS.exists() and _PROJECT_SKILLS.is_dir():
        src = _PROJECT_SKILLS
    else:
        if not quiet:
            typer.echo("  [!]  No skills directory found — skipping skill injection")
        return

    if not quiet:
        typer.echo("\n── Skills Injection ──")
        typer.echo(f"  Source: {src}")

    if not _PROJECT_SKILLS.exists():
        _copy_skills_dir(src, _PROJECT_SKILLS)
        if not quiet:
            typer.echo(f"  [+]  skills/ ({len(list(_PROJECT_SKILLS.iterdir()))} skills)")
    elif src.resolve() != _PROJECT_SKILLS.resolve():
        if quiet or typer.confirm("  skills/ already exists. Update from source?", default=False):
            _copy_skills_dir(src, _PROJECT_SKILLS)
            if not quiet:
                typer.echo(
                    f"  [+]  skills/ updated ({len(list(_PROJECT_SKILLS.iterdir()))} skills)"
                )
    else:
        if not quiet:
            typer.echo("  skills/ already up to date (source install)")

    for platform in sorted(selected_platforms):
        agent_dir = SKILL_DIR_MAP.get(platform)
        if not agent_dir:
            continue
        target = Path(".") / agent_dir
        if target.exists() or target.is_symlink():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Path("..") / "skills", target_is_directory=True)
            if not quiet:
                typer.echo(f"  [+]  {agent_dir} -> skills/")
        except OSError:
            _copy_skills_dir(_PROJECT_SKILLS, target)
            if not quiet:
                typer.echo(f"  [+]  {agent_dir} (copied, symlink unsupported)")

    if not quiet:
        typer.echo()


def _copy_skills_dir(src: Path, dst: Path) -> None:
    """Copy skill directories from src to dst."""
    import shutil as _shutil

    if dst.exists():
        _shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            _shutil.copytree(item, dst / item.name)


def setup_cmd(
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Skip interactive prompts, use defaults"
    ),
    change_password: bool = typer.Option(
        False, "--change-password", help="Change the admin password"
    ),
):
    """Initialize DrBrain — generate config, create directories, validate environment."""
    logger.info("[setup] starting (quick=%s)", quick)
    # ── --change-password: verify old, set new, write config, exit ──
    if change_password:
        from drbrain.auth import has_password, hash_password, verify_password
        from drbrain.config import load_config

        cfg = load_config()
        if not has_password(cfg):
            typer.echo("No admin password is currently set.", err=True)
            raise typer.Exit(1)

        old_pw = typer.prompt("Current admin password", hide_input=True)
        if not verify_password(old_pw, cfg["admin"]["password_hash"]):
            typer.echo("Wrong password.", err=True)
            raise typer.Exit(1)

        new_pw = typer.prompt("New admin password", hide_input=True)
        new_confirm = typer.prompt("Confirm new admin password", hide_input=True)
        if new_pw != new_confirm:
            typer.echo("Passwords don't match.", err=True)
            raise typer.Exit(1)

        from pathlib import Path

        import yaml

        config_path = Path("config.local.yaml")
        if config_path.exists():
            local = yaml.safe_load(config_path.read_text()) or {}
        else:
            local = {}
        local.setdefault("admin", {})["password_hash"] = hash_password(new_pw)
        config_path.write_text(yaml.dump(local, default_flow_style=False, allow_unicode=True))
        typer.echo("Admin password updated.")
        return

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

    # ── Language selection (interactive only) ──
    lang = "en"
    if not quick:
        lang_choice = typer.prompt(_t("lang_prompt", "en"), default="en").strip().lower()
        lang = "zh" if lang_choice in ("zh", "cn", "chinese", "中文") else "en"

    # ── Quick mode: skip prompts, read from env vars ──
    if quick:
        import os

        llm_provider = os.getenv("DRBRAIN_LLM_PROVIDER", "openai")
        llm_model = os.getenv("DRBRAIN_LLM_MODEL", "")
        llm_key = os.getenv(
            "OPENAI_API_KEY", os.getenv("ANTHROPIC_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
        )
        llm_base = os.getenv("DRBRAIN_LLM_BASE_URL", "")

        if not llm_model:
            provider_defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-sonnet-4-6",
                "deepseek": "deepseek-v4-flash",
                "ollama": "qwen2.5:7b",
            }
            llm_model = provider_defaults.get(llm_provider, "gpt-4o")

        embed_provider = os.getenv("DRBRAIN_EMBED_PROVIDER", "local")
        embed_model = os.getenv("DRBRAIN_EMBED_MODEL", "")
        embed_key = os.getenv("DRBRAIN_EMBED_API_KEY", "")
        embed_base = os.getenv("DRBRAIN_EMBED_API_BASE", "")

        config = {
            "llm": {
                "models": [
                    {
                        "provider": llm_provider,
                        "model": llm_model,
                        "api_key": llm_key or "${OPENAI_API_KEY}",
                        "base_url": llm_base or None,
                    }
                ],
            },
            "mineru": {
                "token": os.getenv("MINERU_TOKEN", ""),
                "model": os.getenv("DRBRAIN_MINERU_MODEL", "vlm"),
                "is_ocr": os.getenv("DRBRAIN_MINERU_OCR", "") == "1",
                "enable_formula": os.getenv("DRBRAIN_MINERU_FORMULA", "1") == "1",
                "enable_table": os.getenv("DRBRAIN_MINERU_TABLE", "1") == "1",
            },
            "api": {
                "deepxiv_token": os.getenv("DEEPXIV_TOKEN", ""),
                "s2_api_key": os.getenv("S2_API_KEY", ""),
                "crossref_email": os.getenv("CROSSREF_EMAIL", ""),
                "openalex_token": os.getenv("OPENALEX_TOKEN", ""),
            },
            "embed": {
                "provider": embed_provider,
                "device": os.getenv("DRBRAIN_EMBED_DEVICE", "auto"),
            },
        }
        if embed_provider == "openai-compat":
            config["embed"]["model"] = embed_model or "text-embedding-3-small"
            config["embed"]["api_base"] = embed_base or "https://api.openai.com/v1"
            config["embed"]["api_key"] = embed_key or "${OPENAI_API_KEY}"
        elif embed_provider == "local" and embed_model:
            config["embed"]["model"] = embed_model

        out = Path("config.local.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        typer.echo(_t("quick_config_written", "en", path=str(out)))

        from drbrain.config import load_config

        cfg = load_config()
        created = _ensure_directories(cfg)
        if created:
            plural = "y" if created == 1 else "ies"
            typer.echo(_t("env_dirs_created", "en", count=str(created), plural=plural))
        ok, warn = _brief_validation(cfg)
        for line in ok:
            typer.echo(f"  [OK] {line}")
        for line in warn:
            typer.echo(f"  [!]  {line}")
        typer.echo()

        _inject_agent_entries()

        typer.echo(_t("ready", "en"))
        typer.echo(f"  {_t('next_steps', 'en')}")
        return

    typer.echo("=" * 60)
    typer.echo(_t("header_banner", lang))
    typer.echo("=" * 60)
    typer.echo(f"  {_t('header_note', lang)}")
    typer.echo(f"  {_t('header_local_only', lang)}")
    typer.echo()

    # ── LLM ──
    typer.echo(_t("llm_heading", lang))
    provider = typer.prompt(f"  {_t('llm_provider', lang)}", default="openai")
    model = typer.prompt(f"  {_t('llm_model', lang)}", default="gpt-4o")
    api_key = typer.prompt(f"  {_t('llm_api_key', lang)}", default="", hide_input=True)
    base_url = typer.prompt(f"  {_t('llm_base_url', lang)}", default="", show_default=False)
    base_url = base_url if base_url else None
    models: list[dict] = [
        {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
    ]

    has_fallback = typer.confirm(f"  {_t('llm_add_fallback', lang)}", default=False)
    if has_fallback:
        fb_provider = typer.prompt(f"    {_t('llm_fb_provider', lang)}", default="ollama")
        fb_model = typer.prompt(f"    {_t('llm_fb_model', lang)}", default="qwen2.5:7b")
        fb_key = typer.prompt(f"    {_t('llm_fb_api_key', lang)}", default="", hide_input=True)
        fb_url = typer.prompt(
            f"    {_t('llm_fb_base_url', lang)}", default="http://localhost:11434"
        )
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
    typer.echo(_t("mineru_heading", lang))
    typer.echo(f"  {_t('mineru_note', lang)}")
    mineru_mode = typer.prompt(f"  {_t('mineru_mode', lang)}", default="flash")
    mineru_token = ""
    if mineru_mode == "token":
        typer.echo(f"  {_t('mineru_token_prompt', lang)}")
        mineru_token = typer.prompt(f"  {_t('mineru_token', lang)}", hide_input=True)

    mineru_model = "vlm"
    mineru_is_ocr = False
    mineru_enable_formula = True
    mineru_enable_table = True
    if typer.confirm(f"  {_t('mineru_advanced', lang)}", default=False):
        mineru_model = typer.prompt(f"    {_t('mineru_adv_model', lang)}", default="vlm")
        mineru_is_ocr = typer.confirm(f"    {_t('mineru_adv_ocr', lang)}", default=False)
        mineru_enable_formula = typer.confirm(f"    {_t('mineru_adv_formula', lang)}", default=True)
        mineru_enable_table = typer.confirm(f"    {_t('mineru_adv_table', lang)}", default=True)

    # ── API Keys ──
    typer.echo()
    typer.echo(_t("api_heading", lang))
    deepxiv_token = typer.prompt(f"  {_t('api_deepxiv', lang)}", default="", hide_input=True)
    s2_api_key = typer.prompt(
        f"  {_t('api_s2', lang)}",
        default="",
        hide_input=True,
    )
    crossref_email = typer.prompt(f"  {_t('api_crossref', lang)}", default="", show_default=False)
    openalex_token = typer.prompt(f"  {_t('api_openalex', lang)}", default="", hide_input=True)

    # ── Embedding ──
    typer.echo()
    typer.echo(_t("embed_heading", lang))
    typer.echo(f"  {_t('embed_desc', lang)}")
    embed_provider = typer.prompt(f"  {_t('embed_provider', lang)}", default="local")
    embed_model = ""
    embed_api_base = ""
    embed_api_key = ""
    if embed_provider == "openai-compat":
        embed_model = typer.prompt(
            f"    {_t('embed_oc_model', lang)}", default="text-embedding-3-small"
        )
        embed_api_base = typer.prompt(
            f"    {_t('embed_oc_api_base', lang)}", default="https://api.openai.com/v1"
        )
        embed_api_key = typer.prompt(
            f"    {_t('embed_oc_api_key', lang)}", default="${OPENAI_API_KEY}"
        )
    elif embed_provider == "local":
        embed_model = typer.prompt(
            f"    {_t('embed_local_model', lang)}", default="Qwen/Qwen3-Embedding-0.6B"
        )

    # ── Admin Password ──
    typer.echo()
    typer.echo(_t("admin_heading", lang))
    typer.echo(f"  {_t('admin_desc', lang)}")
    admin_password_hash = ""
    if typer.confirm(f"  {_t('admin_set', lang)}", default=False):
        pw = typer.prompt(f"  {_t('admin_pw', lang)}", hide_input=True)
        pw_confirm = typer.prompt(f"  {_t('admin_confirm', lang)}", hide_input=True)
        if pw == pw_confirm:
            from drbrain.auth import hash_password

            admin_password_hash = hash_password(pw)
        else:
            typer.echo(f"  {_t('admin_mismatch', lang)}")

    # ── Review ──
    typer.echo()
    typer.echo(_t("review_heading", lang))
    review_llm = f"  {_t('review_llm', lang)}:      {provider}/{model}"
    if has_fallback:
        review_llm += f" + {fb_provider}/{fb_model}"
    typer.echo(review_llm)

    review_mineru_label = (
        _t("review_token_set", lang) if mineru_token else _t("review_free_tier", lang)
    )
    typer.echo(f"  {_t('review_mineru', lang)}:   {mineru_mode} ({review_mineru_label})")

    dx_label = _t("review_key_set", lang) if deepxiv_token else _t("review_not_set", lang)
    s2_label = _t("review_key_set", lang) if s2_api_key else _t("review_anonymous", lang)
    cx_label = crossref_email or _t("review_none", lang)
    oa_label = _t("review_key_set", lang) if openalex_token else _t("review_anonymous", lang)
    typer.echo(
        f"  {_t('review_apis', lang)}:     "
        f"DeepXiv({dx_label})  S2({s2_label})  CrossRef({cx_label})  OpenAlex({oa_label})"
    )

    embed_detail = ""
    if embed_model:
        embed_detail = f" ({embed_model})"
    if embed_provider == "none":
        embed_detail += f" ({_t('review_disabled', lang)})"
    typer.echo(f"  {_t('review_embed', lang)}:    {embed_provider}{embed_detail}")

    if not typer.confirm(f"\n  {_t('review_write', lang)}", default=True):
        typer.echo(_t("review_cancelled", lang))
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

    config: dict = {
        "llm": {"models": models},
        "mineru": {
            "token": mineru_token,
            "model": mineru_model,
            "is_ocr": mineru_is_ocr,
            "enable_formula": mineru_enable_formula,
            "enable_table": mineru_enable_table,
        },
    }
    if admin_password_hash:
        config["admin"] = {"password_hash": admin_password_hash}
    if api_cfg:
        config["api"] = api_cfg

    embed_cfg: dict = {"provider": embed_provider, "device": "auto"}
    if embed_provider == "openai-compat":
        if embed_model:
            embed_cfg["model"] = embed_model
        if embed_api_base:
            embed_cfg["api_base"] = embed_api_base
        if embed_api_key:
            embed_cfg["api_key"] = embed_api_key
    elif embed_provider == "local" and embed_model:
        embed_cfg["model"] = embed_model
    config["embed"] = embed_cfg

    out = Path("config.local.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    typer.echo(f"\n  {_t('review_config_written', lang)} {out}")

    # ── Initialize environment ──
    typer.echo(f"\n{_t('env_heading', lang)}")
    from drbrain.config import load_config

    cfg = load_config()
    created = _ensure_directories(cfg)
    if created:
        plural = "y" if created == 1 else "ies"
        typer.echo(f"  {_t('env_dirs_created', lang, count=str(created), plural=plural)}")
    else:
        typer.echo(f"  {_t('env_dirs_present', lang)}")

    ok, warn = _brief_validation(cfg)
    for line in ok:
        typer.echo(f"  [OK] {line}")
    for line in warn:
        typer.echo(f"  [!]  {line}")

    typer.echo()

    _inject_agent_entries(lang=lang)

    if not warn:
        logger.info("[setup] complete — no warnings")
        typer.echo(_t("ready", lang))
        typer.echo(f"  {_t('next_steps', lang)}")
    else:
        logger.info("[setup] complete — %d warning(s)", len(warn))
        typer.echo(_t("setup_warnings", lang, count=str(len(warn))))
        typer.echo(_t("setup_check_hint", lang))
