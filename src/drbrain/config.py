"""Typed config dataclasses with YAML loader, env var resolution, and dict-backward-compat."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# ── Env var pattern for ${VAR} resolution ──

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


# ── Base class for dict-like backward compatibility ──


class _ConfigBase:
    """Mixin providing dict-like access for backward compatibility.

    Supports:
      cfg["key"]           → getattr(cfg, "key")
      cfg.get("key", def)   → getattr(cfg, "key", def)
      cfg.values()          → list of field values (for iteration over DirsConfig paths)
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def values(self):
        """Return field values as a list (supports list(dirs_config.values()) pattern)."""
        return [getattr(self, f.name) for f in fields(self)]


# ── Sub-config dataclasses ──


@dataclass
class LLMConfig(_ConfigBase):
    models: list[dict] = field(default_factory=list)


@dataclass
class MinerUConfig(_ConfigBase):
    token: str = ""
    model: str = "vlm"
    is_ocr: bool = False
    enable_formula: bool = True
    enable_table: bool = True
    max_pages: int = 150


@dataclass
class ApiConfig(_ConfigBase):
    deepxiv_token: str = ""
    s2_api_key: str = ""
    s2_rate_limit: int = 100
    cache_ttl: int = 86400
    crossref_email: str = ""
    openalex_token: str = ""


@dataclass
class DirsConfig(_ConfigBase):
    inbox: str = "data/spool/inbox"
    pending: str = "data/spool/pending"
    papers: str = "data/papers"
    reports: str = "data/reports"
    cache: str = "data/cache"
    logs: str = "data/logs"
    backups: str = "data/backups"
    citation_styles: str = "data/citation_styles"


@dataclass
class DBConfig(_ConfigBase):
    path: str = "data/drbrain.db"


@dataclass
class ExtractConfig(_ConfigBase):
    max_concurrent: int = 10


@dataclass
class BM25Config(_ConfigBase):
    k1: float = 1.5
    b: float = 0.75


@dataclass
class QueueConfig(_ConfigBase):
    weak_threshold: float = 0.7
    auto_accept: float = 0.9


@dataclass
class FetchConfig(_ConfigBase):
    max_concurrent: int = 3
    timeout_per_fetch: int = 60
    user_agent: str = "DrBrain/0.1"
    fallback_order: list[str] = field(
        default_factory=lambda: ["openalex", "arxiv", "unpaywall", "doi_direct"]
    )
    unpaywall_email: str = ""
    institutional_proxy: str = ""
    proxy_type: str = ""  # "ezproxy" or "url_prefix"


@dataclass
class EmbedConfig(_ConfigBase):
    """Semantic vector embedding configuration (ScholarAIO pattern).

    Attributes:
        provider: ``"local"`` | ``"openai-compat"`` | ``"none"``.
        model: Sentence Transformer model name or HuggingFace ID.
        cache_dir: Local model cache directory.
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        top_k: Default number of results for vector search.
        source: Model download source, ``"modelscope"`` | ``"huggingface"``.
        hf_endpoint: Optional HuggingFace mirror URL.
        api_base: OpenAI-compatible API base URL (``/v1`` prefix).
        api_key: API key for cloud embedding.
        batch_size: Batch size for embedding requests.
    """

    provider: str = "local"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    cache_dir: str = "~/.cache/modelscope/hub/models"
    device: str = "auto"
    top_k: int = 10
    source: str = "modelscope"
    hf_endpoint: str = ""
    api_base: str = ""
    api_key: str = ""
    batch_size: int = 64


@dataclass
class BackupTargetConfig(_ConfigBase):
    """Rsync backup target configuration.

    Attributes:
        host: Remote SSH host.
        user: Optional SSH username.
        path: Remote destination path.
        port: SSH port.
        identity_file: Optional SSH identity file path.
        password: Optional SSH password for non-interactive backup.
        mode: Transfer mode — ``"default"`` | ``"append"`` | ``"append-verify"``.
        compress: Whether to enable rsync compression.
        enabled: Whether the target is available for use.
        exclude: Rsync exclude patterns.
    """

    host: str = ""
    user: str = ""
    path: str = ""
    port: int = 22
    identity_file: str = ""
    password: str = ""
    mode: str = "default"
    compress: bool = True
    enabled: bool = True
    exclude: list[str] = field(default_factory=list)


@dataclass
class BackupConfig(_ConfigBase):
    """Backup configuration for rsync-based data sync."""

    ssh_bin: str = "ssh"
    rsync_bin: str = "rsync"
    targets: dict = field(default_factory=dict)


@dataclass
class Config(_ConfigBase):
    llm: LLMConfig = field(default_factory=LLMConfig)
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    dirs: DirsConfig = field(default_factory=DirsConfig)
    db: DBConfig = field(default_factory=DBConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    queue: QueueConfig = field(default_factory=QueueConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    admin: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        base_path: str | Path,
        local_path: str | Path | None = None,
    ) -> Config:
        """Load config from YAML, deep-merge local overlay, resolve env vars.

        Args:
            base_path: Path to config.yaml.
            local_path: Path to config.local.yaml. Defaults to "config.local.yaml" if None.

        Returns:
            Typed Config instance.
        """
        base = Path(base_path)
        if not base.exists():
            raise FileNotFoundError(f"Config not found: {base}")

        with open(base) as f:
            cfg = yaml.safe_load(f) or {}

        local = Path(local_path) if local_path is not None else Path("config.local.yaml")
        if local.exists():
            with open(local) as f:
                overlay = yaml.safe_load(f) or {}
            cfg = merge_dicts(cfg, overlay)

        cfg = _resolve_env_vars(cfg)

        backup_raw = cfg.get("backup", {})
        backup_targets_raw = backup_raw.get("targets", {})
        backup_targets = {name: BackupTargetConfig(**t) for name, t in backup_targets_raw.items()}
        return cls(
            llm=LLMConfig(**cfg.get("llm", {})),
            mineru=MinerUConfig(**cfg.get("mineru", {})),
            api=ApiConfig(**cfg.get("api", {})),
            dirs=DirsConfig(**cfg.get("dirs", {})),
            db=DBConfig(**cfg.get("db", {})),
            extract=ExtractConfig(**cfg.get("extract", {})),
            bm25=BM25Config(**cfg.get("bm25", {})),
            queue=QueueConfig(**cfg.get("queue", {})),
            fetch=FetchConfig(**cfg.get("fetch", {})),
            embed=EmbedConfig(**cfg.get("embed", {})),
            backup=BackupConfig(
                ssh_bin=backup_raw.get("ssh_bin", "ssh"),
                rsync_bin=backup_raw.get("rsync_bin", "rsync"),
                targets=backup_targets,
            ),
        )


# ── Deep merge ──


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge: override wins for leaf values, base keys preserved."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


# ── Env var resolution ──


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR} patterns in all string values."""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(_env_replace, obj)
    return obj


def _env_replace(match: re.Match) -> str:
    """Replace a single ${VAR} match with environment variable value."""
    return os.environ.get(match.group(1), "")


# ── Public loader ──


def load_config(
    base_path: str | Path = "config.yaml",
    local_path: str | Path | None = None,
) -> Config:
    """Load base config and optionally merge local overlay.

    Returns a typed Config object with full dict-like backward compatibility.
    """
    return Config.from_yaml(base_path, local_path)
