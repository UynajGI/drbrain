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
        from typing import Any, cast

        return [getattr(self, f.name) for f in fields(cast(Any, self))]


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
    sciverse_token: str = ""
    sciverse_tokens: list[str] = field(
        default_factory=list
    )  # 多账号 token 列表(真并行:每账号独立限流)
    sciverse_base_url: str = "https://api.sciverse.space"
    sciverse_rate_limit: int = 30
    ref_base_url: str = ""
    ref_model: str = ""
    ref_api_key: str = ""


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
class LlamaIndexEvalConfig(_ConfigBase):
    """Evaluation settings for the LlamaIndex RAG layer (T1 infra).

    Attributes:
        golden_set: Path to the golden query set (JSONL) used for eval.
        split: Dev/val/test split names to prevent leakage during eval.
    """

    golden_set: str = "data/llamaindex/golden.jsonl"
    split: list[str] = field(default_factory=lambda: ["dev", "val", "test"])


@dataclass
class LlamaIndexConfig(_ConfigBase):
    """LlamaIndex RAG layer configuration (T1: infrastructure).

    Attributes:
        enabled: Master switch. ``False`` falls back to legacy implementations.
        llm: LLM bridge backend (``"litellm"``); wired up in T2.
        vector_store: ``"memory"`` | ``"chroma"`` (chromadb is optional).
        storage_dir: Directory for LlamaIndex index persistence.
        retrievers: Fusion legs, e.g. ``["bm25", "vector"]``.
        fusion_mode: ``"reciprocal_rank"`` | ``"relative_score"``.
        rerank: Enable reranking by default.
        rerank_model: Reranker model name (Qwen/Qwen3-Reranker-0.6B).
        rerank_top_k: Candidate count fed to the reranker.
        similarity_cutoff: SimilarityPostProcessor threshold.
        streaming: Enable streaming responses.
         max_node_tokens: Long PageIndex nodes above this size are split into
            paragraph chunks (each chunk keeps the parent node_id + a
            ``#index`` suffix) so embedding stays bounded on GPU. Default 4000:
            measured Qwen3-Embedding-0.6B fp32 per-sample memory is quadratic
            (4096 tokens ≈ 3.6GB, 8192 ≈ 12.2GB) — 8000-token sequences OOM a
             16GB V100 (T9 fix for the 39KB-node OOM; 4 chars ≈ 1 token).
         mcp_require_trusted: Require explicit ``trusted: true`` and a
             non-empty MCP tool allowlist for the LlamaIndex agent path.
             Defaults to ``False`` for existing local configurations.
         eval: Evaluation settings (golden set + split).
    """

    enabled: bool = False
    llm: str = "litellm"
    vector_store: str = "memory"
    storage_dir: str = "data/llamaindex"
    retrievers: list[str] = field(default_factory=lambda: ["bm25", "vector"])
    fusion_mode: str = "reciprocal_rank"
    rerank: bool = True
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    rerank_top_k: int = 20
    similarity_cutoff: float = 0.7
    streaming: bool = True
    max_node_tokens: int = 4000
    mcp_require_trusted: bool = False
    eval: LlamaIndexEvalConfig = field(default_factory=LlamaIndexEvalConfig)

    @classmethod
    def from_dict(cls, data: dict) -> LlamaIndexConfig:
        """Build from a raw YAML dict, nesting the ``eval`` sub-config."""
        raw = dict(data or {})
        eval_raw = raw.pop("eval", None) or {}
        return cls(eval=LlamaIndexEvalConfig(**eval_raw), **raw)


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
class AutoresearchConfig(_ConfigBase):
    """Operator settings for durable autoresearch runs.

    The feature remains opt-in so existing deployments do not start research
    loops accidentally.  ``require_rag_evidence`` enables strict evidence
    gating for operators that intentionally run against the RAG corpus.
    """

    enabled: bool = False
    run_dir: str = "workspace/autoresearch"
    plugins_dir: str = ""
    mcp_servers: list[dict] = field(default_factory=list)
    step_capabilities: dict[str, list[str]] = field(default_factory=dict)
    n_critics: int = 3
    max_cycles: int = 10
    stagnation_cycles: int = 3
    max_adaptations: int = 2
    lease_seconds: float = 900.0
    require_rag_evidence: bool = False


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
    llamaindex: LlamaIndexConfig = field(default_factory=LlamaIndexConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    autoresearch: AutoresearchConfig = field(default_factory=AutoresearchConfig)
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

        with open(base, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        local = Path(local_path) if local_path is not None else Path("config.local.yaml")
        if local.exists():
            with open(local, encoding="utf-8") as f:
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
            llamaindex=LlamaIndexConfig.from_dict(cfg.get("llamaindex", {})),
            backup=BackupConfig(
                ssh_bin=backup_raw.get("ssh_bin", "ssh"),
                rsync_bin=backup_raw.get("rsync_bin", "rsync"),
                targets=backup_targets,
            ),
            autoresearch=AutoresearchConfig(**cfg.get("autoresearch", {})),
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
