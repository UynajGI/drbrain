"""Typed config dataclasses with YAML loader, env var resolution, and dict-backward-compat."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
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
    # 节点级模型覆盖（autoresearch workflow step_name → 该节点的完整 fallback 链）。
    # 未命中的节点用全局 models；列表需写全（不自动追加全局链）。
    node_models: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class MinerUConfig(_ConfigBase):
    token: str = ""
    model: str = "vlm"
    is_ocr: bool = False
    enable_formula: bool = True
    enable_table: bool = True
    max_pages: int = 150
    use_anydoc: bool = True
    ocr_enabled: bool = False
    ocr_language: str = "eng"


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
    extra_gpus: list[int] = field(default_factory=list)  # 大规模嵌入的额外并行卡号
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
    # "sql" = 主库副本（drbrain_rag.db，node_texts+FTS5+tree_vectors）直检，
    # 目标架构（全文入库，无独立索引构建），也是唯一规模就绪的引擎；
    # "llamaindex" 降级为单库/调试用（全量 docstore 反序列化，0.8M 篇不可用）
    rag_engine: str = "sql"
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
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError("Config section 'llamaindex' must be a mapping")
        raw = dict(data)
        eval_raw = raw.pop("eval", {})
        if not isinstance(eval_raw, Mapping):
            raise ValueError("Config section 'llamaindex.eval' must be a mapping")
        try:
            return cls(eval=LlamaIndexEvalConfig(**dict(eval_raw)), **raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid llamaindex config: {exc}") from exc


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
    loops accidentally.  ``require_rag_evidence`` defaults to strict: claims
    without durable RAG evidence citations stay predictions. Hosts running
    without a corpus (compute-only loops governed by the T4 numeric gate)
    opt out explicitly.
    """

    enabled: bool = False
    run_dir: str = "workspace/autoresearch"
    plugins_dir: str = ""
    mcp_servers: list[dict] = field(default_factory=list)
    step_capabilities: dict[str, list[str]] = field(default_factory=dict)
    n_critics: int = 3
    single_agent: bool = False
    max_cycles: int = 10
    stagnation_cycles: int = 3
    max_adaptations: int = 2
    lease_seconds: float = 900.0
    # Per-workflow-step timeout; rate-limited LLM fallbacks can stretch a
    # single step (identify_gaps etc.) well past the 600s default.
    step_timeout_seconds: float = 600.0
    # Passed through to ResearchDirector, which validates supported limit names
    # and numeric values before a durable run is created or resumed.
    budget: dict[str, int | float] = field(default_factory=dict)
    require_rag_evidence: bool = True
    # Strict compute mode: a durable run proposing hypotheses without any
    # visible job-contract tool (run_python/check_job/read_job) hard-pauses for
    # manual review instead of settling literature-only claims. Opt out only
    # for intentionally literature-only runs.
    require_compute_tools: bool = True
    # Explicit job-contract tool names arming the T4 gate (exact match); empty
    # keeps the built-in (run_python, check_job, read_job).
    compute_tool_names: list[str] = field(default_factory=list)


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
        overlay_path: str | Path | None = None,
    ) -> Config:
        """Load config layers from YAML and resolve environment variables.

        Args:
            base_path: Path to config.yaml.
            local_path: Path to the local layer. When omitted, the loader looks
                for ``config.local.yaml`` next to ``base_path``. Passing an
                explicit path preserves the legacy API and disables that
                sibling lookup (a missing path is ignored).
            overlay_path: Optional highest-precedence overlay. This is useful
                for a command-specific config while retaining the normal
                ``base -> config.local.yaml -> overlay`` order. An explicitly
                requested overlay must exist.

        Returns:
            Typed Config instance.
        """
        runtime = _active_runtime_context()
        base = _prepare_config_path(
            base_path,
            runtime=runtime,
            label="base config",
            required=True,
        )

        cfg = _read_yaml_mapping(base)

        # Resolve the implicit local layer relative to the base file, rather
        # than the process CWD. This keeps custom/test configurations isolated
        # from an unrelated config.local.yaml in the caller's directory.
        local = Path(local_path) if local_path is not None else base.parent / "config.local.yaml"
        if runtime is not None and not local.is_absolute():
            local = runtime.root / local
        if local.exists() or local.is_symlink():
            local = _prepare_config_path(
                local,
                runtime=runtime,
                label="local config",
                required=True,
            )
            cfg = merge_dicts(cfg, _read_yaml_mapping(local))

        if overlay_path is not None:
            overlay = Path(overlay_path)
            if runtime is not None and not overlay.is_absolute():
                overlay = runtime.root / overlay
            overlay = _prepare_config_path(
                overlay,
                runtime=runtime,
                label="config overlay",
                required=True,
            )
            cfg = merge_dicts(cfg, _read_yaml_mapping(overlay))

        cfg = _resolve_env_vars(cfg)

        sections = {
            name: _mapping_section(cfg, name)
            for name in (
                "llm",
                "mineru",
                "api",
                "dirs",
                "db",
                "extract",
                "bm25",
                "queue",
                "fetch",
                "embed",
                "llamaindex",
                "backup",
                "autoresearch",
            )
        }
        backup_raw = sections["backup"]
        backup_targets_raw = backup_raw.get("targets", {})
        if not isinstance(backup_targets_raw, Mapping):
            raise ValueError("Config section 'backup.targets' must be a mapping")
        backup_targets: dict[str, BackupTargetConfig] = {}
        for name, target in backup_targets_raw.items():
            if not isinstance(target, Mapping):
                raise ValueError(f"Config backup target {name!r} must be a mapping")
            try:
                backup_targets[str(name)] = BackupTargetConfig(**dict(target))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid backup target {name!r}: {exc}") from exc

        admin_raw = cfg.get("admin", {})
        if not isinstance(admin_raw, Mapping):
            raise ValueError("Config section 'admin' must be a mapping")
        try:
            result = cls(
                llm=LLMConfig(**sections["llm"]),
                mineru=MinerUConfig(**sections["mineru"]),
                api=ApiConfig(**sections["api"]),
                dirs=DirsConfig(**sections["dirs"]),
                db=DBConfig(**sections["db"]),
                extract=ExtractConfig(**sections["extract"]),
                bm25=BM25Config(**sections["bm25"]),
                queue=QueueConfig(**sections["queue"]),
                fetch=FetchConfig(**sections["fetch"]),
                embed=EmbedConfig(**sections["embed"]),
                llamaindex=LlamaIndexConfig.from_dict(sections["llamaindex"]),
                backup=BackupConfig(
                    ssh_bin=backup_raw.get("ssh_bin", "ssh"),
                    rsync_bin=backup_raw.get("rsync_bin", "rsync"),
                    targets=backup_targets,
                ),
                autoresearch=AutoresearchConfig(**sections["autoresearch"]),
                admin=dict(admin_raw),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid config: {exc}") from exc
        if runtime is not None:
            # Direct library callers do not pass through the CLI callback.  A
            # selected runtime must therefore validate the typed paths here as
            # well, before any service can open its database or artifact dirs.
            runtime.validate_config(result)
        return result


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


def _active_runtime_context():
    """Return the environment-selected runtime, if one is active.

    Import lazily to keep configuration usable in minimal deployments and to
    avoid making the runtime module depend on the typed config classes.
    """
    if "DRBRAIN_ROOT" not in os.environ and "DRBRAIN_RUNTIME_ROOT" not in os.environ:
        return None
    from drbrain.runtime import RuntimeContext

    # Let RuntimeContext enforce primary/legacy precedence and reject an
    # explicitly empty selector instead of treating it as unset.
    return RuntimeContext.create()


def _prepare_config_path(
    path: str | Path,
    *,
    runtime,
    label: str,
    required: bool,
) -> Path:
    """Validate a config layer before it is opened.

    With an active runtime, every layer must be beneath that root.  Without
    one, retain the legacy ability to load an explicitly supplied absolute
    config while still rejecting lexical symlink aliases (including ancestors).
    """
    candidate = Path(path).expanduser()
    if runtime is not None:
        if not candidate.is_absolute():
            candidate = runtime.root / candidate
        return runtime.validate_config_file(candidate, label=label, required=required)

    from drbrain.runtime import _first_symlink_component

    alias = _first_symlink_component(candidate)
    if alias is not None:
        raise ValueError(f"{label} must not contain a symlink component: {alias}")
    if not candidate.exists():
        if required:
            raise FileNotFoundError(f"{label} not found: {candidate}")
        return candidate
    if not candidate.is_file():
        raise ValueError(f"{label} is not a regular file: {candidate}")
    return candidate


def _read_yaml_mapping(path: Path) -> dict:
    """Read a YAML config layer and require a mapping at its root."""
    path = Path(path)
    # Config layers can contain credentials.  Never follow a symlink here:
    # callers outside the CLI (pipeline workers and library integrations) must
    # receive the same isolation guarantee as the root callback.
    from drbrain.runtime import _first_symlink_component

    alias = _first_symlink_component(path)
    if alias is not None:
        raise ValueError(f"Config file must not contain a symlink component: {alias}")
    if path.is_symlink():
        raise ValueError(f"Config file must not be a symlink: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config file is not a regular file: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            value = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        # Parser diagnostics include the offending source line.  That line can
        # contain an inline API key, so scrub it before the exception reaches
        # a CLI/logging boundary or a library caller.
        raise ValueError(f"Invalid YAML config: {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return value


def _mapping_section(config: Mapping[str, Any], name: str) -> dict:
    """Return a config section as a mapping, or fail closed with context."""
    # Missing sections are valid and use dataclass defaults.  An explicitly
    # supplied falsey value (``null``, ``[]``, ``''``) is malformed, however;
    # coercing it to ``{}`` would silently discard a deployment mistake.
    value = config[name] if name in config else {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return dict(value)


def _env_replace(match: re.Match) -> str:
    """Replace a single ${VAR} match with environment variable value."""
    return os.environ.get(match.group(1), "")


# ── Public loader ──


def load_config(
    base_path: str | Path = "config.yaml",
    local_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> Config:
    """Load base config and optional local/command overlays.

    Returns a typed Config object with full dict-like backward compatibility.
    """
    # Standalone workers launched with ``DRBRAIN_ROOT`` may have a different
    # current working directory.  Keep the default config lookup in the same
    # runtime namespace while preserving explicit ``base_path`` behavior.
    if str(base_path) == "config.yaml":
        if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
            from drbrain.runtime import RuntimeContext

            runtime_base = RuntimeContext.create().base_config_path
            base_path = runtime_base
        env_overlay = None
        if overlay_path is None:
            if "DRBRAIN_CONFIG" in os.environ:
                env_overlay = os.environ["DRBRAIN_CONFIG"]
            elif "DRBRAIN_CONFIG_PATH" in os.environ:
                env_overlay = os.environ["DRBRAIN_CONFIG_PATH"]
            if env_overlay == "":
                raise ValueError("DRBRAIN_CONFIG must not be empty")
            if env_overlay:
                candidate = Path(env_overlay).expanduser()
                if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
                    if not candidate.is_absolute():
                        candidate = Path(base_path).parent / candidate
                    try:
                        is_base = candidate.resolve() == Path(base_path).resolve()
                    except OSError:
                        is_base = False
                    if not is_base:
                        overlay_path = candidate
                else:
                    overlay_path = candidate
    return Config.from_yaml(base_path, local_path, overlay_path)
