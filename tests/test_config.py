"""Tests for typed Config dataclass with YAML loader, env var resolution, and dict-backward-compat."""

import os
import tempfile
from pathlib import Path

from drbrain.config import (
    ApiConfig,
    BackupConfig,
    BM25Config,
    Config,
    DBConfig,
    DirsConfig,
    EmbedConfig,
    ExtractConfig,
    LLMConfig,
    MinerUConfig,
    QueueConfig,
    load_config,
    merge_dicts,
)

# ── Default value tests ──


def test_llm_config_defaults():
    c = LLMConfig()
    assert c.models == []
    assert c["models"] == []
    assert c.get("models", [{"x": 1}]) == []
    assert c.get("nonexistent", "fallback") == "fallback"


def test_mineru_config_defaults():
    c = MinerUConfig()
    assert c.token == ""
    assert c.model == "vlm"
    assert c.is_ocr is False
    assert c.enable_formula is True
    assert c.enable_table is True
    assert c.max_pages == 150
    assert c["model"] == "vlm"
    assert c.get("token") == ""
    assert c.get("nonexistent", 42) == 42


def test_api_config_defaults():
    c = ApiConfig()
    assert c.deepxiv_token == ""
    assert c.s2_api_key == ""
    assert c.s2_rate_limit == 100
    assert c.cache_ttl == 86400
    assert c.crossref_email == ""
    assert c.openalex_token == ""


def test_dirs_config_defaults():
    c = DirsConfig()
    assert c.inbox == "data/spool/inbox"
    assert c.pending == "data/spool/pending"
    assert c.papers == "data/papers"
    assert c.reports == "data/reports"
    assert c.cache == "data/cache"
    assert c.logs == "data/logs"


def test_dirs_config_values():
    c = DirsConfig()
    vals = c.values()
    assert isinstance(vals, list)
    assert "data/spool/inbox" in vals
    assert "data/papers" in vals
    assert "data/logs" in vals


def test_db_config_defaults():
    c = DBConfig()
    assert c.path == "data/drbrain.db"


def test_extract_config_defaults():
    c = ExtractConfig()
    assert c.max_concurrent == 10


def test_bm25_config_defaults():
    c = BM25Config()
    assert c.k1 == 1.5
    assert c.b == 0.75


def test_queue_config_defaults():
    c = QueueConfig()
    assert c.weak_threshold == 0.7
    assert c.auto_accept == 0.9


def test_config_defaults():
    c = Config()
    assert isinstance(c.llm, LLMConfig)
    assert isinstance(c.mineru, MinerUConfig)
    assert isinstance(c.api, ApiConfig)
    assert isinstance(c.dirs, DirsConfig)
    assert isinstance(c.db, DBConfig)
    assert isinstance(c.extract, ExtractConfig)
    assert isinstance(c.bm25, BM25Config)
    assert isinstance(c.queue, QueueConfig)
    assert isinstance(c.embed, EmbedConfig)
    assert isinstance(c.backup, BackupConfig)
    assert c.embed.provider == "local"
    assert c.embed.device == "auto"


# ── Typed access tests ──


def test_typed_dot_access():
    c = Config()
    assert c.db.path == "data/drbrain.db"
    assert c.dirs.inbox == "data/spool/inbox"
    assert c.bm25.k1 == 1.5
    assert c.bm25.b == 0.75
    assert c.extract.max_concurrent == 10
    assert c.embed.provider == "local"
    assert c.embed.top_k == 10
    assert c.backup.ssh_bin == "ssh"
    assert c.backup.rsync_bin == "rsync"
    assert c.backup.targets == {}


# ── Backward-compat dict access tests ──


def test_backward_compat_getitem_chained():
    """cfg['db']['path'] should work."""
    c = Config()
    assert c["db"]["path"] == "data/drbrain.db"


def test_backward_compat_get_chained():
    """cfg.get('llm', {}).get('models', []) should work."""
    c = Config()
    assert c.get("llm", {}).get("models", []) == []


def test_backward_compat_get_chained_with_default():
    """cfg.get('dirs', {}).get('inbox', 'data/spool/inbox') should work."""
    c = Config()
    assert c.get("dirs", {}).get("inbox", "data/spool/inbox") == "data/spool/inbox"


def test_backward_compat_get_nonexistent_field_with_default():
    """cfg.get('dirs', {}).get('backups', 'data/backups') should return default."""
    c = Config()
    assert c.get("dirs", {}).get("backups", "data/backups") == "data/backups"


def test_backward_compat_getitem_nonexistent_raises():
    c = Config()
    try:
        _ = c["nonexistent_section"]
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_backward_compat_get_nonexistent_returns_default():
    c = Config()
    assert c.get("nonexistent_section", {"foo": "bar"}) == {"foo": "bar"}


def test_backward_compat_list_dirs_values():
    """list(dirs_config.values()) should work (used in commands.py and setup.py)."""
    c = Config()
    dirs_config = c.get("dirs", {})
    assert dirs_config  # truthy
    dir_paths = list(dirs_config.values())
    assert "data/spool/inbox" in dir_paths
    assert "data/papers" in dir_paths
    assert "data/logs" in dir_paths
    assert len(dir_paths) == 6


def test_backward_compat_api_get_chained():
    """cfg.get('api', {}).get('deepxiv_token', '') should work."""
    c = Config()
    assert c.get("api", {}).get("deepxiv_token", "") == ""


def test_backward_compat_mineru_getitem():
    """cfg['mineru']['model'] should work."""
    c = Config()
    assert c["mineru"]["model"] == "vlm"


# ── from_yaml tests ──


def test_from_yaml_loads_real_config():
    """Config.from_yaml() should load the real config.yaml successfully."""
    c = Config.from_yaml("config.yaml")
    assert isinstance(c, Config)
    assert c.db.path == "data/drbrain.db"
    assert c.bm25.k1 == 1.5
    assert c.extract.max_concurrent == 10
    assert c.queue.weak_threshold == 0.7
    assert c.queue.auto_accept == 0.9
    assert c.mineru.model == "vlm"
    assert c.dirs.inbox == "data/spool/inbox"


def test_from_yaml_with_partial_config():
    """Config.from_yaml should handle missing sections via defaults."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        base.write_text("""
db:
  path: /custom/path.db
""")
        c = Config.from_yaml(base, local_path=Path(td) / "nonexistent.yaml")
        assert c["db"]["path"] == "/custom/path.db"
        # Missing sections use defaults
        assert c.dirs.inbox == "data/spool/inbox"
        assert c.bm25.b == 0.75
        assert c.extract.max_concurrent == 10


def test_from_yaml_with_local_overlay():
    """Local overlay should deep-merge and override base values."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        local = Path(td) / "config.local.yaml"
        base.write_text("""
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: null
      base_url: null
db:
  path: data/drbrain.db
""")
        local.write_text("""
llm:
  models:
    - provider: anthropic
      model: claude-sonnet-4-20250514
      api_key: sk-test123
      base_url: null
db:
  path: /custom/path.db
""")
        c = Config.from_yaml(base, local_path=local)
        assert c["llm"]["models"][0]["provider"] == "anthropic"
        assert c["llm"]["models"][0]["api_key"] == "sk-test123"
        assert c["db"]["path"] == "/custom/path.db"


# ── Env var resolution tests ──


def test_env_var_resolution_in_strings():
    """${VAR} patterns should be resolved from environment."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        base.write_text("""
db:
  path: "${MY_DB_PATH}"
api:
  deepxiv_token: "${MY_TOKEN}"
mineru:
  token: "${MINERU_TOKEN_VAL}"
""")
        os.environ["MY_DB_PATH"] = "/env/path.db"
        os.environ["MY_TOKEN"] = "secret-token-123"
        os.environ["MINERU_TOKEN_VAL"] = "mineru-token-456"
        try:
            c = Config.from_yaml(base, local_path=Path(td) / "nonexistent.yaml")
            assert c.db.path == "/env/path.db"
            assert c.api.deepxiv_token == "secret-token-123"
            assert c.mineru.token == "mineru-token-456"
        finally:
            del os.environ["MY_DB_PATH"]
            del os.environ["MY_TOKEN"]
            del os.environ["MINERU_TOKEN_VAL"]


def test_env_var_resolution_unknown_var_becomes_empty():
    """Unknown ${VAR} should become empty string."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        base.write_text("""
api:
  deepxiv_token: "${NONEXISTENT_VAR_FOO_BAR_XYZ}"
""")
        c = Config.from_yaml(base, local_path=Path(td) / "nonexistent.yaml")
        assert c.api.deepxiv_token == ""


def test_env_var_resolution_in_dict_values():
    """${VAR} in nested dict values should be resolved."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        base.write_text("""
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: "${LLM_KEY}"
      base_url: null
""")
        os.environ["LLM_KEY"] = "sk-test-env-key"
        try:
            c = Config.from_yaml(base, local_path=Path(td) / "nonexistent.yaml")
            assert c.llm.models[0]["api_key"] == "sk-test-env-key"
        finally:
            del os.environ["LLM_KEY"]


def test_env_var_resolution_no_placeholders_unchanged():
    """Strings without ${VAR} patterns should not be modified."""
    c = Config()
    assert c.dirs.inbox == "data/spool/inbox"
    assert c.db.path == "data/drbrain.db"


# ── merge_dicts tests (existing behavior preserved) ──


def test_merge_dicts_nested():
    """merge_dicts does deep merge of nested dicts."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    result = merge_dicts(base, override)
    assert result["a"]["b"] == 10
    assert result["a"]["c"] == 2
    assert result["d"] == 3
    assert result["e"] == 4


def test_merge_dicts_empty_override():
    """merge_dicts with empty override returns copy of base."""
    base = {"a": 1, "b": {"c": 2}}
    result = merge_dicts(base, {})
    assert result == base
    assert result is not base


def test_merge_dicts_empty_base():
    """merge_dicts with empty base returns override."""
    override = {"a": 1}
    result = merge_dicts({}, override)
    assert result == override
    assert result is not override


def test_merge_dicts_new_top_level_key():
    """New top-level keys from override are added."""
    result = merge_dicts({"a": 1}, {"b": 2})
    assert result == {"a": 1, "b": 2}


# ── load_config backward compat tests ──


def test_load_config_defaults():
    """load_config without local overlay loads base YAML."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        base.write_text("""
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: null
      base_url: null
mineru:
  token: ""
  model: vlm
  is_ocr: false
  enable_formula: true
  enable_table: true
db:
  path: data/db/drbrain.db
dirs:
  inbox: data/spool/inbox
  papers: data/papers
  reports: data/reports
  cache: data/cache
  logs: data/logs
api:
  s2_rate_limit: 100
  cache_ttl: 86400
bm25:
  k1: 1.5
  b: 0.75
""")
        cfg = load_config(base_path=base, local_path=Path(td) / "nonexistent.yaml")
        assert cfg["llm"]["models"][0]["model"] == "gpt-4o"
        assert cfg["mineru"]["model"] == "vlm"
        assert cfg["db"]["path"] == "data/db/drbrain.db"
        assert cfg["bm25"]["k1"] == 1.5


def test_load_config_with_local_overlay():
    """Local overlay overrides base config values."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "config.yaml"
        local = Path(td) / "config.local.yaml"
        base.write_text("""
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: null
      base_url: null
db:
  path: data/drbrain.db
""")
        local.write_text("""
llm:
  models:
    - provider: anthropic
      model: claude-sonnet-4-20250514
      api_key: sk-test123
      base_url: null
db:
  path: /custom/path.db
""")
        cfg = load_config(base_path=base, local_path=local)
        assert cfg["llm"]["models"][0]["provider"] == "anthropic"
        assert cfg["llm"]["models"][0]["api_key"] == "sk-test123"
        assert cfg["db"]["path"] == "/custom/path.db"


# ── Constructor tests ──


def test_config_constructor_with_custom_values():
    """Config can be constructed directly with custom sub-configs."""
    c = Config(
        db=DBConfig(path=":memory:"),
        bm25=BM25Config(k1=2.0, b=0.5),
    )
    assert c.db.path == ":memory:"
    assert c["db"]["path"] == ":memory:"
    assert c.bm25.k1 == 2.0
    assert c.dirs.inbox == "data/spool/inbox"  # default


def test_sub_config_constructor_with_kwargs():
    """Sub-configs can be constructed with specific values."""
    d = DirsConfig(inbox="/custom/inbox", papers="/custom/papers")
    assert d.inbox == "/custom/inbox"
    assert d["papers"] == "/custom/papers"
    assert d.get("reports") == "data/reports"  # default


def test_config_isinstance_checks():
    """Config and sub-configs should be instances of their types."""
    c = Config()
    assert isinstance(c, Config)
    assert isinstance(c.dirs, DirsConfig)
    assert isinstance(c.api, ApiConfig)
