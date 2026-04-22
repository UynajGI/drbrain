"""Tests for YAML config loader with local overlay."""
import tempfile
from pathlib import Path
from brbrain.config import load_config, merge_dicts

def test_load_defaults():
    """Load config from example YAML only, no local overlay."""
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
  path: data/drbrain.db
dirs:
  pdfs: data/pdfs
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
        cfg = load_config(base_path=base)
        assert cfg["llm"]["models"][0]["model"] == "gpt-4o"
        assert cfg["mineru"]["model"] == "vlm"
        assert cfg["db"]["path"] == "data/drbrain.db"
        assert cfg["bm25"]["k1"] == 1.5

def test_load_with_local_overlay():
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

def test_merge_dicts_nested():
    """merge_dicts does deep merge of nested dicts."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    result = merge_dicts(base, override)
    assert result["a"]["b"] == 10
    assert result["a"]["c"] == 2
    assert result["d"] == 3
    assert result["e"] == 4
