"""T1 smoke tests: ``llamaindex:`` config section + Settings init (offline-safe).

Must pass with no network and no GPU. The enabled-path tests are skipped when
llama-index is not installed in the environment.
"""

import importlib.util

import pytest

from drbrain.config import Config, LlamaIndexConfig, LlamaIndexEvalConfig
from drbrain.rag.config import get_llamaindex_config
from drbrain.rag.llm import DrbrainEmbedding, init_llamaindex_settings

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None


# ── Config section defaults ──


def test_llamaindex_config_defaults():
    c = LlamaIndexConfig()
    assert c.enabled is False
    assert c.llm == "litellm"
    assert c.vector_store == "memory"
    assert c.storage_dir == "data/llamaindex"
    assert c.retrievers == ["bm25", "vector"]
    assert c.fusion_mode == "reciprocal_rank"
    assert c.rerank is True
    assert c.rerank_model == "Qwen/Qwen3-Reranker-0.6B"
    assert c.rerank_top_k == 20
    assert c.similarity_cutoff == 0.7
    assert c.streaming is True
    assert c.eval.golden_set == "data/llamaindex/golden.jsonl"
    assert c.eval.split == ["dev", "val", "test"]


def test_config_defaults_include_llamaindex():
    c = Config()
    assert isinstance(c.llamaindex, LlamaIndexConfig)
    assert isinstance(c.llamaindex.eval, LlamaIndexEvalConfig)
    assert c.llamaindex.enabled is False  # opt-in by default


# ── YAML parsing ──


def test_from_yaml_parses_llamaindex_section(tmp_path):
    base = tmp_path / "config.yaml"
    base.write_text(
        """
llamaindex:
  enabled: true
  vector_store: chroma
  rerank_top_k: 30
  eval:
    golden_set: /custom/golden.jsonl
    split: [dev, test]
"""
    )
    c = Config.from_yaml(base, local_path=tmp_path / "nonexistent.yaml")
    assert c.llamaindex.enabled is True
    assert c.llamaindex.vector_store == "chroma"
    assert c.llamaindex.rerank_top_k == 30
    assert c.llamaindex.eval.golden_set == "/custom/golden.jsonl"
    assert c.llamaindex.eval.split == ["dev", "test"]
    # Unset keys fall back to defaults
    assert c.llamaindex.fusion_mode == "reciprocal_rank"
    assert c.llamaindex.storage_dir == "data/llamaindex"


def test_from_yaml_missing_llamaindex_section(tmp_path):
    base = tmp_path / "config.yaml"
    base.write_text("db:\n  path: /custom/path.db\n")
    c = Config.from_yaml(base, local_path=tmp_path / "nonexistent.yaml")
    assert isinstance(c.llamaindex, LlamaIndexConfig)
    assert c.llamaindex.enabled is False
    assert c.llamaindex.retrievers == ["bm25", "vector"]


def test_real_config_has_llamaindex_section():
    c = Config.from_yaml("config.yaml")
    assert isinstance(c.llamaindex, LlamaIndexConfig)
    assert c.llamaindex.storage_dir == "data/llamaindex"
    assert isinstance(c.llamaindex.eval, LlamaIndexEvalConfig)


def test_get_llamaindex_config_tolerates_raw_dict():
    assert get_llamaindex_config({"llamaindex": {"enabled": True}}).enabled is True
    assert get_llamaindex_config({}).enabled is False
    # None → loads config.yaml from cwd (or defaults when absent)
    c = get_llamaindex_config(None)
    assert isinstance(c, LlamaIndexConfig)
    assert c.storage_dir == "data/llamaindex"


# ── init_llamaindex_settings ──


def test_init_settings_disabled_returns_false():
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=False))
    assert init_llamaindex_settings(cfg) is False


def test_init_settings_import_failure_returns_false(monkeypatch):
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))

    def _no_settings():
        return None

    monkeypatch.setattr("drbrain.rag.llm._import_settings", _no_settings)
    assert init_llamaindex_settings(cfg) is False


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_init_settings_enabled_returns_true():
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    assert init_llamaindex_settings(cfg) is True


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_init_settings_sets_embed_model_from_config():
    from llama_index.core import Settings

    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    assert init_llamaindex_settings(cfg) is True
    assert isinstance(Settings.embed_model, DrbrainEmbedding)
    # model name mirrors the configured embed provider model
    assert Settings.embed_model.model_name == cfg.embed.model


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_drbrain_embedding_is_lazy():
    """Constructing the adapter must not load models / touch the network."""
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    emb = DrbrainEmbedding(cfg)
    assert emb.model_name == cfg.embed.model
    assert emb.embed_batch_size == cfg.embed.batch_size
