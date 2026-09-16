"""T47: the isolated local acceptance config resolves roles without secrets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from drbrain.config import Config
from drbrain.services.model_roles import (
    MODEL_ROLES,
    ROLE_CHAT,
    ROLE_EMBEDDING,
    ROLE_INDEX,
    ROLE_RERANK,
    resolve_model_role,
)

ACCEPTANCE_DIR = Path(__file__).resolve().parents[2] / "data" / "integration" / "unified-tree"
ACCEPTANCE_CONFIG = ACCEPTANCE_DIR / "config.yaml"

_KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(r"['\"][A-Za-z0-9+/]{40,}={0,2}['\"]"),  # long base64-looking literal
)


class TestEnvironmentEndpoints:
    def test_named_endpoint_key_resolves_from_the_environment(self, tmp_path, monkeypatch):
        """A ${ENV} endpoint key resolves at load time and stays redacted."""
        monkeypatch.setenv("DRBRAIN_TEST_DEEPSEEK_KEY", "sk-test-secret-123456")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
llm:
  endpoints:
    deepseek:
      provider: openai
      model: deepseek-flash
      api_key: "${DRBRAIN_TEST_DEEPSEEK_KEY}"
      base_url: "https://api.deepseek.com/v1"
  roles:
    chat_model: deepseek
""",
            encoding="utf-8",
        )
        cfg = Config.from_yaml(str(config_path), local_path=tmp_path / "missing.local.yaml")
        role = resolve_model_role(cfg, ROLE_CHAT)
        assert role.api_key == "sk-test-secret-123456"
        assert role.endpoint_name == "deepseek"
        assert role.redacted()["api_key"] == "[REDACTED]"
        assert "sk-test-secret-123456" not in str(role.redacted())


@pytest.mark.skipif(not ACCEPTANCE_CONFIG.is_file(), reason="local acceptance config absent")
class TestAcceptanceConfig:
    def _cfg(self) -> Config:
        return Config.from_yaml(
            str(ACCEPTANCE_CONFIG), local_path=ACCEPTANCE_DIR / "config.local.yaml"
        )

    def test_config_is_key_free(self):
        text = ACCEPTANCE_CONFIG.read_text(encoding="utf-8")
        for pattern in _KEY_PATTERNS:
            assert not pattern.search(text), f"key-like literal in {ACCEPTANCE_CONFIG}"

    def test_all_four_roles_resolve_to_the_expected_endpoints(self):
        cfg = self._cfg()
        for role_name in MODEL_ROLES:
            role = resolve_model_role(cfg, role_name)
            assert role.model, role_name
            # The credential may resolve from the environment, but the
            # serializable projection never carries its value.
            rendered = role.redacted()
            assert rendered["api_key"] in ("[REDACTED]", "not set"), role_name
            if len(role.api_key) >= 8:
                assert role.api_key not in str(rendered), role_name
        assert resolve_model_role(cfg, ROLE_INDEX).endpoint_name == "spark_local"
        assert "127.0.0.1:8010" in resolve_model_role(cfg, ROLE_INDEX).base_url
        assert resolve_model_role(cfg, ROLE_CHAT).endpoint_name == "deepseek"
        assert resolve_model_role(cfg, ROLE_EMBEDDING).endpoint_name == "bge_embed_cpu"
        assert resolve_model_role(cfg, ROLE_RERANK).endpoint_name == "bge_rerank_cpu"

    def test_unified_retrieval_defaults_are_configured(self):
        cfg = self._cfg()
        assert cfg.llamaindex.rag_engine == "sql"
        assert cfg.llamaindex.retrievers == ["bm25", "vector", "tree"]
        assert 8 <= cfg.llamaindex.context_docs <= 10
        assert 20 <= cfg.llamaindex.rerank_top_k <= 50
        assert cfg.llamaindex.tree_storage

    def test_paths_stay_inside_the_runtime_root(self):
        cfg = self._cfg()
        for value in (cfg.db.path, cfg.dirs.papers, cfg.dirs.cache, cfg.llamaindex.tree_storage):
            assert value and not Path(value).is_absolute(), value
