"""T17 contract tests: the single resolution entry point for the four roles.

Covers the frozen plan's required cases — same model name at different URLs,
different roles sharing one endpoint, legacy conflicts, missing roles — plus the
redaction contract: no API key may appear in ``redacted()``, ``repr()`` or the
``drbrain check`` rows.  No test touches the network.
"""

from __future__ import annotations

import json

import pytest

from drbrain.config import Config, LLMConfig, RetrievalConfig
from drbrain.services.model_roles import (
    MODEL_ROLES,
    ROLE_CHAT,
    ROLE_EMBEDDING,
    ROLE_INDEX,
    ROLE_RERANK,
    ModelRoleError,
    resolve_model_role,
    role_summary,
)

# A distinctive value so a leak is impossible to miss in serialized output.
KEY = "sk-unit-test-SECRET-4f9c2a"
OTHER_KEY = "sk-unit-test-SECRET-b7d18e"


def _llm_cfg(**overrides) -> dict:
    cfg = {
        "endpoints": {
            "spark_a": {
                "provider": "openai",
                "model": "spark-x25-4b",
                "api_key": KEY,
                "base_url": "http://127.0.0.1:8010/v1",
                "max_concurrent": 2,
            },
            "spark_b": {
                "provider": "openai",
                "model": "spark-x25-4b",
                "api_key": OTHER_KEY,
                "base_url": "http://127.0.0.1:8020/v1",
                "max_concurrent": 5,
            },
            "deepseek": {
                "provider": "openai",
                "model": "deepseek-flash",
                "api_key": KEY,
                "base_url": "https://api.deepseek.com/v1",
            },
        },
        "roles": {},
        "models": [
            {"provider": "openai", "model": "gpt-4o", "api_key": OTHER_KEY, "base_url": None}
        ],
        "index": [],
        "chat": [],
    }
    cfg.update(overrides)
    return cfg


def _retrieval_cfg(**overrides) -> dict:
    cfg = {
        "embed": "bge_embed_cpu",
        "rerank": "bge_rerank_cpu",
        "endpoints": {
            "bge_embed_cpu": {
                "provider": "local",
                "model": "BAAI/bge-small-en-v1.5",
                "device": "cpu",
            },
            "bge_rerank_cpu": {
                "provider": "local",
                "model": "BAAI/bge-reranker-base",
                "device": "cpu",
            },
        },
    }
    cfg.update(overrides)
    return cfg


def _cfg(llm: dict | None = None, retrieval: dict | None = None) -> dict:
    return {"llm": llm or _llm_cfg(), "retrieval": retrieval or _retrieval_cfg()}


# -- endpoint identity --------------------------------------------------------


def test_same_model_at_different_urls_resolves_to_different_endpoints():
    """A model name is not an endpoint: the URL decides which one serves it."""
    cfg = _cfg(_llm_cfg(roles={"index_model": "spark_a"}))
    first = resolve_model_role(cfg, ROLE_INDEX)
    cfg["llm"]["roles"] = {"index_model": "spark_b"}
    second = resolve_model_role(cfg, ROLE_INDEX)

    assert first.model == second.model == "spark-x25-4b"
    assert first.base_url != second.base_url
    assert first.endpoint_name != second.endpoint_name
    # Identity (used for the shared concurrency gate) is per endpoint, not per
    # model name — same model at another URL must not share a budget.
    assert first.identity != second.identity
    assert first.max_concurrent == 2
    assert second.max_concurrent == 5


def test_different_roles_may_share_one_endpoint():
    """index and chat on the same named endpoint keep the same binding."""
    cfg = _cfg(_llm_cfg(roles={"index_model": "spark_a", "chat_model": "spark_a"}))
    index_role = resolve_model_role(cfg, ROLE_INDEX)
    chat_role = resolve_model_role(cfg, ROLE_CHAT)

    assert index_role.identity == chat_role.identity
    assert index_role.base_url == chat_role.base_url == "http://127.0.0.1:8010/v1"
    assert index_role.api_key == chat_role.api_key
    assert index_role.source == chat_role.source == "roles"
    assert index_role.host == "127.0.0.1:8010"


def test_typed_config_and_dict_config_resolve_alike():
    """The typed Config dataclasses are a supported (not parallel) input."""
    typed = Config(
        llm=LLMConfig(
            endpoints={
                "spark_a": {
                    "provider": "openai",
                    "model": "spark-x25-4b",
                    "api_key": KEY,
                    "base_url": "http://127.0.0.1:8010/v1",
                    "max_concurrent": 2,
                }
            },
            roles={"index_model": "spark_a"},
        ),
        retrieval=RetrievalConfig(
            embed="bge_embed_cpu",
            endpoints={"bge_embed_cpu": {"provider": "local", "model": "BAAI/bge-small-en-v1.5"}},
        ),
    )
    role = resolve_model_role(typed, ROLE_INDEX)
    assert role.endpoint_name == "spark_a"
    assert role.max_concurrent == 2
    assert resolve_model_role(typed, ROLE_EMBEDDING).model == "BAAI/bge-small-en-v1.5"
    assert resolve_model_role(typed, ROLE_EMBEDDING).source == "endpoint"


# -- precedence and conflicts -------------------------------------------------


def test_explicit_index_chain_never_falls_back_to_generic_models():
    """`llm.models` must not silently build the index role (plan T17)."""
    cfg = _cfg(_llm_cfg(models=[{"provider": "openai", "model": "gpt-4o", "api_key": KEY}]))
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(cfg, ROLE_INDEX)
    assert "llm.models" in str(excinfo.value)
    assert "index" in str(excinfo.value)


def test_legacy_conflict_between_roles_and_index_chain_raises():
    """Two declaration paths that disagree must raise, not pick one silently."""
    llm = _llm_cfg(
        roles={"index_model": "spark_a"},
        index=[
            {
                "provider": "openai",
                "model": "spark-x25-4b",
                "api_key": OTHER_KEY,
                "base_url": "http://127.0.0.1:9999/v1",
            }
        ],
    )
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(_cfg(llm), ROLE_INDEX)
    message = str(excinfo.value)
    assert "conflicting definitions" in message
    assert "127.0.0.1:8010" in message and "127.0.0.1:9999" in message
    assert "remove one definition" in message


def test_agreeing_declarations_are_not_a_conflict():
    """The shipped config style — role alias plus an equivalent chain — is fine."""
    llm = _llm_cfg(
        roles={"pageindex_index": "spark_a"},
        index=[
            {
                "provider": "openai",
                "model": "spark-x25-4b",
                "api_key": KEY,
                "base_url": "http://127.0.0.1:8010/v1",
                "max_concurrent": 1,
            }
        ],
    )
    role = resolve_model_role(_cfg(llm), ROLE_INDEX)
    assert role.source == "roles"  # the routing table wins, the chain only agrees
    assert role.endpoint_name == "spark_a"


def test_conflicting_legacy_aliases_raise():
    """pageindex_index vs the canonical index_model alias must agree."""
    llm = _llm_cfg(roles={"index_model": "spark_a", "pageindex_index": "spark_b"})
    with pytest.raises(ModelRoleError, match="conflicting definitions"):
        resolve_model_role(_cfg(llm), ROLE_INDEX)


def test_missing_role_raises_with_actionable_hint():
    cfg = _cfg(_llm_cfg())
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(cfg, ROLE_INDEX)
    assert "llm.roles.index_model" in str(excinfo.value)

    # Retrieval roles with no registered endpoint are just as explicit.
    unregistered = _cfg(
        _llm_cfg(),
        retrieval={"embed": "bge_embed_cpu", "rerank": "bge_rerank_cpu", "endpoints": {}},
    )
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(unregistered, ROLE_RERANK)
    assert "retrieval.rerank" in str(excinfo.value)


def test_unknown_endpoint_name_raises():
    cfg = _cfg(_llm_cfg(roles={"chat_model": "ghost"}))
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(cfg, ROLE_CHAT)
    assert "ghost" in str(excinfo.value)
    assert "llm.endpoints" in str(excinfo.value)


def test_unknown_role_name_raises():
    with pytest.raises(ModelRoleError, match="unknown model role"):
        resolve_model_role(_cfg(), "indexmodel")


def test_chat_chain_is_authoritative_over_generic_models():
    """A configured chat chain wins; `llm.models` is legacy-only."""
    llm = _llm_cfg(
        chat=[
            {
                "provider": "openai",
                "model": "deepseek-flash",
                "api_key": KEY,
                "base_url": "https://api.deepseek.com/v1",
            }
        ]
    )
    role = resolve_model_role(_cfg(llm), ROLE_CHAT)
    assert role.source == "chat"
    assert role.model == "deepseek-flash"
    assert role.base_url == "https://api.deepseek.com/v1"
    assert role.host == "api.deepseek.com"


def test_generic_models_remain_the_documented_legacy_chat_default():
    role = resolve_model_role(_cfg(_llm_cfg()), ROLE_CHAT)
    assert role.source == "models"
    assert role.model == "gpt-4o"
    assert role.endpoint_name == ""


def test_embedded_roles_resolve_from_retrieval_endpoints():
    cfg = _cfg()
    embed = resolve_model_role(cfg, ROLE_EMBEDDING)
    rerank = resolve_model_role(cfg, ROLE_RERANK)
    assert embed.endpoint_name == "bge_embed_cpu"
    assert embed.model == "BAAI/bge-small-en-v1.5"
    assert embed.source == "endpoint"
    assert embed.is_local and not embed.requires_api_key
    assert rerank.model == "BAAI/bge-reranker-base"

    # An explicit llm.roles entry outranks the retrieval registration.
    cfg["llm"]["roles"] = {"embedding_model": "bge_embed_cpu"}
    assert resolve_model_role(cfg, ROLE_EMBEDDING).source == "roles"


def test_retrieval_role_naming_a_missing_endpoint_raises():
    cfg = _cfg(retrieval=_retrieval_cfg(embed="not_registered"))
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(cfg, ROLE_EMBEDDING)
    message = str(excinfo.value)
    assert "retrieval.embed" in message and "not_registered" in message
    assert "retrieval.endpoints" in message


def test_key_pool_only_endpoint_raises_instead_of_silently_using_one_key():
    llm = _llm_cfg(
        endpoints={
            "pool": {
                "provider": "openai",
                "model": "spark-x25-4b",
                "api_keys": [KEY, OTHER_KEY],
                "base_url": "http://127.0.0.1:8010/v1",
            }
        },
        roles={"index_model": "pool"},
    )
    with pytest.raises(ModelRoleError) as excinfo:
        resolve_model_role(_cfg(llm), ROLE_INDEX)
    assert "api_keys" in str(excinfo.value)


# -- redaction ----------------------------------------------------------------


def test_key_never_appears_in_serialized_role_output():
    cfg = _cfg(_llm_cfg(roles={"index_model": "spark_a", "chat_model": "deepseek"}))
    for role_name in MODEL_ROLES:
        role = resolve_model_role(cfg, role_name)
        for rendered in (
            repr(role),
            str(role),
            json.dumps(role.redacted(), default=str),
            json.dumps(role_summary(cfg, role_name), default=str),
        ):
            assert KEY not in rendered
            assert OTHER_KEY not in rendered
    assert resolve_model_role(cfg, ROLE_INDEX).api_key == KEY  # accessible to clients
    assert resolve_model_role(cfg, ROLE_INDEX).redacted()["api_key"] == "[REDACTED]"


def test_check_rows_show_all_four_roles_without_keys():
    """`drbrain check` renders roles + host but never the credential (T17/T47)."""
    from drbrain.cli.check_commands import _model_role_rows

    cfg = _cfg(_llm_cfg(roles={"index_model": "spark_a", "chat_model": "deepseek"}))
    rows, warnings = _model_role_rows(cfg, probe=False)

    labels = [label.strip() for label, _, _ in rows]
    assert labels == list(MODEL_ROLES)
    rendered = "\n".join(f"{label} {status} {detail}" for label, status, detail in rows)
    rendered += "\n" + "\n".join(warnings)
    assert KEY not in rendered and OTHER_KEY not in rendered
    assert "127.0.0.1:8010" in rendered  # host shown
    assert "api.deepseek.com" in rendered
    assert "not configured" not in rendered  # all four resolve in this config


def test_check_rows_report_unresolved_roles_as_warnings_without_probing():
    """An index role that no path configures is reported, not probed, not fatal.

    The legacy ``llm.models`` chat fallback is reported as ``source=models`` and
    is deliberately never probed here (the LLM-connectivity section covers it),
    which also keeps this test hermetic.
    """
    from drbrain.cli.check_commands import _model_role_rows

    cfg = _cfg(_llm_cfg())
    rows, warnings = _model_role_rows(cfg)
    index_row = next(row for row in rows if row[0].strip() == ROLE_INDEX)
    chat_row = next(row for row in rows if row[0].strip() == ROLE_CHAT)
    assert "not configured" in index_row[1]
    assert "source=models" in chat_row[2]
    assert any(ROLE_INDEX in warning for warning in warnings)
    assert not any("probe" in warning for warning in warnings)


def test_missing_credential_reason_flags_env_placeholders():
    """An unresolved ${ENV} marker is reported instead of being sent as a key."""
    llm = _llm_cfg(
        endpoints={
            "deepseek": {
                "provider": "openai",
                "model": "deepseek-flash",
                "api_key": "${DEEPSEEK_API_KEY}",
                "base_url": "https://api.deepseek.com/v1",
            }
        },
        roles={"chat_model": "deepseek"},
    )
    role = resolve_model_role(_cfg(llm), ROLE_CHAT)
    assert role.missing_credential_reason
    assert "DEEPSEEK_API_KEY" in role.missing_credential_reason
    assert not role.usable_api_key
    # loopback + keyless providers are not treated as broken credentials
    local = resolve_model_role(_cfg(_llm_cfg(roles={"index_model": "spark_a"})), ROLE_INDEX)
    assert local.usable_api_key
