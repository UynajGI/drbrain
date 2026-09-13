"""Backend-independent retrieval contracts with SQL and LlamaIndex adapters.

Agent and model exports are loaded lazily; importing core retrieval or snapshot
utilities does not initialize the agent runtime."""

__all__ = ["build_agent", "init_llamaindex_settings", "reason_llamaindex"]


def __getattr__(name):
    """Keep the core and SQL backend importable without loading agent SDKs."""
    from importlib import import_module

    if name not in __all__:
        raise AttributeError(name)
    module = "llm" if name == "init_llamaindex_settings" else "agent"
    value = getattr(import_module(f"drbrain.rag.{module}"), name)
    globals()[name] = value
    return value
