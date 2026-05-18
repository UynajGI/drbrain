# Setup-Config-Check Contract

> When setup.py gains a new config section, 5 other files must be updated. This doc prevents partial updates.

## Scope / Trigger

- Adding a new config section to `drbrain setup` (interactive wizard)
- Adding a new field to `generate_local_config()` signature
- Adding a new env var to `--quick` mode

## The 5-File Update Chain

Every new config key written by `setup.py` MUST be updated in:

| # | File | What to add |
|---|------|-------------|
| 1 | `check_commands.py` | Validation + display in "API Keys & Tokens" section |
| 2 | `test_setup.py` | `assert "<key>" in data` in `test_generate_local_config_writes_and_contains_keys` |
| 3 | `test_config.py` | `isinstance(c.<section>, <ConfigClass>)` in `test_config_defaults`, dot-access in `test_typed_dot_access` |
| 4 | `config.example.yaml` | Full section with comments and defaults |
| 5 | `src/drbrain/templates/agents/*.j2` | Any template referencing config keys or skill names |

Plus docs (if the feature is user-facing):
| 6 | `docs/cli-reference.md` | Command flags table update |
| 7 | `docs/configuration.md` | Config section docs |
| 8 | `docs/getting-started.md` | Pipeline steps if workflow changes |

## Contracts

### setup.py writes config.local.yaml

Three code paths, all must write identical keys:

```
generate_local_config()  → programmatic API (called by tests)
setup_cmd(quick=True)    → reads env vars, writes inline dict
setup_cmd(interactive)   → prompts user, writes inline dict
```

### config.py declares schema

```python
# src/drbrain/config.py
@dataclass
class EmbedConfig:
    provider: str = "local"      # local | openai-compat | none
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "auto"
    api_base: str = ""
    api_key: str = ""
    # ... more fields
```

### services/embedding.py reads at runtime

Must handle all provider values without crashing:
- `"local"` → load model via sentence-transformers
- `"openai-compat"` → POST to api_base/embeddings with api_key
- `"none"` → early return [], log info

## Validation & Error Matrix

| Condition | Error |
|-----------|-------|
| `embed.provider == "openai-compat"` and `api_base == ""` | `ValueError("embed.api_base is required")` |
| `embed.provider == "openai-compat"` and `api_key == ""` | `ValueError("embed.api_key is required")` |
| `embed.provider == "local"` and sentence-transformers not installed | `ImportError` at `_load_model()` |
| `embed.provider == "none"` | No error — all call sites handle gracefully |

## Good/Base/Bad Cases

### Good: Quick mode openai-compat with all env vars set
```bash
DRBRAIN_EMBED_PROVIDER=openai-compat \
DRBRAIN_EMBED_MODEL=text-embedding-3-small \
DRBRAIN_EMBED_API_KEY=sk-xxx \
DRBRAIN_EMBED_API_BASE=https://api.openai.com/v1 \
drbrain setup --quick
```
→ config.local.yaml has complete openai-compat section. `drbrain embed --tree` works.

### Base: Interactive mode with defaults
```bash
drbrain setup
# Accept all defaults → local provider, Qwen3-Embedding-0.6B
```
→ config.local.yaml has `embed: {provider: local, device: auto, model: Qwen/Qwen3-Embedding-0.6B}`.

### Bad: Quick mode openai-compat without api_base/api_key (FIXED 2026-05-18)
```bash
DRBRAIN_EMBED_PROVIDER=openai-compat drbrain setup --quick
# Before fix: api_base and api_key absent → ValueError at runtime
# After fix: always writes defaults (https://api.openai.com/v1 + ${OPENAI_API_KEY})
```

## Tests Required

- [x] `test_generate_local_config_writes_and_contains_keys` — asserts `"embed" in data`
- [x] `test_config_defaults` — asserts `isinstance(c.embed, EmbedConfig)`
- [x] `test_typed_dot_access` — asserts `c.embed.provider == "local"`, `c.embed.top_k == 10`
- [ ] Test quick mode with `DRBRAIN_EMBED_PROVIDER=openai-compat` env vars (not yet)
- [ ] Test quick mode with `DRBRAIN_EMBED_PROVIDER=none` (not yet)

## Wrong vs Correct

### Wrong: Adding a config section only to setup.py
```python
# setup.py only — check.py, tests, docs NOT updated
embed_cfg = {"provider": embed_provider, "device": "auto"}
config["embed"] = embed_cfg
```
→ `drbrain check` doesn't show embedding status. Tests fail to detect regressions. Docs are stale.

### Correct: Full chain update
1. Add `embed` prompts to setup.py interactive + quick modes
2. Add embedding validation to check_commands.py
3. Add `assert "embed" in data` to test_setup.py
4. Add `EmbedConfig` import + assertions to test_config.py
5. Update config.example.yaml embed section
6. Update CLI reference, configuration docs
