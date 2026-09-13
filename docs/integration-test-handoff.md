# Integration-test handoff

This handoff describes the state of the local `phy` environment after merging the current autoresearch, RAG, capability, storage, and documentation work.

## Workspace

- Worktree: `/home/jiangyuan/drbrain-phy`
- Branch: `phy`
- Merge commit: created locally from `feat/autoresearch-orchestration`
- Knowledge-graph and WebUI behavior were not the subject of this handoff.

The worktree contained no local changes before the merge. Keep any integration-test configuration or generated data out of commits.

## Install and validate

```bash
cd /home/jiangyuan/drbrain-phy
uv sync
uv pip install -e .
uv run drbrain --help
uv run drbrain check
uv run ruff check src/ tests/
uv run mypy src/drbrain
```

The fast test suite excludes external integrations:

```bash
uv run pytest -m "not integration" -q
```

## Runtime setup

Use an isolated runtime root for integration data:

```bash
export DRBRAIN_ROOT=/tmp/drbrain-integration
uv run drbrain setup --quick
```

Configure model and provider credentials in the runtime-local `config.local.yaml` or environment variables. Never place credentials in the repository or in captured logs.

## Recommended integration sequence

1. **Ingest** a small PDF or Markdown fixture:

   ```bash
   uv run drbrain ingest path/to/paper.pdf
   ```

2. **Build** structured extraction:

   ```bash
   uv run drbrain build
   ```

3. **Build retrieval indexes** when an embedding provider is available:

   ```bash
   uv run drbrain embed --tree
   uv run drbrain rag index
   ```

4. **Exercise retrieval and sessions**:

   ```bash
   uv run drbrain query "research question"
   uv run drbrain reason "compare the methods in the corpus"
   uv run drbrain session new --title "integration smoke"
   ```

5. **Exercise the durable research loop** with an explicit bounded budget and, if enabled, one allowlisted capability:

   ```bash
   uv run drbrain autoresearch run "bounded integration objective" --json
   uv run drbrain autoresearch status "bounded integration objective" --json
   uv run drbrain autoresearch trace "bounded integration objective"
   uv run drbrain autoresearch audit "bounded integration objective"
   ```

6. **Verify backup and restore** into a fresh target:

   ```bash
   uv run drbrain backup
   uv run drbrain restore data/backups/<archive>.tar.gz --target /tmp/drbrain-restore
   ```

## Capability checks

The capability catalog is the integration boundary for Python plugins, models, APIs, CLI programs, MCP tools, and Skills. Verify discovery before running a loop:

```python
from drbrain.capabilities import CapabilityCatalog

catalog = CapabilityCatalog()
# Register adapters or PluginRegistry/MCP sources explicitly for the test.
print([descriptor.id for descriptor in catalog.list()])
```

For MCP, configure a trusted server and an explicit step allowlist. Skills are instruction packages and intentionally do not execute.

## What success means

- The source artifact remains in the runtime root after ingestion.
- Paper records and extraction results are persisted in SQLite.
- Retrieval results include source/section provenance; unavailable legs are reported as degraded.
- Capability calls return normalized status and provenance; duplicate IDs are rejected.
- Loop runs produce a durable run ID, ordered events, tool intent/observation records, and checkpoints.
- Claims are settled only after the configured evidence and verification gates pass.
- Backup restore validates its manifest and does not write internal metadata into the restored tree.

## Known boundaries

- A local smoke test does not establish provider quality or scientific answer quality.
- Real MCP, external API, GPU embedding, and long-running compute paths require separate credentials and fixtures.
- Knowledge-graph and WebUI integration remain outside this handoff.
- Do not treat generated reports, caches, or `/tmp` runtime data as source changes.

## Troubleshooting

- Run `uv run drbrain check` first.
- Confirm `DRBRAIN_ROOT` and database paths resolve inside the intended runtime root.
- If vector retrieval is unavailable, continue with lexical/tree retrieval and inspect the retrieval trace.
- If a loop cannot resume, inspect the ledger status, lease owner, checkpoint manifest, and strict RAG evidence setting.
- For provider failures, capture the sanitized error and model/provider name, never the credential.

The normative architecture and contracts are in `docs/architecture.md`, `docs/rag-layer-completion.md`, `docs/plugins.md`, and `docs/loop-current-state.md`.
