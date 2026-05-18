<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

## Subagents

- ALWAYS wait for all subagents to complete before yielding.
- Spawn subagents automatically when:
  - Parallelizable work (e.g., install + verify, npm test + typecheck, multiple tasks from plan)
  - Long-running or blocking tasks where a worker can run independently.
  - Isolation for risky changes or checks

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->

## DrBrain — Project Context

DrBrain is a **symbol-driven academic knowledge graph with lightweight vector retrieval**. It ingests PDFs,
extracts structured concepts/arguments via LLM, deduplicates identities, and infers
new relationships through rule-based graph closure.

### Quick Reference

- CLI: `drbrain --help`
- Key commands: `setup`, `ingest`, `build`, `embed`, `closure`, `query`, `ask`, `reason`, `graph`, `analyze`, `evolve`, `landscape`, `frontier`, `citations`, `ws`, `audit`
- Skills: `skills/*/SKILL.md` (27 total) — paper-ingest, kg-build, kg-reason, paper-query, knowledge-cartography, graph, research-analysis, citation-tracking, workspace-analysis, library-maintenance, audit, export, import, index, show, translate, citation-styles, backup, document, patent-search, pipeline, fsearch, proceedings, explore, enrich, metrics, ingest-link
- Data: `data/spool/inbox/`, `data/papers/`, `workspace/`
- Tests: `uv run pytest -m "not integration"` (fast), `uv run pytest` (all)
- Lint: `uv run ruff check . && uv run ruff format .`

### How To Work In This Repo

- Prefer project skills in `skills/` when the user request matches one.
- Use the `drbrain` CLI instead of describing what should be done.
- Define verifiable success criteria before implementing. Write the test first, then make it pass.
- Match existing code style; don't refactor adjacent code unless the task requires it.

### Repo Map

| Directory | Purpose |
|-----------|---------|
| `src/drbrain/cli/` | Typer CLI (main.py registration, 9 *_commands.py modules, _common.py helpers, setup.py) |
| `src/drbrain/extractor/` | LLM extraction, reasoning, API clients (openalex, crossref) |
| `src/drbrain/graph/` | Graph engine, TransE embeddings (learn/predict/similar), rule closure, query embeddings |
| `src/drbrain/storage/` | SQLite database, export, workspace, paths, proceedings, explore silos |
| `src/drbrain/services/` | Embedding, audit, repair, enrich, translate, zotero import, citation_styles, document, fsearch, pipeline, metrics_panel, parser_benchmark |
| `src/drbrain/providers/` | Web extraction (qt-web-extractor), USPTO ODP + PPUBS patent search |
| `src/drbrain/parser/` | MinerU PDF parser, PageIndex tree parser |
| `src/drbrain/query/` | BM25 search, RAPTOR two-stage tree traversal retrieval |
| `src/drbrain/report/` | Knowledge frontier analyzer |
| `tests/` | pytest test suite |
| `skills/` | Project skills (AgentSkills.io standard, canonical source) |
| `.github/` | CI workflow, issue/PR templates |
