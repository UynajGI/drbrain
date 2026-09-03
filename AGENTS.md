## DrBrain — Project Context

DrBrain is a **symbol-driven academic knowledge graph with corpus-scale hybrid retrieval**. It ingests PDFs,
extracts structured concepts/arguments via LLM, deduplicates identities, and infers
new relationships through rule-based graph closure.

### Quick Reference

- CLI: `drbrain --help`
- Key commands: `setup`, `ingest`, `build`, `embed`, `closure`, `query`, `ask`, `reason`, `graph`, `analyze`, `evolve`, `landscape`, `frontier`, `citations`, `export`, `export-okf`, `ws`, `audit`
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
| `src/drbrain/cli/` | Typer CLI (main.py registration, *_commands.py modules, _common.py helpers, setup.py) |
| `src/drbrain/extractor/` | LLM extraction, reasoning, API clients (openalex, crossref) |
| `src/drbrain/graph/` | Graph engine, TransE embeddings (learn/predict/similar, incremental train), rule closure, query embeddings |
| `src/drbrain/storage/` | SQLite database (schema v15, centralized writes), BibTeX/RIS export, GraphML/JSON-LD/Cypher graph export, OKF v0.1 markdown export, workspace, paths, proceedings, explore silos, backup |
| `src/drbrain/rag/` | LlamaIndex RAG layer — BM25/vector/tree/graph/raptor fusion retrieval, FunctionAgent, rerank, eval, plus Epistemic Layer (RAGState / authority / status) |
| `src/drbrain/services/` | Embedding, audit, repair, enrich, translate, zotero import, citation_styles, document, fsearch, pipeline, metrics_panel, parser_benchmark |
| `src/drbrain/providers/` | Web extraction (qt-web-extractor), USPTO ODP + PPUBS patent search |
| `src/drbrain/parser/` | MinerU PDF parser, PageIndex tree parser |
| `src/drbrain/plugins/` | Model-as-Tool plugin interface — Plugin/PluginResult/PluginRegistry/discover (generic abstraction, concrete plugins load externally at runtime) |
| `src/drbrain/loop/` | Research loop — LlamaIndex Workflow 编排闭环 (13 节点 + agent-backed + 4 角色 analyst/critic/compute/verifier + 讨论层 discussion.py(消息板+非作者门+queue claim) + 互验/实算门 + 闭环沉淀) |
| `src/drbrain/query/` | BM25 search, RAPTOR two-stage tree traversal retrieval |
| `src/drbrain/report/` | Knowledge frontier analyzer |
| `scripts/pipeline/` | 全量语料增强管线（scibase/openalex 342k 篇）— ingest(build/rebuild_trees)、build(jsonl-out 并发)、load_build(_merge) 入库、embed_batch(本地 0.6B 多路)、vec_backfill/vec_quantize_int8(sqlite-vec)、launch_*.sh 启动器。走"先缓存后入库"：build 只写 jsonl，完成后统一入主库 |
| `scripts/serve_embedding.py` | 本地 Qwen3-Embedding-0.6B 常驻服务（openai-compat /v1/embeddings，max_seq_length=512，batch_size=8 防 OOM，GPU 绑卡） |
| `tests/` | pytest test suite |
| `skills/` | Project skills (AgentSkills.io standard, canonical source) |
| `.github/` | CI workflow, issue/PR templates |
