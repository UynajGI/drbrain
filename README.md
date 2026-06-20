<div align="center">

# 🧠 DrBrain

**Symbol-driven academic knowledge graph with lightweight vector retrieval.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0a3-orange.svg)](https://github.com/UynajGI/DrBrain/releases)
[![Agent Skills](https://img.shields.io/badge/Agent_Skills-DrBrain-purple.svg)](skills/)
[![CI](https://github.com/UynajGI/DrBrain/actions/workflows/ci.yml/badge.svg)](https://github.com/UynajGI/DrBrain/actions)

**[English](README.md)** · [简体中文](README.zh-CN.md)

</div>

---

Your AI coding agent already reads code and writes code. DrBrain gives it a
**structured knowledge graph of academic papers** — so it can search
literature, trace causal chains, find research gaps, and infer new
relationships through rule-based reasoning.

- 📚 Your paper library becomes a queryable knowledge graph with
  **concept-level granularity**.
- 🧩 Reasoning is **symbol-driven**: closure rules, confidence propagation,
  counterfactuals — not just embedding similarity.
- ⚡ Lightweight vectors for retrieval: **semantically-complete tree nodes**
  only, never arbitrary chunks.
- 🤖 **Built for AI agents**: every feature is accessible through the CLI
  that your agent already uses.

---

## ✨ Highlights

- **Incremental by default** — add one paper to an N-paper library and only
  that paper builds; closure scans its neighborhood; embeddings micro-adjust.
- **7 reasoning workflows** — review, gap-analysis, impact, compare,
  frontier, lineage, paradigm.
- **OKF export** — emit the entire knowledge graph as an
  [OKF](https://github.com/UynajGI/DrBrain) v0.1 markdown bundle that humans
  and agents can read with `cat`.
- **27 agent skills** following the open
  [AgentSkills.io](https://agentskills.io) standard.

---

## 🚀 Quick Start

```bash
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync && uv pip install -e .
drbrain setup          # interactive wizard (bilingual EN / 中文)
```

This creates `~/DrBrain/` as your library root
(`%USERPROFILE%/DrBrain` on Windows).

```bash
# Ingest → build → embed → closure (all incremental)
drbrain fetch "10.1038/nature14539"     # grab a paper by DOI
drbrain build                           # 5-stage LLM extraction
drbrain embed                           # TransE graph embeddings
drbrain closure                         # rule-based inference
drbrain ask "What gaps remain in deep learning?"

# Or chain everything at once
drbrain pipeline --preset full
```

> `pipx install drbrain` and `uv tool install drbrain` are coming in beta.

---

## 📖 What It Does

| Category | Feature | Details |
|----------|---------|---------|
| **Ingest** | PDF → structured knowledge | MinerU parsing → 5-source metadata cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv) → LLM tree structuring |
| **Build** | 5-stage concept extraction *(incremental)* | Ontology extension → entity extraction (10-way concurrent) → relation extraction → coreference → iterative refinement |
| **Query** | BM25 + graph-enhanced search | Keyword search with multiplicative PageRank boost, directed graph traversal, hybrid ranking |
| **Knowledge Graph** | Rule-based closure *(incremental)* | 8+4 inference rules, t-norm transitive grounding, TransE embeddings for link prediction |
| **Reasoning** | Symbol-driven discovery | Causal chains, confidence propagation, counterfactual analysis, cross-domain isomorphism, hypothesis generation |
| **Workflows** | 7 structured reasoning pipelines | review, gap-analysis, impact, compare, frontier, lineage, paradigm |
| **Sessions** | Persistent reasoning context | DB-backed multi-turn sessions, build context injection, cross-invocation continuity |
| **Analyze** | Knowledge frontier reports | Research seeds, debate zones, technology cliffs, LLM executive summary |
| **Citations** | Multi-source expansion | Forward/backward citations, shared-reference analysis, citation verification |
| **Export** | BibTeX, RIS, Markdown + **OKF** | 4 citation styles (APA, Vancouver, Chicago, MLA) + OKF v0.1 markdown bundle |
| **Import** | Zotero, BibTeX, Endnote | Web API + local SQLite for Zotero, XML/RIS for Endnote |
| **Genealogy** | Concept lineage + descendants | Evolution trees, academic offspring, paradigm shift detection, cross-domain migration |
| **Fetch** | PDF acquisition from OA sources | 5-stage fallback (arXiv, OpenAlex, Unpaywall, direct DOI), proxy support |
| **Federated Search** | Local + arXiv | Cross-source search with ingested-status annotation |
| **Patent Search** | USPTO PPUBS + ODP | Free (PPUBS) or API-key (ODP) patent search |
| **Pipeline** | Step chaining *(incremental)* | Presets (full/quick/embed) + custom steps; `--full` forces rebuild |
| **Audit** | Data quality scan | 15 severity-graded rules, PDF pre-validation, ingest quality gates |

<details>
<summary><b>All commands</b></summary>

`setup` `ingest` `fetch` `build` `embed` `closure` `query` `search` `ask`
`reason` `graph` `analyze` `evolve` `landscape` `frontier` `paradigm`
`citations` `export` `export-okf` `import` `translate` `session` `ws`
`pipeline` `repair` `enrich` `audit` `backup` `restore` `metrics`
`document` `fsearch` `patent-search` `proceedings` `explore` `check` `clean`

Run `drbrain --help` for the full list, or see the
[CLI Reference](docs/cli-reference.md).

</details>

---

## 🤖 Works With Your Agent

Install DrBrain skills so your coding agent can use them:

```bash
npx skills add https://github.com/UynajGI/DrBrain/skills
```

Skills follow the open [AgentSkills.io](https://agentskills.io) standard and
work with Claude Code, Codex, Cline, Cursor, Windsurf, Qwen Code, GitHub
Copilot, and other AI coding tools.

---

## ⚙️ Configuration

`drbrain setup` walks you through the basics interactively (bilingual
EN / 中文):

- LLM API key (any litellm provider: OpenAI, Anthropic, Ollama, DeepSeek, …)
- MinerU token (optional; PyMuPDF fallback for PDF parsing)
- Semantic Scholar / CrossRef / OpenAlex API keys (optional; higher rate
  limits)

`drbrain check` verifies your environment end-to-end. See
[Configuration](docs/configuration.md) for every setting.

---

## 📚 Documentation

| | |
|---|---|
| [Getting Started](docs/getting-started.md) | From install to first query |
| [CLI Reference](docs/cli-reference.md) | All commands with examples |
| [Architecture](docs/architecture.md) | System design and reasoning modules |
| [Configuration](docs/configuration.md) | Every setting, default, and provider template |
| [Workflows](docs/workflows.md) | Structured reasoning workflow guide |
| [Sessions](docs/sessions.md) | Persistent Session Agent deep dive |
| [Embedding](docs/embedding.md) | Local, openai-compat, and none providers |
| [Troubleshooting](docs/troubleshooting.md) | Common problems and recovery |
| [Skills Reference](docs/skills.md) | 27 agent skills and their CLI commands |
| [Contributing](docs/contributing.md) | How to add commands, modules, and skills |

---

## 🙏 Inspired By

DrBrain's agent-first design is inspired by
[ScholarAIO](https://github.com/ZimoLiao/scholaraio) — the pioneering
"research infrastructure for AI agents." DrBrain takes a different technical
path: symbol-driven knowledge graph reasoning with lightweight vector
retrieval for semantically-complete nodes (vs full-text chunk embedding).
Tree-structured retrieval is inspired by
[PageIndex](https://github.com/answerdotai/pageindex) and
[RAPTOR](https://arxiv.org/abs/2401.18059).

---

## 🤝 Contributing

Contributions are welcome — bug reports, feature requests, docs, and code
all help. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide.

Quick start for contributors:

```bash
git clone https://github.com/UynajGI/DrBrain.git && cd DrBrain
uv sync && uv pip install -e .
pre-commit install                      # optional: auto-lint on commit
uv run pytest -m "not integration"      # fast tests
```

Before opening a PR, make sure these pass:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/drbrain
uv run pytest -m "not integration"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

Have a question or idea? Open a
[discussion](https://github.com/UynajGI/DrBrain/discussions). Found a bug?
File an [issue](https://github.com/UynajGI/DrBrain/issues).

---

## 📄 License

MIT — see [LICENSE](LICENSE).
