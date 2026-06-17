<div align="center">

# DrBrain

**Symbol-driven academic knowledge graph with lightweight vector retrieval.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Agent Skills](https://img.shields.io/badge/Agent_Skills-DrBrain-purple.svg)](skills/)

</div>

---

Your AI coding agent already reads code and writes code. DrBrain gives it a structured
knowledge graph of academic papers — so it can search literature, trace causal chains,
find research gaps, and infer new relationships through rule-based reasoning.

- Your paper library becomes a queryable knowledge graph with concept-level granularity.
- Reasoning is symbol-driven: closure rules, confidence propagation, counterfactuals.
- Lightweight vectors for retrieval: semantically-complete tree nodes only, never arbitrary chunks.
- Built for AI agents: every feature is accessible through the CLI that your agent already uses.

## Quick Start

```bash
# Source install (alpha)
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync && uv pip install -e .
drbrain setup

# Coming in beta:
# pipx install drbrain
# uv tool install drbrain
```

This creates `~/DrBrain/` as your library root (cross-platform: `~/DrBrain` on macOS/Linux,
`%USERPROFILE%/DrBrain` on Windows).

## What It Does

|  | Feature | Details |
|--|---------|---------|
| **Ingest** | PDF to structured knowledge | MinerU parsing → 5-source metadata cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv) → LLM tree structuring |
| **Build** | 5-stage concept extraction | Ontology extension → entity extraction (10-way concurrent) → relation extraction → coreference → iterative refinement |
| **Query** | BM25 + graph-enhanced search | Keyword search with multiplicative PageRank boost, directed graph traversal, hybrid ranking |
| **Knowledge Graph** | Rule-based closure | 8+4 inference rules, t-norm transitive grounding, TransE embeddings for link prediction |
| **Reasoning** | Symbol-driven discovery | Causal chains, confidence propagation, counterfactual analysis, cross-domain isomorphism, hypothesis generation |
| **Workflows** | 7 structured reasoning pipelines | review, gap-analysis, impact, compare, frontier, lineage, paradigm — symbolic steps + LLM synthesis |
| **Sessions** | Persistent reasoning context | DB-backed multi-turn sessions, build context injection, cross-invocation continuity |
| **Analyze** | Knowledge frontier reports | Research seeds, debate zones, technology cliffs, LLM executive summary |
| **Citations** | Multi-source expansion | Forward/backward citations, shared-reference analysis, citation verification against library |
| **Export** | BibTeX, RIS, Markdown with citation styles | 4 built-in styles (APA, Vancouver, Chicago, MLA) + custom styles, full library or workspace export with venue metadata |
| **Import** | Zotero, BibTeX, Endnote | Web API + local SQLite for Zotero, XML/RIS for Endnote, BibTeX files |
| **Translate** | LLM paper translation | Placeholder-protected chunking, language detection, concurrent translation with resume |
| **Knowledge Genealogy** | Concept lineage + paper descendants | Concept evolution trees (evolve), academic offspring (descendants), paradigm shift detection, cross-domain migration discovery |
| **Fetch** | PDF acquisition from OA sources | 5-stage fallback (arXiv, OpenAlex, Unpaywall, direct DOI), institutional proxy support |
| **Federated Search** | Local + arXiv with annotation | Cross-source search with automatic ingested status, DOI/arXiv cross-reference |
| **Patent Search** | USPTO PPUBS + ODP | Free (PPUBS) or API-key (ODP) patent search, application number lookup |
| **Pipeline** | Step chaining | Presets (full/quick/embed) and custom step lists for batch processing |
| **Proceedings** | Conference management | Create/list/show proceedings, associate papers by conference |
| **Explore** | Discovery collections | Lightweight JSONL-backed silos with keyword search for literature discovery |
| **Enrich** | CrossRef metadata backfill | Fill missing fields, detect scrub-worthy records |
| **Audit** | Data quality scan | 15 severity-graded rules, PDF pre-validation, ingest quality gates |
| **Backup** | tar.gz + rsync | Local archive backups and remote SSH rsync sync with configurable targets |
| **Document** | Office file inspection | Structured text summaries for DOCX/PPTX/XLSX without a GUI |
| **Metrics** | Usage analytics | Top search keywords, most-read papers, weekly trends |

## Documentation

- [Getting Started](docs/getting-started.md) -- from install to first query
- [CLI Reference](docs/cli-reference.md) -- all commands with examples
- [Configuration](docs/configuration.md) -- every setting, default, and provider template
- [Architecture](docs/architecture.md) -- system design and reasoning modules
- [API Reference](docs/api-reference.md) -- module-level function and class signatures
- [Workflows](docs/workflows.md) -- structured reasoning workflow user + developer guide
- [Sessions](docs/sessions.md) -- persistent Session Agent deep dive
- [Embedding](docs/embedding.md) -- local, openai-compat, and none providers
- [Troubleshooting](docs/troubleshooting.md) -- common problems and recovery
- [Glossary](docs/glossary.md) -- terminology reference
- [Skills Reference](docs/skills.md) -- 27 agent skills and their CLI commands
- [Contributing](docs/contributing.md) -- how to add commands, modules, and skills

## Works With Your Agent

Install DrBrain skills so your coding agent can use them:

```bash
npx skills add https://github.com/UynajGI/DrBrain/skills
```

Skills follow the open [AgentSkills.io](https://agentskills.io) standard and work with
Claude Code, Codex, Cline, Cursor, Windsurf, Qwen Code, GitHub Copilot, and other
AI coding tools.

## Configuration

`drbrain setup` walks you through the basics interactively (bilingual EN/ZH):

- Language selection at start (English / 中文)
- LLM API key (any litellm provider: OpenAI, Anthropic, Ollama, DeepSeek, etc.)
- MinerU token (optional; PyMuPDF fallback for PDF parsing)
- Semantic Scholar / CrossRef / OpenAlex API keys (optional; higher rate limits)
- `drbrain check` verifies your environment

## Inspired By

DrBrain's agent-first design is inspired by [ScholarAIO](https://github.com/ZimoLiao/scholaraio) — the
pioneering "research infrastructure for AI agents." DrBrain takes a different technical path:
symbol-driven knowledge graph reasoning with lightweight vector retrieval for semantically-complete nodes
(vs full-text chunk embedding). Tree-structured retrieval is inspired by
[PageIndex](https://github.com/answerdotai/pageindex) and [RAPTOR](https://arxiv.org/abs/2401.18059).

## License

MIT — see [LICENSE](LICENSE).
