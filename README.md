<div align="center">

# DrBrain

**Symbol-driven academic knowledge graph with lightweight vector retrieval.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Claude Code Skills](https://img.shields.io/badge/Claude_Code_Skills-DrBrain-purple.svg)](.claude/skills/)

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
pipx install drbrain     # or: uv tool install drbrain
drbrain setup
```

This creates `~/DrBrain/` as your library root (cross-platform: `~/DrBrain` on macOS/Linux,
`%USERPROFILE%/DrBrain` on Windows). `drbrain setup` detects your AI platforms and injects
agent entries so your coding agent can use DrBrain skills directly.

## What It Does

|  | Feature | Details |
|--|---------|---------|
| **Ingest** | PDF to structured knowledge | MinerU parsing → 5-source metadata cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv) → LLM tree structuring |
| **Build** | 5-stage concept extraction | Ontology extension → entity extraction (10-way concurrent) → relation extraction → coreference → iterative refinement |
| **Query** | BM25 + graph-enhanced search | Keyword search with multiplicative PageRank boost, directed graph traversal, hybrid ranking |
| **Knowledge Graph** | Rule-based closure | 8+4 inference rules, t-norm transitive grounding, TransE embeddings for link prediction |
| **Reasoning** | Symbol-driven discovery | Causal chains, confidence propagation, counterfactual analysis, cross-domain isomorphism, hypothesis generation |
| **Analyze** | Knowledge frontier reports | Research seeds, debate zones, technology cliffs, LLM executive summary |
| **Citations** | Multi-source expansion | Forward/backward citations, shared-reference analysis, citation verification against library |
| **Export** | BibTeX, RIS, Markdown | Full library or workspace export with complete venue metadata (journal, volume, pages) |
| **Import** | Zotero, BibTeX, Endnote | Web API + local SQLite for Zotero, XML/RIS for Endnote, BibTeX files |
| **Translate** | LLM paper translation | Placeholder-protected chunking, language detection, concurrent translation with resume |
| **Knowledge Genealogy** | Concept lineage + paper descendants | Concept evolution trees (evolve), academic offspring (descendants), paradigm shift detection, cross-domain migration discovery |
| **Fetch** | PDF acquisition from OA sources | 5-stage fallback (arXiv, OpenAlex, Unpaywall, direct DOI), institutional proxy support |
| **Audit** | Data quality scan | 15 severity-graded rules, PDF pre-validation, ingest quality gates |

## Documentation

- [Getting Started](docs/getting-started.md) -- from install to first query
- [CLI Reference](docs/cli-reference.md) -- all commands with examples
- [Architecture](docs/architecture.md) -- system design and reasoning modules
- [Contributing](docs/contributing.md) -- how to add commands, modules, and skills

## Works With Your Agent

DrBrain is designed to work through AI coding agents. `drbrain setup` injects the right
entry files into your project so your agent can discover and use skills directly.

| Agent / IDE | Entry injected |
|-------------|---------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `CLAUDE.md` + `.claude-plugin/` + `.mcp.json` |
| [Codex](https://openai.com/codex) / OpenClaw | `AGENTS.md` |
| [Cline](https://github.com/cline/cline) | `.clinerules` |
| [Qwen Code](https://github.com/QwenLM/qwen-code) | `.qwen/QWEN.md` |
| [Cursor](https://cursor.sh) | `.cursor/rules/drbrain.mdc` |
| [Windsurf](https://codeium.com/windsurf) | `.windsurfrules` |
| [GitHub Copilot](https://github.com/features/copilot) | `.github/copilot-instructions.md` |

Skills follow the open [AgentSkills.io](https://agentskills.io) standard. Claude Code users
also get `.claude-plugin/` + `.mcp.json` injected for full plugin integration.

## Configuration

`drbrain setup` walks you through the basics interactively:

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
