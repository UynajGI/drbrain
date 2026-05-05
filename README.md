<div align="center">

# DrBrain

**Vector-free, symbol-driven academic knowledge graph.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Claude Code Skills](https://img.shields.io/badge/Claude_Code_Skills-DrBrain-purple.svg)](.claude/skills/)

</div>

---

Your AI coding agent already reads code and writes code. DrBrain gives it a structured
knowledge graph of academic papers — so it can search literature, trace causal chains,
find research gaps, and infer new relationships through rule-based reasoning.

- Your paper library becomes a queryable knowledge graph with concept-level granularity.
- Reasoning is symbol-driven: closure rules, confidence propagation, counterfactuals — zero vectors required.
- Built for AI agents: every feature is accessible through the CLI that your agent already uses.

## Quick Start

```bash
pip install drbrain
drbrain setup
```

`drbrain setup` detects your AI platforms (Claude Code, Codex, Cursor, Cline, Windsurf, Qwen, Copilot)
and injects agent entries so your coding agent can use DrBrain skills directly.

Then drop PDFs into `data/spool/inbox/` and let your agent take it from there — or use the CLI yourself.

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
| **Audit** | Data quality scan | 15 severity-graded rules, PDF pre-validation, ingest quality gates |

## Works With Your Agent

DrBrain is designed to work through AI coding agents. `drbrain setup` injects the right
entry files into your project so your agent can discover and use skills directly.

| Agent / IDE | Entry injected |
|-------------|---------------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `CLAUDE.md` + `.claude-plugin/` + `.mcp.json` |
| [Codex](https://openai.com/codex) / OpenClaw | `AGENTS.md` |
| [Cline](https://github.com/cline/cline) | `.clinerules` |
| [Qwen](https://qwen.ai/) | `QWEN.md` |
| [Cursor](https://cursor.sh) | `.cursorrules` |
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
symbol-driven knowledge graph reasoning over vector-based semantic search.

## License

MIT — see [LICENSE](LICENSE).
