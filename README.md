# DrBrain — Academic Knowledge Graph System

Vector-free, symbol-driven research discovery engine. Ingest PDFs, build a knowledge graph,
and discover research hypotheses, causal chains, and knowledge frontier signals.

## Quick Start

```bash
uv sync
uv run drbrain check               # verify environment
uv run drbrain ingest               # ingest PDFs from data/spool/inbox/
uv run drbrain query "graph neural networks"
uv run drbrain analyze <id> --full  # knowledge frontier analysis
uv run drbrain serve                # launch Streamlit UI
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `drbrain ingest` | Parse and ingest PDFs into knowledge graph |
| `drbrain query` | Search concepts/arguments (BM25 + PageIndex tree retrieval) |
| `drbrain analyze` | Generate knowledge frontier report (seeds, chains, hypotheses) |
| `drbrain citations` | Query citation graph (refs, citing, shared-refs) |
| `drbrain check-citations <text>` | Verify in-text citations against library |
| `drbrain ws` | Manage paper workspaces |
| `drbrain export` | Export to BibTeX/RIS/Markdown |
| `drbrain backup` | Create tar.gz backup |
| `drbrain repair` | Auto-fix metadata via CrossRef/arXiv |
| `drbrain import` | Import from Zotero or BibTeX |
| `drbrain translate` | Translate paper markdown via LLM |
| `drbrain check` | Full environment diagnostics |
| `drbrain serve` | Web UI at http://127.0.0.1:8501 |

## Architecture

- **Parser**: MinerU CLI → PyMuPDF fallback, PDF → Markdown
- **Extractor**: LLM structured extraction with PageIndex tree-based concurrency
- **Dedup**: DOI → arXiv → S2 → OpenAlex → title+year fuzzy match
- **Graph**: NetworkX in-memory + SQLite, rule-based closure with 8 inference rules
- **Reasoning**: Causal chains, counterfactual analysis, isomorphism detection, hypothesis generation
- **Skills**: 5 AgentSkills.io-compatible skills in `skills/`
