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

| Command | Purpose | E2E |
|---------|---------|-----|
| `drbrain setup` | Interactive config wizard + env init | ✅ |
| `drbrain check` | Full environment diagnostics + auto-fix | ✅ |
| `drbrain clean` | Clear data (db/cache/logs/papers/reports) | ✅ |
| `drbrain ingest` | Parse PDFs (metadata + tree) — lightweight | ✅ |
| `drbrain build` | 5-stage LLM graph extraction | ✅ |
| `drbrain embed` | TransE graph embeddings for link prediction | ✅ |
| `drbrain reason` | LLM agent reasoning over knowledge graph | ✅ |
| `drbrain query` | BM25 + `--hybrid` + `--neighbors` | ✅ |
| `drbrain graph neighbors` | Direct graph traversal from a node | ✅ |
| `drbrain graph path` | Shortest path between two nodes | ✅ |
| `drbrain graph related` | Shared concept analysis (concepts/graph/edges) | ✅ |
| `drbrain closure` | Rule-based inference (`--mode hybrid`, `--dry-run`) | ✅ |
| `drbrain seed` | Detect research seeds from graph patterns | ✅ |
| `drbrain citations` | Multi-source citation expansion (`--limit`/`--sort`) | ✅ |
| `drbrain check-citations` | Verify in-text citations against library | ✅ |
| `drbrain analyze` | Knowledge frontier report (`--papers`/`--query`/`--discover`) | ✅ |
| `drbrain ws` | Manage paper workspaces | — |
| `drbrain export` | Export to BibTeX/RIS/Markdown | — |
| `drbrain backup` | Create tar.gz backup | — |
| `drbrain repair` | Auto-fix metadata via CrossRef/arXiv | — |
| `drbrain import` | Import from Zotero or BibTeX | — |
| `drbrain translate` | Translate paper markdown via LLM | — |
| `drbrain serve` | Streamlit UI at http://127.0.0.1:8501 | — |

## Architecture

- **Parser**: MinerU CLI → PyMuPDF fallback, PDF → Markdown
- **Extractor**: 5-stage LLM pipeline (ontology→entities→relations→coreference→refine), 10-way concurrent
- **Dedup**: DOI → arXiv → S2 → OpenAlex → title+year fuzzy match
- **Graph**: NetworkX in-memory + SQLite, rule-based closure with 8+4 inference rules, directed BFS traversal with relation filtering, shortest-path queries
- **Reasoning**: Causal chains, counterfactual analysis, isomorphism detection, hypothesis generation
- **Skills**: 5 AgentSkills.io-compatible skills in `skills/`
