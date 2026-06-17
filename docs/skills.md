# Skills Reference

DrBrain ships 27 agent skills that wrap CLI commands for AI-assisted research workflows.
Each skill is a `SKILL.md` file in `skills/<name>/`, installed via `npx skills add`.

## Data In (7 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `paper-ingest` | `ingest`, `fetch`, `batch-fetch` | PDF download + ingest pipeline (MinerU → tree → paper record) |
| `ingest-link` | `ingest-link` | Web URL → external extractor → paper record |
| `import` | `import` | Zotero / BibTeX / Endnote import |
| `enrich` | `enrich` | CrossRef metadata backfill + scrub detection |
| `translate` | `translate` | LLM translation with resume support |
| `document` | `document` | Inspect Office docs (DOCX/PPTX/XLSX) |
| `pipeline` | `pipeline` | Chain ingest→build→embed→closure steps |

## KG Build (3 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `kg-build` | `build`, `embed`, `closure` | 5-stage LLM extraction + TransE embeddings + rule inference |
| `kg-reason` | `reason`, `ask` | LLM agent reasoning over KG (with bidirectional validation, workflows) |
| `index` | `index` | Rebuild BM25 search index |

## Query & Explore (4 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `paper-query` | `query`, `search`, `fsearch` | BM25 + graph-enhanced + federated search |
| `graph` | `graph neighbors/path/related/describe/query/traverse-from/export` | Graph traversal, query, and export subcommands |
| `explore` | `explore` | Literature discovery collections (JSONL silos) |
| `fsearch` | `fsearch` | Federated search: local DB + arXiv with annotation |

## Analysis & Cartography (4 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `research-analysis` | `analyze`, `seed`, `frontier`, `difficulty` | Seeds, causal chains, hypotheses, counterfactuals |
| `knowledge-cartography` | `evolve`, `descendants`, `landscape`, `paradigm`, `transfers`, `isomorphism` | Concept lineage, paradigm shifts, landscape mapping |
| `workspace-analysis` | `ws` + analysis commands | Workspace-scoped multi-paper analysis |
| `citation-tracking` | `citations`, `check-citations`, `lineage` | Citation graph + in-text verification + author lineage |

## Library Management (5 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `library-maintenance` | `list`, `show`, `stats`, `delete`, `report`, `clean` | Paper CRUD + DB statistics |
| `export` | `export` | BibTeX / RIS / Markdown export with citation styles |
| `proceedings` | `proceedings` | Conference proceedings management |
| `backup` | `backup`, `restore` | Local tar.gz + rsync remote backup and restore |
| `citation-styles` | `style` | Custom citation style management |

## Quality & Maintenance (2 skills)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `audit` | `audit`, `repair`, `check` | 15 quality rules + metadata repair + dependency check |
| `metrics` | `metrics` | User behavior analytics: keywords, reading trends |

## Patent Search (1 skill)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `patent-search` | `patent-search` | USPTO patent search (PPUBS free or ODP with API key) |

## Utility (1 skill)

| Skill | CLI Command(s) | What |
|-------|---------------|------|
| `show` | `show` | Single-paper detail view |

---

## Skill ↔ Command Mapping

```
paper-ingest     → ingest, fetch, batch-fetch
ingest-link      → ingest-link
import           → import
enrich           → enrich
translate        → translate
document         → document
pipeline         → pipeline
kg-build         → build, embed, closure
kg-reason        → reason, ask
index            → index
paper-query      → query, search, fsearch
graph            → graph neighbors/path/related/describe/query/traverse-from/export
explore          → explore
fsearch          → fsearch
research-analysis→ analyze, seed, frontier, difficulty
knowledge-cartography → evolve, descendants, landscape, paradigm, transfers, isomorphism
workspace-analysis → ws + analyze/evolve/landscape
citation-tracking → citations, check-citations, lineage
library-maintenance → list, show, stats, delete, report, clean
export           → export
proceedings      → proceedings
backup           → backup, restore
citation-styles  → style
audit            → audit, repair, check
metrics          → metrics
patent-search    → patent-search
show             → show
```
