# Skills Completeness + Docs

## A. New Skills (7)

Each skill follows existing pattern: `skills/<name>/SKILL.md` + register in `clawhub.yaml`.

| Skill | CLI backing | Description |
|-------|-----------|-------------|
| `drbrain/show` | `show` | View paper metadata, concepts, edges, arguments at any depth |
| `drbrain/export` | `export` | Export papers/workspace to BibTeX, RIS, Markdown |
| `drbrain/audit` | `audit` | Data quality scan with 15 severity-graded rules |
| `drbrain/translate` | `translate` | Translate paper markdown via LLM with resume |
| `drbrain/graph` | `graph neighbors/path/related` | Query knowledge graph — traverse, find paths, cross-paper analysis |
| `drbrain/import` | `import` | Import from Zotero Web API, local SQLite, BibTeX, Endnote XML/RIS |
| `drbrain/index` | `index` | Rebuild BM25 search index |

## B. Documentation

### `docs/getting-started.md`
- Prerequisites (Python 3.12+, git)
- Install (pip install + git clone both paths)
- `drbrain setup` walkthrough
- Drop PDF → ingest → build → query (first pipeline)
- Verify with `drbrain check`

### `docs/cli-reference.md`
- All CLI commands with:
  - What it does
  - Key flags/options
  - Example invocation
- Grouped by: Setup, Ingest, Build, Query, Graph, Analysis, Export/Import, Maintenance

### `docs/architecture.md`
- System overview (vector-free, symbol-driven)
- Ingestion pipeline (2-phase: ingest + build)
- Knowledge graph (TBox types, relations, closure rules)
- Reasoning modules (causal chains, confidence, counterfactuals, isomorphism, hypotheses)
- Data layout
- Key design decisions

### `docs/contributing.md`
- CONTRIBUTING.md at root already exists — `docs/contributing.md` should be deeper:
  - Codebase tour (key files and what they do)
  - How to add a new CLI command
  - How to add a new reasoning module
  - How to add a new skill
  - Testing patterns
  - Documentation standards

## C. Update clawhub.yaml

Register all 7 new skills.

## D. Update README.md

Add link to docs/ in README.

## Acceptance
- 7 new skill directories with SKILL.md
- 4 new doc files in docs/
- clawhub.yaml updated with 12 skills (5 existing + 7 new)
- Each skill SKILL.md: name + description frontmatter + usage instructions
- Each doc: detailed, example-driven, newcomer-friendly
