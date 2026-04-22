# DrBrain MVP v1.0 Design Spec

> **Status:** Approved
> **Date:** 2026-04-22
> **Goal:** Pure CLI-driven academic knowledge graph. PDF ingest → cognitive map → research seeds.

## 1. Architecture Overview

DrBrain is a vector-free, symbol-driven research discovery engine. It maps academic papers to a knowledge graph of Problems, Methods, Gaps, Debates, Conclusions, and Actors. The full pipeline runs from the terminal.

**Pipeline:**
```
PDF → MinerU(Markdown) → Chapter Filter → LLM JSON Extraction
  → ID Dedup → Alias Canonicalization → SQLite Insert
  → Citation API Extension → Placeholder Nodes → Rule Closure
  → Seed Detection → JSON Report
```

**No vector databases, no embeddings, no web scraping.** All lookups are exact ID matching or LLM-structured extraction.

## 2. Tech Stack

| Module | Choice |
|--------|--------|
| Package manager | `uv` |
| Version control | `jj` (jujutsu, git colocate) |
| CLI framework | `typer` |
| PDF parsing | MinerU SDK (`from mineru import MinerU`) |
| LLM extraction | `litellm` with YAML fallback chain |
| Configuration | Flat YAML: `config.yaml` + `config.local.yaml` overlay |
| Storage | SQLite (5 core tables) |
| Graph computation | `networkx` (in-memory) |
| Search | BM25 (`rank-bm25`) |
| Frontend | None (v2) |

## 3. CLI Commands

| Command | Function | Input | Output |
|---------|----------|-------|--------|
| `drbrain setup` | Init DB, YAML, MinerU token guide | None | Interactive prompt, config files created |
| `drbrain ingest <path>` | Full pipeline ingest | PDF/MD path or dir | Terminal summary + JSON report |
| `drbrain expand --id <id>` | Citation topology extension | Paper local_id | JSONL of refs/cits, placeholders created |
| `drbrain list` | List all papers | None | Markdown table |
| `drbrain query <text>` | BM25 full-text search | Query string | Matching concepts + edges |
| `drbrain closure` | Rule-based closure engine | None | Inferred edge count |
| `drbrain seed` | Research seed detection | None | Structured seed list |
| `drbrain report --id <id>` | Single paper JSON report | Paper local_id | JSON to stdout or file |
| `drbrain export --id <id>` | Export report to file | Paper local_id | Saved JSON file path |
| `drbrain stats` | Graph statistics | None | Terminal summary table |

## 4. Configuration

Single flat YAML at project root. `config.yaml` is committed, `config.local.yaml` is gitignored and overlays on top.

```yaml
llm:
  models:
    - name: openai/gpt-4o
      api_key_env: OPENAI_API_KEY
      timeout: 30
      max_tokens: 4096
    - name: ollama/qwen2.5:14b
      api_base: http://localhost:11434/v1
      api_key_env: null
      timeout: 120
      max_tokens: 4096
  temperature: 0.1

mineru:
  token: null              # null = Flash mode (free, no auth)
  model: vlm               # pipeline / vlm / MinerU-HTML
  is_ocr: false
  enable_formula: true
  enable_table: true

db:
  path: data/drbrain.db

dirs:
  cache: data/cache
  reports: data/reports
  pdfs: data/pdfs
```

Loading: `config.yaml` first, then `config.local.yaml` deep-merges on top. `DRBRAIN_CONFIG` env var or `--config-path` flag overrides.

## 5. MinerU Integration

Three modes via `from mineru import MinerU`:

- **Flash mode:** `MinerU()` — free, no token, cloud-based, IP rate limited
- **Token mode:** `MinerU("token")` — registered token, higher limits, 1000 pages/day
- **Model choice:** `pipeline` (default), `vlm` (recommended, enhanced), `MinerU-HTML` (HTML only)

`setup` command guides the user: shows token申请 URL (`https://mineru.net/apiManage/token`), stores in `config.local.yaml`.

Fallback: if MinerU fails (network error, rate limit, token invalid), fall back to `pypdf` for basic text extraction.

## 6. ID Deduplication

Priority order: DOI > arXiv > S2 ID > OpenAlex ID > Title+Year (fuzzy)

| Key | Priority | Cleanup | Conflict |
|-----|----------|---------|----------|
| DOI | 1 | Strip `https://doi.org/`, lowercase | Absolute unique |
| arXiv ID | 2 | Strip `v\d+` suffix | DOI overrides |
| S2 Paper ID | 3 | Direct `paperId` | Auxiliary only |
| OpenAlex ID | 4 | Extract `W` digits | Auxiliary only |
| Title+Year | 5 | Strip articles/punctuation, Jaccard similarity | Mark `weak_match` if >0.85 |

## 7. SQLite Schema

**papers:** `local_id PK, title, year, status (uploaded/placeholder/merged), created_at`
**paper_ids:** `local_id FK, doi UNIQUE, arxiv UNIQUE, s2_id UNIQUE, openalex_id UNIQUE`
**concepts:** `concept_id PK AUTO, local_id FK, type, label, confidence`
**edges:** `src_id, dst_id, relation, source_paper, weight` — composite PK `(src,dst,rel,source)`
**aliases:** `variant PK, canonical_id FK→concepts`
**research_seeds:** `id PK AUTO, seed_type, node, signal, created_at`

## 8. Graph Engine

NetworkX in-memory graph loaded from SQLite.

**Closure rules:**
1. `creates_debate` — if two papers reach same Conclusion with opposite relations
2. `gap_addressed` — if a Method addressing a Problem also proposes a Conclusion that resolves a Gap
3. `indirect_evolution` — if Method A extends Method B which addresses Problem P, link A→P

**Seed patterns:**
1. `stale_problem` — Problem with in-degree ≥3, no recent addresses
2. `unaddressed_gap` — Gap with no incoming edges
3. `debate_zone` — Conclusion with both `supports` and `challenges` edges
4. `tech_discontinuity` — Method with historical activity then silence + related Gap
5. `cross_domain` — Two subgraphs sharing Problem/Metric but path length >3

## 9. BM25 Query

`rank-bm25` library. Index built on-the-fly from `concepts.label` + `papers.title`. Query returns ranked concept matches with paper context. No persistent index — rebuilt per query for MVP scale.

## 10. Output Protocol

All commands output machine-readable by default:
- Reports: JSON to `reports/<local_id>.json`
- Seeds: JSON array
- Lists: Markdown table to stdout
- Errors: JSONL to stderr
- `--json` flag forces JSON output where applicable

## 11. Error Handling

- MinerU failure → pypdf fallback → skip with warning
- LLM failure → skip concept extraction, log conflict
- API rate limit → exponential backoff, 3 retries → cache miss allowed
- DB write failure → transaction rollback, no partial state
- All errors logged to `data/logs/conflicts.jsonl`
