# DrBrain v1 — Academic Knowledge Graph (CLI-Driven)

> **Design Date:** 2026-04-22
> **Status:** Approved
> **Scope:** v1.0 — pure CLI, no frontend

---

## 1. Core Purpose

Engineer the process of "a PhD student spending 2-3 years exploring domain context, locating research boundaries, discovering innovation entry points" into an **automatically buildable, queryable, boundary-reasoning lightweight cognitive operating system**.

## 2. Core Principles

1. **User-driven full text**: PDF provided by user, avoids copyright/anti-scraping
2. **API-driven topology**: Citation network auto-expanded via public APIs, placeholder nodes maintain connectivity
3. **Zero vector dependency**: Entire pipeline uses exact ID matching, symbolic rule closure, LLM structured extraction
4. **Cognitive ontology first**: Nodes/edges map directly to academic cognitive structure (Problem/Method/Gap/Debate/Actor), not generic triplets
5. **CLI & LLM-Code friendly**: Pure terminal interaction, standardized JSON/Markdown output
6. **Schema-first validation**: TBox/RBox constraints filter LLM hallucination before ingestion
7. **Confidence queue**: Single-model extraction + confidence threshold + human review, no expensive multi-model voting
8. **Argument-centric**: Store argument structure (claim + evidence + target), not just flat concept lists
9. **Temporal evolution**: Track concept definitions and usage across time, detect shifts and obsolescence

## 3. Technology Stack

| Module | Choice | Notes |
|--------|--------|-------|
| **PDF parsing** | mineru-open-api CLI (`-o` output dir) | Outputs Markdown + images to temp directory. Images copied to `data/papers/images/<local_id>/`, MD saved to `data/papers/<local_id>.md` with rewritten refs. Flash mode (free, 10MB) + Token mode (registered, 200MB). CLI auto-retries on failure with exponential backoff, falls back to pypdfium2. |
| **LLM extraction** | litellm with YAML fallback chain | Single model per paper, ordered chain for failover |
| **DOI enrichment** | CrossRef API (polite pool) + S2 externalIds | Fallback: if S2 returns 429 or no DOI, try CrossRef via title search |
| **Schema validation** | Python symbolic rule engine | TBox type constraints + RBox relation restrictions, no external reasoner |
| **Graph computation** | NetworkX (in-memory) | Centrality, connected components, path queries, pattern matching |
| **Full-text search** | rank-bm25 | BM25 ranking over paper titles, abstracts, concept labels, argument claims |
| **Storage** | SQLite (single file) | 8 core tables: papers, paper_ids, concepts, arguments, edges, aliases, confidence_queue, research_seeds |
| **CLI framework** | typer | Auto-generated help, subcommands, typed arguments/options |
| **Package manager** | uv | pyproject.toml, uv.lock, uv run |
| **Version control** | jj (jujutsu) with git colocate | `jj git init --colocate` |
| **Configuration** | Flat YAML (`config.yaml` + `config.local.yaml` overlay) | pyyaml, local overrides main; includes `api.crossref_email` and `api.openalex_token` |
| **Testing** | pytest + `integration` marker | `-m "not integration"` runs fast unit tests; `-m integration` runs real PDFs from `data/pdfs/` |

## 4. CLI Commands

| Command | Function | Input | Output |
|---------|----------|-------|--------|
| `drbrain setup` | Interactive 16-step configuration wizard | None | Writes `config.local.yaml` |
| `drbrain ingest [<pdf> ...]` | Parse PDF(s), extract concepts+arguments, validate, ingest | PDF path(s) or directory | Structured JSON report + terminal summary; `--json` outputs machine-readable result to stdout |
| `drbrain expand --id <local_id> --depth 2` | Pull references/citations, create placeholders | Node ID, depth | Citation marking JSON + topology update log |
| `drbrain report --id <local_id>` | Generate single-paper report (coverage/blind spots) | Node ID | `reports/<id>.json` + terminal Markdown table |
| `drbrain seed` | Detect knowledge boundaries from argument patterns | None | `seeds.json` + terminal structured list |
| `drbrain query --type Problem --bm25 "attention"` | Query concepts/arguments with BM25 + type filter | Query params | JSONL result stream |
| `drbrain closure` | Run rule engine (transitive closure, debate binding, gap detection) | None | Closure update log |
| `drbrain queue` | Review confidence queue (human-in-the-loop) | None | Terminal table of pending items |
| `drbrain queue resolve --id <qid> --accept` | Accept/reject a queue item | Queue item ID | Updates aliases/concepts, removes from queue |
| `drbrain timeline --concept "transformer"` | Show concept evolution over time | Concept label | Terminal timeline + year-by-year stats |
| `drbrain list` | List all papers in database | None | Terminal table |
| `drbrain stats` | Database statistics (nodes, edges, coverage, queue depth) | None | Terminal summary |
| `drbrain export --format json` | Export graph data | Format | JSON/GraphML output |

## 5. Configuration (Flat YAML)

`config.yaml` (template) + `config.local.yaml` (user overrides, gitignored).

```yaml
# config.yaml — annotated template
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: "${OPENAI_API_KEY}"
      base_url: null
    - provider: ollama
      model: qwen2.5:7b
      api_key: null
      base_url: "http://localhost:11434"

mineru:
  token: "${MINERU_TOKEN}"  # empty = flash mode
  model: "vlm"              # pipeline | vlm | MinerU-HTML
  is_ocr: false
  enable_formula: true
  enable_table: true

db:
  path: "data/drbrain.db"

dirs:
  pdfs: "data/pdfs"
  reports: "data/reports"
  cache: "data/cache"
  logs: "data/logs"

api:
  s2_rate_limit: 100        # requests per minute
  cache_ttl: 86400          # 24h local cache
  crossref_email: ""        # polite pool email for DOI enrichment
  openalex_token: ""        # optional OpenAlex token

bm25:
  k1: 1.5
  b: 0.75

# Confidence queue thresholds
queue:
  weak_threshold: 0.7       # confidence below this goes to queue
  auto_accept: 0.9          # confidence above this auto-accepted
```

Local overlay (`config.local.yaml`) only contains keys the user wants to override.

## 6. Setup Wizard (16 Steps)

`drbrain setup` — interactive wizard with `typer.prompt()` / `typer.confirm()`:

1. **LLM primary model** — provider, model name, API key, base_url
2. **LLM fallback model** — optional secondary (provider/model/key/url)
3. ~~LLM third model~~ (removed)
4. **MinerU mode** — token mode or flash (free) mode
5. **MinerU token** — if token mode, input token (with link to https://mineru.net/apiManage/token)
6. **MinerU model** — pipeline (default) / vlm (recommended) / MinerU-HTML
7. **MinerU OCR** — enable/disable OCR extraction
8. **MinerU formula** — enable/disable formula parsing
9. **MinerU table** — enable/disable table parsing
10. **Database path** — default `data/drbrain.db`
11. ~~PDF storage directory~~ (removed, uses config default)
12. ~~Reports directory~~ (removed, uses config default)
13. ~~Cache directory~~ (removed, uses config default)
14. **S2 API rate limit** — default 100 req/min
15. **External APIs** — CrossRef email (polite pool), OpenAlex token (anonymous if empty)
16. **BM25 k1/b parameters** — default 1.5 / 0.75

Each step shows current default, allows skip (Enter for default), validates input where applicable. Writes to `config.local.yaml`.

## 7. MinerU Integration

**Implementation: mineru-open-api CLI with `-o` output directory (not stdout).**

```bash
# Output to temp directory (captures both Markdown AND images)
mineru-open-api extract paper.pdf --model vlm --token <token> -o /tmp/mineru_xxx/out/
```

Python wrapper (`MinerUParser`):
- Creates temp output dir, invokes CLI with `-o <temp_dir>`
- Reads generated `.md` from output dir after successful run
- Extracts `images/` subdirectory for downstream processing
- On non-zero return code, retries up to `max_retries` times with exponential backoff
- After all retries fail, falls back to `pypdfium2` for basic text extraction
- Images are saved to `data/papers/images/<local_id>/`, MD refs rewritten to point there
- Raw MD saved to `data/papers/<local_id>.md` for inspection

## 8. Triple ID Dedup Strategy

| Match key | Priority | Cleaning rules | Conflict handling |
|-----------|----------|----------------|-------------------|
| DOI | 1 (highest) | Strip `https://doi.org/`, lowercase, no spaces | Absolute unique key, merge on hit |
| arXiv ID | 2 | Strip `v\d+` version suffix, normalize to `YYMM.NNNNN` | DOI overrides arXiv if conflict |
| S2 Paper ID | 3 | Direct `paperId` field | Auxiliary only, doesn't override |
| OpenAlex ID | 4 | Strip `W` prefix, digits only | Auxiliary, fill missing metadata |
| Title + Year | 5 (fallback) | Strip articles/punctuation/case, compute edit distance | Threshold >0.85, mark `weak_match` for manual review |

Priority order prevents split identities. Conflict resolution: DOI > arXiv > S2 > OpenAlex > title/year.

## 9. SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS papers (
    local_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    status TEXT NOT NULL DEFAULT 'placeholder'
        CHECK(status IN ('uploaded', 'placeholder', 'merged')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_ids (
    local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    doi TEXT UNIQUE,
    arxiv TEXT UNIQUE,
    s2_id TEXT UNIQUE,
    openalex_id TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_id TEXT NOT NULL REFERENCES papers(local_id),
    type TEXT NOT NULL CHECK(type IN ('Problem', 'Method', 'Conclusion', 'Debate', 'Gap', 'Actor')),
    label TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    first_seen INTEGER,
    last_seen INTEGER
);

CREATE TABLE IF NOT EXISTS arguments (
    arg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL REFERENCES papers(local_id),
    claim TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK(claim_type IN ('supports', 'challenges', 'extends', 'limits', 'solves', 'proposes')),
    target_label TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK(target_type IN ('Method', 'Problem', 'Conclusion', 'Gap', 'Debate', 'Argument')),
    evidence_type TEXT CHECK(evidence_type IN ('empirical', 'theoretical', 'case_study', 'survey')),
    evidence_detail TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, relation, source_paper)
);

CREATE TABLE IF NOT EXISTS aliases (
    variant TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS confidence_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL,
    item_type TEXT NOT NULL CHECK(item_type IN ('concept', 'alias', 'relation')),
    item_data TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_seeds (
    seed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type);
CREATE INDEX IF NOT EXISTS idx_concepts_label ON concepts(label);
CREATE INDEX IF NOT EXISTS idx_concepts_first_seen ON concepts(first_seen);
CREATE INDEX IF NOT EXISTS idx_arguments_source ON arguments(source_paper);
CREATE INDEX IF NOT EXISTS idx_arguments_target ON arguments(target_label);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON confidence_queue(status);
```

**Key additions vs original schema:**
- `arguments` table: stores argument units (claim + evidence + target), not just flat concepts
- `confidence_queue` table: pending items for human-in-the-loop review
- `concepts.first_seen` / `concepts.last_seen`: temporal tracking fields

## 10. Node States

| State | Trigger | Data characteristics | System behavior | Seed generation impact |
|-------|---------|---------------------|-----------------|------------------------|
| `placeholder` | Citation expansion finds uningested paper | Only ID/title/year, no concepts | Participates in graph traversal, no LLM extraction | Marked "boundary fuzzy", filters its Gap weight |
| `uploaded` | Extraction complete, ingested | Concepts + arguments + edges populated | Activates citation expansion, triggers local rule closure | Activates deep pattern mining |
| `merged` | New paper's DOI/arXiv matches an existing placeholder during ingest | ID cross-match points to same paper | `_check_and_merge_duplicates()` merges concepts/arguments/edges, deletes duplicate, upgrades surviving to `uploaded` | Eliminates noise, improves subgraph confidence |

## 11. Pipeline Stages (Updated)

| Stage | Input | Processing | Output | Quality control |
|-------|-------|------------|--------|-----------------|
| **1. Parse** | User PDF | MinerU `-o` → Markdown + images in temp dir → chapter filter | Structured text blocks (≤12k chars), raw MD, images dir | Skip formulas/tables/appendices |
| **1.5. Save Assets** | raw MD + images dir | Save MD to `data/papers/<local_id>.md`, copy images to `data/papers/images/<local_id>/`, rewrite refs | Persistent raw extraction for inspection | None (fail silently if images missing) |
| **2. Identify** | Text + filename | Extract title/year/IDs → triple ID parse → local dedup → duplicate merge check | `local_id` assigned, `uploaded`/`placeholder`/`merged` | Priority matching prevents splits; `_check_and_merge_duplicates` merges existing placeholders |
| **3. Extract** | Filtered text | LLM outputs JSON: concepts + arguments + relations (with confidence) | Structured concepts + argument units + relation triples | Forced JSON Schema, confidence per item |
| **3.5. Validate** | Extracted concepts + relations | TBox type check + RBox relation check | Valid items pass, invalid items logged | Reject impossible pairs (e.g., Problem --proposes--> Method) |
| **3.6. Queue** | Items with confidence < threshold | Route to `confidence_queue` table | Accepted items proceed, low-confidence items pending | Configurable weak_threshold (default 0.7) |
| **4. Align** | Valid concept list | Rule cleaning → alias table match → LLM light judgment for new | `canonical_id`, update `Aliases` | 90%+ auto-align, 10% to confidence queue |
| **5. Ingest** | Concepts + arguments + relations + metadata | Write to `concepts`/`arguments`/`edges`, bind `source_paper`, update `first_seen`/`last_seen` | Graph partial update, node activated | Transactional write, rollback on failure |
| **6. Expand** | `s2_id`/`openalex_id` | Call S2 API for references & citations (limit 50/direction) → ID parse → local match. Also backfills DOI/arXiv from S2 `externalIds` if missing. | Neighbor metadata list + status tagging + ID backfill | Rate limiting + 24h local cache |
| **6.5. DOI Enrichment** | Paper with arXiv but no DOI | If S2 expansion failed (429) or returned no DOI, try CrossRef title search via polite API | DOI added to `paper_ids` table | Requires `crossref_email` config; fails silently on 404 |
| **7. Placeholder** | `in_graph=false` neighbors | Create `status='placeholder'` nodes, only ID/title/year, write `cites` edges | Topology connected, concepts empty | Placeholder nodes don't trigger re-expansion |
| **8. Closure** | New edges and arguments | Run rule engine (transitive closure / debate binding / gap承接 detection / evolution chain completion) | Implicit relations filled, Debate/Gap updated | Only runs on `uploaded` connected component |
| **9. Report** | All intermediate results | Assemble single-paper JSON report, compute coverage, generate boundary hints, persist to `reports/` | Pipeline-ready structured document | Coverage <30% or high-citation missing triggers terminal highlight |

**New stages:** 3.5 (Schema validation), 3.6 (Confidence queue routing)
**Updated stages:** 3 (now extracts arguments), 4 (routes to queue), 5 (writes arguments + temporal fields), 8 (uses arguments for closure)

## 12. Schema-First Validation Layer

Before any LLM output enters the database, it passes through symbolic constraint checks. No external reasoner needed — Python rule engine covers 90% of cases.

### TBox Constraints (Type Restrictions)

Each concept type has a whitelist of valid relation types:

```python
TBOX = {
    "Problem":   {"addresses", "leaves_open", "points_to"},
    "Method":    {"addresses", "proposes", "extends", "replaces", "solves"},
    "Conclusion":{"supports", "challenges", "limits"},
    "Debate":    {"supports", "challenges"},
    "Gap":       {"leaves_open", "points_to", "constrains"},
    "Actor":     {"affiliated_with", "proposes"},
}
```

Validation rule: if `concept.type == "Problem"` and `relation == "proposes"`, reject — Problems cannot propose Methods, they are addressed by them.

### RBox Constraints (Relation Restrictions)

```python
RBOX = {
    "transitive": {"extends"},
    "asymmetric": {"extends", "replaces", "challenges", "supports"},
    "irreflexive": {"extends", "replaces", "challenges", "supports", "limits"},
}
```

- Transitive: A extends B, B extends C → infer A extends C (handled in closure)
- Asymmetric: A extends B → B cannot extend A
- Irreflexive: A cannot extend A

### Validation Result

- **Pass**: item proceeds to ingestion
- **Fail**: item logged to `data/logs/`, replaced with warning in terminal output
- **Edge case** (valid but unusual): item routed to confidence queue

## 13. Confidence Queue (Single-Model + Threshold + Human Review)

Instead of expensive multi-model voting, use a single LLM with confidence-aware routing:

```python
# Ingest pipeline decision tree:
if item.confidence >= config["queue"]["auto_accept"]:   # default 0.9
    → direct ingestion
elif item.confidence >= config["queue"]["weak_threshold"]:  # default 0.7
    → ingestion with "weak" marker
else:
    → confidence_queue table (status = "pending")
```

### Queue CLI

`drbrain queue` — list all pending items:

```
┌──────────┬──────────┬──────────┬────────────┬──────────┐
│ Queue ID │ Type     │ Concept  │ Confidence │ Paper    │
├──────────┼──────────┼──────────┼────────────┼──────────┤
│ q001     │ concept  │ neuro-symbolic reasoning │ 0.52 │ p1a2b3 │
│ q002     │ relation │ solves: X → Y     │ 0.48 │ p1a2b3 │
└──────────┴──────────┴─────────────────────┴──────────┴────────┘
```

`drbrain queue resolve --id q001 --accept` — accept item (moves to concepts)
`drbrain queue resolve --id q001 --reject` — reject item (discards)

### Consensus Feedback Loop

When a concept appears in multiple papers:
- If 3+ papers independently extract the same normalized label with confidence > 0.8 → auto-promote to "consensus"
- Queue items matching a consensus concept → auto-accept
- This creates a self-improving system without multi-model cost

## 14. Argument Units (论证单元)

The core difference between "flat concept list" and "argument structure":

**Before (flat):**
```
Paper p1 → {Method: "Transformer", Problem: "long-range dependency"}
```

**After (argument):**
```
Argument a1:
  source: p1
  claim: "Self-attention replaces RNN for sequence modeling"
  type: proposes
  target: "Transformer" (Method)
  target_problem: "long-range dependency" (Problem)
  evidence: empirical (WMT14 EN-DE, BLEU +2.0)
  confidence: 0.95
```

This enables:
- Cross-paper argument comparison (A claims X works, B claims X fails under Y)
- Debate detection (two arguments with same target, opposite claim_type)
- Gap tracking (argument with type "limits" that identifies a Gap)

### LLM Extraction for Arguments

The extraction prompt is extended to output arguments alongside concepts:

```json
{
  "concepts": { /* existing schema */ },
  "arguments": [
    {
      "claim": "3-15 word claim statement",
      "claim_type": "supports|challenges|extends|limits|solves|proposes",
      "target": "target concept label",
      "target_type": "Method|Problem|Conclusion|Gap|Debate",
      "evidence_type": "empirical|theoretical|case_study|survey",
      "evidence_detail": "brief description of supporting evidence",
      "confidence": 0.0-1.0
    }
  ]
}
```

## 15. Temporal Concept Evolution

Track how concepts change over time using `first_seen` and `last_seen` fields on `concepts`:

```sql
-- Concept evolution timeline
SELECT c.label, c.type, MIN(p.year) as first_seen, MAX(p.year) as last_seen,
       COUNT(DISTINCT c.local_id) as paper_count,
       AVG(c.confidence) as avg_confidence
FROM concepts c JOIN papers p ON c.local_id = p.local_id
GROUP BY c.label
ORDER BY first_seen;
```

### Evolution Signals

| Signal | Detection logic | Meaning |
|--------|----------------|---------|
| **Emerging** | first_seen in last 2 years, paper_count growing | New concept gaining traction |
| **Established** | paper_count > 10, avg_confidence > 0.8 | Consensus concept |
| **Declining** | last_seen > 3 years ago, paper_count plateau | Method/Problem being superseded |
| **Contested** | avg_confidence < 0.7, paper_count > 5 | Active debate around this concept |
| **Resurging** | dormant > 3 years, then new paper_count in last year | Revived approach (e.g., symbolic AI) |

### Timeline CLI

`drbrain timeline --concept "transformer"` output:

```
Concept: transformer (Method)
  2017: first appeared (1 paper, confidence 0.95)
  2018: 12 papers (avg confidence 0.92) — rapid adoption
  2020: 45 papers (avg confidence 0.88) — peak
  2023: 23 papers (avg confidence 0.71) — declining, efficiency concerns
  2025: 8 papers (avg confidence 0.65) — contested by state-space models
Status: DECLINING
```

## 16. Knowledge Boundary Discovery (Upgraded from Research Seed)

Only runs on high-coverage (`uploaded` dense) subgraphs. Uses both concept and argument data.

**Implementation note:** Signal names in code use short identifiers: `stale_problem`, `unaddressed_gap`, `debate_zone`. These map to the conceptual patterns below:

| Boundary Pattern | Detection logic (code signal) | Output |
|------------------|-----------------|--------|
| 🔴 **Consensus Bottleneck** | Problem with >=3 incoming edges, no recent `addresses` (`stale_problem`) | "Problem X has no substantial progress, addressed by N papers but unresolved" |
| 🟡 **Unaddressed Gap** | Gap node with no incoming `addresses` or `extends` (`unaddressed_gap`) | "Gap Y identified but no proposed solution exists" |
| 🔵 **Debate Zone** | Same target has both `supports` and `challenges` edges (`debate_zone`) | "N papers with conflicting views — active debate, needs new benchmark" |
| 🟢 **Technology Cliff** | Method with dense `extends` chain then gap, related Gap exists | "Method D stalled due to constraint E, current conditions may enable revival" |
| 🟣 **Cross-Domain Isomorphism** | Two disconnected subgraphs share same Problem, path length > 3 | "Domain G and H both address Problem P — potential transfer opportunity" |

## 17. Citation Marking & JSON Report Structure

```json
{
  "paper": {
    "local_id": "p001",
    "title": "Attention Is All You Need",
    "year": 2017,
    "ids": { "doi": "10.x/xxx", "arxiv": "1706.03762" },
    "status": "uploaded"
  },
  "concepts": {
    "problems": [{ "label": "长程依赖建模", "confidence": 0.92 }],
    "methods": [{ "label": "Transformer", "confidence": 0.95 }],
    "conclusions": [],
    "debates": [],
    "gaps": [],
    "actors": []
  },
  "arguments": [
    {
      "claim": "Self-attention replaces RNN for sequence modeling",
      "claim_type": "proposes",
      "target": "Transformer",
      "evidence_type": "empirical",
      "confidence": 0.95
    }
  ],
  "references": [
    { "title": "...", "year": 2016, "ids": {}, "in_graph": true, "local_id": "p002" }
  ],
  "citations": [
    { "title": "...", "year": 2018, "ids": {}, "in_graph": false, "local_id": null }
  ],
  "summary": {
    "refs_in_graph": 12,
    "cits_in_graph": 3,
    "total_refs": 20,
    "total_cits": 15,
    "graph_coverage": 0.43
  },
  "boundary_alert": {
    "missing_core_refs": false,
    "isolated_subgraph": false,
    "low_coverage": false
  },
  "validation": {
    "items_rejected": 0,
    "items_queued": 1,
    "tbox_violations": [],
    "rbox_violations": []
  }
}
```

## 18. BM25 Query

Full-text search over paper titles, abstracts (from MinerU extraction), concept labels, and argument claims. Uses `rank-bm25` library. Configurable `k1` and `b` parameters. Supports type filtering (`--type Problem`), argument type filtering (`--arg-type challenges`), year range, and combined with graph traversal.

## 19. .gitignore Rules

```
# Python
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.pytest_cache/
*.egg

# Virtual envs
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Coverage
.coverage
htmlcov/
coverage.xml

# DrBrain runtime data (generated, not committed)
data/reports/
data/papers/
data/logs/
data/cache/*
!data/cache/.gitkeep
data/drbrain.db

# Local config (contains secrets)
config.local.yaml

# MinerU temp output
mineru_*/

# Superpowers tooling docs (local, not part of project)
docs/superpowers/
```

## 20. Error Handling & Output Protocol

- All commands output structured JSON to stdout with `--json` flag
- `drbrain ingest --json` suppresses terminal output and emits: `{"ingested": N, "successful": N, "failed": N, "papers": [...], "errors": [...]}`
- Terminal output uses `rich` for colored tables, progress bars
- Errors: non-zero exit code, error message to stderr
- LLM failures: fallback chain, final failure logged to `data/logs/validation.log` with timestamp
- API rate limits: exponential backoff, local cache
- Transaction rollback on DB write failure
- Coverage <30% or high-citation missing triggers terminal highlight (yellow/red)
- Schema validation failures: logged to `data/logs/validation.log`, terminal warning with count
- Confidence queue items with status "pending" counted in `drbrain stats`
- Duplicate placeholder detection: when a new paper's DOI/arXiv matches an existing placeholder, merge concepts/edges and promote to `uploaded` status
