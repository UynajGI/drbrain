---
name: paper-query
description: >
  Search and explore papers in the DrBrain library: three-leg evidence retrieval (`drbrain search`),
  bibliographic keyword search (`drbrain library search`), and paper-scoped deep reading. Use this
  skill whenever the user asks "what papers do I have about X?", "find papers on...", "search
  for...", "show me evidence about...", "look up concept X", "browse my library", wants to explore
  their collection by topic or concept type, or wants to read a specific paper's sections. Trigger
  proactively for any kind of library search or exploration.
---

# Paper query

Search and explore the DrBrain library. Two entry points: `drbrain search` (evidence rows with
sources and text locators, over the same chain as `ask`) and `drbrain library search`
(bibliographic BM25 rows over papers, concepts and arguments).

## Prerequisites

The index must be prepared: `drbrain index build`. `drbrain index status` shows whether evidence
retrieval is ready; a missing index makes `search` exit 1 with the remedy command.

## Search modes

### Evidence retrieval

```bash
drbrain search "graph neural networks"
drbrain search "over-smoothing" --limit 20
drbrain search "knowledge distillation" --json | jq '.evidence[]'
```

Each row carries `evidence_id`, `paper_id`, `node_id`, text (with `block_id`/`char_start`/
`char_end` when the tree leg read the leaf), `source`, `score`, and `content_checksum`; the JSON
payload also reports `route` and `generations` (which index version answered).

### Paper-scoped reading

```bash
drbrain search "how does the proposed method handle overfitting" --paper p3f8a2
drbrain search "regularization strategy" --paper p3f8a2 --paper p3f8a3 --json
```

`--paper` scopes every leg (including the tree leg's ANN entry search) to the given papers, so a
scoped query never silently reads another paper.

### External sources

```bash
drbrain search "transformer variants" --source all      # local evidence + arXiv
drbrain search "graph transformers" --source arxiv      # arXiv only
```

External rows are marked `source="arxiv"` and carry `url`/`doi`/`arxiv_id` with no local locator
(see the `fsearch` skill).

### Bibliographic keyword search

```bash
drbrain library search "graph neural networks" --type Method --limit 20
```

BM25 over paper titles, concept labels and argument claims; `--type` filters by document type
(Problem, Method, Conclusion, Gap, Debate, Actor, Paper, Argument).

## After finding evidence

- `drbrain ask "what is the main contribution of this paper?"` — natural language answer over the
  same retrieval chain
- `drbrain show p3f8a2` — inspect a paper's full record
- `drbrain reason "compare approach A and B"` — deep LLM agent reasoning (see kg-reason skill)
- `drbrain analyze p3f8a2` — run knowledge frontier analysis
- `drbrain citations p3f8a2 --type shared-refs` — find related work
- `drbrain ws add attention-methods p3f8a2` — save to a workspace

## Examples

**Topic search, machine-readable:**
```bash
drbrain search "contrastive learning" --limit 20 --json | jq '.evidence[] | {paper_id, node_id, score}'
```

**Check why a search returns nothing:**
```bash
drbrain index status --json        # is the index ready? which leg is not?
drbrain index build                # prepare what is missing
```

## CLI reference

| Command | What it does |
|---------|--------------|
| `drbrain search <terms>` | Three-leg evidence retrieval (bm25/vector/tree), no answer |
| `drbrain search <terms> --paper <id>` | Scope every leg to one or more papers |
| `drbrain search <terms> --source all\|arxiv` | Add external (arXiv) rows |
| `drbrain search <terms> --json` | `{query, engine, route, generations, legs, evidence}` |
| `drbrain library search <terms> [--type T]` | Bibliographic BM25 search (historical `search`) |
| `drbrain ask "<question>"` | Natural language answer over the same chain |
| `drbrain query <terms>` | Historical fusion query (hidden alias of `search`) |
