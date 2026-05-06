# Layer 2 — Knowledge Genealogy

**Branch**: `dev/layer2-genealogy`
**First feature**: `drbrain evolve <concept>`

## Goal

Show how knowledge develops — concept lineage trees from graph traversal.

## Design

### `drbrain evolve <concept>` 

**Input**: concept label (e.g., "Transformer", "graph neural network")
**Graph traversal**: BFS from concept node, following relations: `extends`, `refines`, `applies`
**Output**: Rich tree showing ancestors + descendants with paper IDs and years

```
Transformer
  └─ Attention Is All You Need (2017) — origin
      ├─→ BERT (2018) — extends
      │   └─→ RoBERTa (2019) — refines
      │   └─→ ALBERT (2019) — refines
      ├─→ GPT (2018) — extends
      │   └─→ GPT-2 (2019) — refines
      │       └─→ GPT-3 (2020) — refines
      └─→ ViT (2020) — applies (cross-domain)
```

**Flags**:
- `--direction ancestors|descendants|both` (default: both)
- `--max-depth N` (default: 3)
- `--mermaid` — export as Mermaid graph
- `--json` — structured JSON output

## Implementation

### New module: `src/drbrain/graph/genealogy.py`

- `evolve_concept(graph, db, label, direction, max_depth) -> dict`
- BFS traversal starting from matching concept nodes
- Return nested tree structure with: node label, paper_id, year, relation, children

### CLI: `commands.py` — `evolve_cmd`
- Parse concept label → DB lookup → graph traversal → Rich tree output
- Register in `main.py`

### Mermaid export
- `--mermaid` flag renders tree as Mermaid flowchart
- Use `std::graph TD` syntax

## Files
- `src/drbrain/graph/genealogy.py` — new
- `src/drbrain/cli/commands.py` — `evolve_cmd`
- `src/drbrain/cli/main.py` — register command
- `tests/test_genealogy.py` — new

## Acceptance
- `drbrain evolve "Transformer"` outputs a concept lineage tree
- `--direction ancestors` shows only ancestors
- `--mermaid` exports valid Mermaid syntax
- `--json` exports structured JSON
- All tests pass, ruff clean
