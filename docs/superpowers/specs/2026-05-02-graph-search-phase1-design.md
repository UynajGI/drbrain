# Graph Search Phase 1: Directed Traversal with Relation Filtering

## Scope

Phase 1 of graph-enhanced cross-paper search. Fixes `--neighbors` to return concept nodes and adds directed, relation-aware graph traversal. Covers Scenario B: "Given a seed paper/concept, find what else in the graph relates to it and how."

## Non-Goals (deferred to Phase 2+)

- Graph query primitives (`drbrain graph` command)
- Hybrid ranking (BM25 + graph centrality)
- Graph-only query entry (no text input required)

---

## 1. GraphEngine API

Two new dataclasses + one new method. Existing `get_neighbors()` left untouched.

### Dataclasses (`graph/engine.py`)

```python
@dataclass
class TraverseStep:
    src: str          # source node label
    relation: str     # edge type, e.g. "addresses"
    dst: str          # target node label
    hop: int          # 1-based hop number

@dataclass
class TraverseResult:
    target: str               # target node ID
    target_type: str          # "Paper" | "Problem" | "Method" | ...
    source: str               # which seed node this was reached from
    distance: int             # number of hops
    path: list[TraverseStep]  # full path from seed to target
```

### Method (`graph/engine.py`)

```python
class GraphEngine:
    def traverse(
        self,
        start_nodes: set[str],
        hops: int = 2,
        relations: set[str] | None = None,  # None = all types
        direction: str = "both",            # "forward" | "backward" | "both"
    ) -> list[TraverseResult]:
```

**Behavior:**
- `direction="forward"` — only `successors` (out-edges)
- `direction="backward"` — only `predecessors` (in-edges)
- `direction="both"` — both directions (default, backward-compatible)
- `relations=None` — no filtering, traverse all edge types
- BFS layer-by-layer, records full path for each result
- Same target reachable via different seeds/paths → keep all (no dedup)
- `start_node` not in graph → silently skipped
- Empty `start_nodes` → returns `[]`

---

## 2. CLI Interface

New optional flags on `query_cmd` in `cli/commands.py`:

| Flag | Short | Default | Description |
|---|---|---|---|
| `--neighbors` | `-n` | `0` | Existing, unchanged |
| `--relation` | `-R` | `None` (all types) | Comma-separated: `addresses,extends,challenges` |
| `--direction` | `-D` | `both` | `forward` / `backward` / `both` |

**Behavior:**
- `--neighbors 0` → `--relation` and `--direction` are ignored (no graph traversal)
- `--neighbors N` (>0) → calls `traverse(start_nodes, hops=N, relations=..., direction=...)`
- Invalid `--relation` value → error listing valid relation types
- `--direction` invalid value → error with valid options

**Examples:**
```bash
drbrain query "reward hacking" -n 2 -R addresses -D forward
drbrain query "RLHF" -n 1 -R extends -D backward
drbrain query "transformer" -n 2                     # backward-compatible default
```

---

## 3. Output Format

### Graph-expanded result fields (JSON mode)

Each graph-discovered result adds these fields on top of existing fields:

```python
{
    "_via_graph": True,                  # replaces old _via_neighbors
    "_source_seed": "concept_transformer_v1",
    "_distance": 2,
    "_path": [
        {"src": "seed_id", "relation": "addresses", "dst": "gap_id", "hop": 1},
        {"src": "gap_id", "relation": "leaves_open", "dst": "target_id", "hop": 2},
    ],
}
```

### Terminal output (Rich table)

```
  #1  Paper    "Safe RLHF via Reward Modeling"          2023  score: 0.85
      └─ graph: seed → addresses → Gap "reward over-optimization" → leaves_open → Paper

  #2  Method   "Iterated RLHF"                          2023  score: 0.72
      └─ graph: seed → extends → Method

  #3  Gap      "reward hacking"                               score: 0.00
      └─ graph: seed → addresses → Gap
```

- BM25 direct hits: normal display, no graph sub-line
- Graph-discovered: indented sub-line showing seed, relation chain, and target type
- Concept nodes use their `type` field (Problem/Method/Gap/Conclusion/Debate/Actor) instead of title
- JSONL mode: `_via_graph` + `_path` output as-is

---

## 4. Files Touched

| File | Change |
|---|---|
| `graph/engine.py` | Add `TraverseStep`, `TraverseResult` dataclasses + `traverse()` method |
| `cli/commands.py` | Add `--relation`, `--direction` options; call `traverse()`; format output |
| `tests/test_graph_engine.py` | Add tests for `traverse()` |
| `tests/test_query_cmd.py` | Add tests for new CLI flags |

---

## 5. Testing

### Graph Engine (`test_graph_engine.py`)
- `traverse` with no relations filter
- `traverse` with specific relations
- `traverse` forward only / backward only / both
- `traverse` with empty start_nodes
- `traverse` with start_node not in graph
- `traverse` multi-hop path correctness
- `traverse` multiple seeds merging

### CLI (`test_query_cmd.py`)
- `--neighbors 2 --relation addresses --direction forward` calls traverse with correct args
- `--relation invalid_value` raises error
- `--direction invalid_value` raises error
- Output includes concept nodes (not just papers)
- Output JSON contains `_via_graph`, `_path` fields
- Backward compat: `--neighbors 2` without new flags works as before
