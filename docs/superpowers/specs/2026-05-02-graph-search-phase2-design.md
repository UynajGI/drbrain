# Graph Search Phase 2: Graph Query Primitives

## Scope

Phase 2 of graph-enhanced cross-paper search. Adds `drbrain graph` subcommand group with direct graph query primitives, enhances `closure` with rule filtering and dry-run mode, and fixes latent bugs.

## Non-Goals

- `graph related` (shared-concept query — algorithm not yet defined)
- Node auto-complete or `--list-nodes`
- Hybrid ranking (BM25 + graph centrality)

---

## 1. Bugfix: seed_cmd Key Access

**File:** `cli/commands.py:876-897`

`seed_cmd` accesses dict keys `seed['node']` and `seed['signal']`, but `detect_research_seeds()` returns dicts with keys `concept` and `description`.

**Fix:**

```python
# Before
typer.echo(f"  [{seed['type']}] {seed['node']}: {seed['signal']}")

# After
typer.echo(f"  [{seed['type']}] {seed['concept']}: {seed['description']}")
```

---

## 2. `closure --rule` / `--dry-run`

Two new flags on `closure_cmd` in `cli/commands.py`:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rule` | `str` (repeatable) | None (all rules) | Run only the named rule(s) |
| `--dry-run` | `bool` | False | Output inferred edges, don't persist |

### Valid `--rule` values

```
creates_debate, gap_addressed, indirect_evolution, gap_to_debate,
shared_actor, transitive_closure, asymmetric_violations,
method_supersedes_problem, challenge_chain, gap_inheritance, indirect_support
```

First 7 map to closure rules in `GraphEngine.closure()`. Last 4 map to `apply_path_rules()` in `path_reasoning.py`.

### Behavior

- No `--rule` → all rules run (current behavior, backward compatible)
- `--rule gap_addressed` → only runs that one rule
- `--rule gap_addressed --rule indirect_evolution` → runs the two specified rules
- No `--dry-run` → persist inferred edges to DB (current behavior)
- `--dry-run` → run `closure()`, output inferred edges, skip `persist_to_db()`

### `--dry-run` output

```
Rule: gap_addressed
  gap_reward_hacking --gap_addressed--> method_safe_rlhf (via gap_reward_hacking, confidence: 0.85)

Rule: indirect_evolution
  method_original_rlhf --indirect_evolution--> method_dpo (via method_ppo, confidence: 0.72)
```

### Implementation note

`GraphEngine.closure()` runs all rules in one method. To support `--rule`, either:
- Add a `rules: set[str] | None` parameter to `closure()` that gates each rule block; or
- Run full `closure()` and post-filter the returned list by `relation` field.

Post-filtering is simpler and avoids touching the closure internals. Each inferred dict has a `relation` key matching the rule name.

---

## 3. `drbrain graph neighbors`

```bash
drbrain graph neighbors <node_label> [--hops N] [--relation X,Y] [--direction D] [--json] [--workspace W]
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `node_label` | (required) | Concept label or paper ID to start traversal from |
| `--hops` / `-n` | `1` | Number of hops |
| `--relation` / `-R` | None (all) | Comma-separated relation types |
| `--direction` / `-D` | `both` | `forward` / `backward` / `both` |
| `--json` | `false` | Output JSON to stdout |
| `--workspace` / `-w` | None | Limit graph to workspace papers |

### Implementation

Reuses `GraphEngine.traverse()` with DB-based node type resolution. Extract the type-resolution logic from `query_cmd` into a shared helper in `cli/commands.py` or `graph/engine.py`.

If `--workspace` is set, use `load_from_db(db, paper_ids=workspace_ids)` to only load edges from those papers.

### Output

Same format as Phase 1 graph-expanded results:

**Terminal:**
```
Neighbors of method_rlhf (Method):
  gap_reward_hacking (Gap)
    └─ graph: method_rlhf -> addresses -> gap_reward_hacking
  method_ppo (Method)
    └─ graph: method_rlhf -> extends -> method_ppo
```

**JSON:** `_via_graph`, `_source_seed`, `_distance`, `_path` fields.

### Edge cases

- Node not in graph → "Node X not found in graph"
- Node exists but has no neighbors → "No neighbors found for X"

---

## 4. `drbrain graph path`

```bash
drbrain graph path <src_label> <dst_label> [--max-length N] [--json] [--workspace W]
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `src_label` | (required) | Start node label |
| `dst_label` | (required) | End node label |
| `--max-length` | `6` | Maximum path length (cutoff for BFS) |
| `--json` | `false` | Output JSON to stdout |
| `--workspace` / `-w` | None | Limit graph to workspace papers |

### Algorithm

1. Load graph from DB (optionally workspace-filtered)
2. Check both nodes exist in graph — if not, error with "Node X not found in graph"
3. Compute shortest path on `graph.to_undirected()` using `nx.shortest_path(G, src, dst, cutoff=max_length)`
4. For each hop in the node sequence, query the original directed graph for edge data:
   - Check `graph[src][dst]` (forward edge) or `graph[dst][src]` (backward edge)
   - Extract `relation` from the first matching edge
5. Annotate each hop with direction: `→` (forward) or `←` (backward against original edge direction)

### Output

**Terminal:**
```
Path from method_rlhf to method_dpo (2 hops):
  method_rlhf --extends--> method_ppo --replaces--> method_dpo
```

**No path:**
```
No path found between method_rlhf and paper_xyz (max length: 6)
```

**JSON:**
```json
{
  "src": "method_rlhf",
  "dst": "method_dpo",
  "length": 2,
  "path": [
    {"src": "method_rlhf", "relation": "extends", "dst": "method_ppo", "direction": "forward"},
    {"src": "method_ppo", "relation": "replaces", "dst": "method_dpo", "direction": "forward"}
  ]
}
```

### Edge cases

- Source node not in graph → error
- Target node not in graph → error
- No path within max_length → "No path found between X and Y"
- Source == target → "Source and target are the same node"

---

## 5. Files Touched

| File | Change |
|---|---|
| `cli/commands.py` | Fix seed_cmd key bug; add `--rule`/`--dry-run` to closure_cmd |
| `cli/graph_commands.py` | New file: typer app with `neighbors` and `path` subcommands |
| `cli/main.py` | Register `graph_app` on main app |
| `graph/engine.py` | Optionally add `rules` param to `closure()` (or post-filter in CLI) |

---

## 6. Testing

### Bugfix (seed_cmd)
- Verify `seed_cmd` does not crash with `KeyError` when run against a populated DB

### closure --rule/--dry-run
- `--dry-run` outputs edges without persisting
- `--rule gap_addressed` returns only gap_addressed edges
- `--rule invalid_rule` raises error
- `--rule` with multiple values runs multiple rules
- Backward compat: no new flags → behavior unchanged

### graph neighbors
- Neighbors of concept node with outgoing edges
- Neighbors of concept node with incoming edges (direction=backward)
- Neighbors of paper ID
- Node not found in graph
- --relation filter limits results
- --workspace limits traversal scope
- JSON output contains expected fields

### graph path
- Path between directly connected nodes
- Path between nodes 3 hops apart
- Source node not found → error
- Target node not found → error
- No path within max_length → message
- JSON output format correct
