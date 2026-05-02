# Graph Related: Multi-Paper Shared Concept Analysis

## Scope

New `drbrain graph related` subcommand that analyzes shared concepts, graph connections, and edge patterns across 2+ papers. Three analysis modes, default `concepts`.

## Command Signature

```bash
drbrain graph related <paper_id...> [--mode concepts|graph|edges] [--min-shared N] [--json] [--workspace W]
```

| Param | Default | Description |
|---|---|---|
| `paper_id...` | (required, min 2) | Paper local_id list |
| `--mode` / `-m` | `concepts` | Analysis dimension |
| `--min-shared` | `2` | Minimum number of papers a concept/edge must appear in to be shown |
| `--json` | false | JSON output |
| `--workspace` / `-w` | None | Limit scope |

## Mode 1: `concepts` (default)

### Algorithm
Pure SQL query on the `concepts` table. No graph loading.

```sql
SELECT label, type, COUNT(DISTINCT local_id) as paper_count,
       GROUP_CONCAT(local_id) as papers
FROM concepts
WHERE local_id IN (?, ?, ...)
GROUP BY label, type
HAVING paper_count >= ?
ORDER BY paper_count DESC, type, label
```

Additionally, compute per-paper coverage: total concept count per paper and shared concept count.

### Terminal Output

```
Shared concepts across 3 papers (min-shared: 2):
  gap_reward_hacking (Gap)            3 papers  [paper_a, paper_b, paper_c]
  method_rlhf (Method)                2 papers  [paper_a, paper_b]
  problem_safety (Problem)            2 papers  [paper_b, paper_c]

Coverage:
  paper_a: 12 concepts, 2 shared
  paper_b:  8 concepts, 3 shared
  paper_c: 15 concepts, 2 shared
```

### JSON Output

```json
{
  "mode": "concepts",
  "papers": ["paper_a", "paper_b", "paper_c"],
  "min_shared": 2,
  "shared": [
    {"label": "gap_reward_hacking", "type": "Gap", "paper_count": 3, "papers": ["paper_a", "paper_b", "paper_c"]},
    {"label": "method_rlhf", "type": "Method", "paper_count": 2, "papers": ["paper_a", "paper_b"]}
  ],
  "coverage": [
    {"paper_id": "paper_a", "total_concepts": 12, "shared_concepts": 2},
    {"paper_id": "paper_b", "total_concepts": 8, "shared_concepts": 3}
  ]
}
```

### Edge Cases
- Paper not found → "Paper X not found in database"
- Fewer than 2 paper_ids → "At least 2 paper IDs required"
- No shared concepts → "No shared concepts found (min-shared: N)"

## Mode 2: `graph`

### Algorithm
1. Load graph via `GraphEngine.load_from_db()` (optionally workspace-filtered)
2. For each paper, query its concept labels: `SELECT label FROM concepts WHERE local_id = ?`
3. For each paper's concept labels, run `traverse()` (1 hop, direction=both, all relations)
4. Intersect the neighbor sets across papers — concepts that appear in the traversal results of 2+ papers
5. For each shared neighbor, annotate which paper reached it through which relation+concept

### Complexity
- Graph load: O(E) where E = edge count
- Per-paper traverse: O(k * d) where k = paper's concept count, d = average degree
- Set intersection: O(n * m) where n = papers, m = union of all neighbor sets

For large libraries, workspace filtering is strongly recommended.

### Terminal Output

```
Graph connections via shared concepts (1-hop):

  gap_reward_hacking (Gap) — shared by 2 papers:
    paper_a:  method_rlhf --addresses--> gap_reward_hacking
    paper_b:  problem_safety --leaves_open--> gap_reward_hacking

  conclusion_safety (Conclusion) — shared by 2 papers:
    paper_b:  method_rlhf --challenges--> conclusion_safety
    paper_c:  method_dpo --supports--> conclusion_safety
```

### JSON Output

```json
{
  "mode": "graph",
  "papers": ["paper_a", "paper_b"],
  "connections": [
    {
      "concept": "gap_reward_hacking",
      "type": "Gap",
      "paper_count": 2,
      "paths": [
        {"paper_id": "paper_a", "path": [{"src": "method_rlhf", "relation": "addresses", "dst": "gap_reward_hacking"}]},
        {"paper_id": "paper_b", "path": [{"src": "problem_safety", "relation": "leaves_open", "dst": "gap_reward_hacking"}]}
      ]
    }
  ]
}
```

### Edge Cases
- Paper has no concepts in graph → skipped with note in output
- No shared graph connections → "No shared graph connections found"
- Graph is empty after workspace filtering → "No graph data available for given papers"

## Mode 3: `edges`

### Algorithm
Pure SQL query on the `edges` table.

```sql
SELECT relation, dst_id, COUNT(DISTINCT source_paper) as paper_count,
       GROUP_CONCAT(DISTINCT source_paper) as papers
FROM edges
WHERE source_paper IN (?, ?, ...)
GROUP BY relation, dst_id
HAVING paper_count >= ?
ORDER BY paper_count DESC, relation, dst_id
```

### Terminal Output

```
Shared edge patterns across 3 papers (min-shared: 2):
  addresses → gap_reward_hacking      2 papers  [paper_a, paper_b]
  challenges → conclusion_safety       2 papers  [paper_a, paper_c]
```

### JSON Output

```json
{
  "mode": "edges",
  "papers": ["paper_a", "paper_b", "paper_c"],
  "min_shared": 2,
  "shared_edges": [
    {"relation": "addresses", "target": "gap_reward_hacking", "paper_count": 2, "papers": ["paper_a", "paper_b"]},
    {"relation": "challenges", "target": "conclusion_safety", "paper_count": 2, "papers": ["paper_a", "paper_c"]}
  ]
}
```

### Edge Cases
- No shared edges → "No shared edge patterns found (min-shared: N)"
- Paper has no outgoing edges → omitted from results with note

## Files Touched

| File | Change |
|---|---|
| `cli/graph_commands.py` | Add `related` subcommand to `graph_app` |
| `tests/test_graph_commands.py` | Add tests for 3 modes + edge cases |

## Testing

### concepts mode
- 2 papers with 1 shared concept → shared concept shown
- 3 papers with concept in only 2 → shown with paper_count=2
- --min-shared 3 with concept in only 2 papers → not shown
- Paper not found → error
- Fewer than 2 inputs → error

### graph mode
- 2 papers sharing a concept via 1-hop graph connection
- Paper with no concepts → skipped
- No shared connections → message

### edges mode
- 2 papers sharing same (relation, target) edge
- --min-shared filtering
- No shared edges → message
