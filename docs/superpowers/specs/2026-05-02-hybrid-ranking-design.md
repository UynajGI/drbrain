# Hybrid Ranking: BM25 + Graph Centrality

## Scope

Add a `--hybrid` flag to `drbrain query` that re-ranks BM25 results using graph centrality (PageRank). The graph signal amplifies structurally important results but never overrides text relevance.

## Approach: Multiplicative Boost

```
final_score = bm25_score * graph_boost
graph_boost = 1.0 + percentile_rank(page_rank)  # range [1.0, 2.0]
```

- node with highest PageRank → boost 2.0
- node with lowest PageRank → boost 1.0  
- node not in graph → boost 1.0 (no change)
- empty/sparse graph → all boosts 1.0, degrades to pure BM25

Why not weighted sum or re-rank: the graph is supplementary signal on a small library (dozens to low hundreds of papers). Text relevance must be the floor. Graph can amplify, never override.

## CLI Interface

```bash
drbrain query "reward hacking" --hybrid
```

| Flag | Default | Description |
|---|---|---|
| `--hybrid` | `false` | Enable hybrid ranking (BM25 + PageRank boost) |

No `--boost-alpha` or tunable parameter exposed. The [1.0, 2.0] bound is correct for the expected small-graph distribution. Tuning knobs YAGNI — no researcher wants to tune a parameter to get good results.

## Algorithm

### Step 1: BM25 search (existing)

Same as current `query_cmd` flow. Returns `results: list[dict]` with `score` field.

### Step 2: Load graph and compute PageRank

```
graph = GraphEngine()
graph.load_from_db(db)
pr = nx.pagerank(graph.graph, alpha=0.85)
```

### Step 3: Compute percentile rank for each node

Given PageRank dict `pr: dict[str, float]`, compute percentile rank for each node:

```
sorted_nodes = sorted(pr.items(), key=lambda x: x[1])
n = len(sorted_nodes)
percentile: dict[str, float] = {}
for rank, (node, _) in enumerate(sorted_nodes):
    percentile[node] = rank / (n - 1) if n > 1 else 0.5
```

Percentile rank maps to [0.0, 1.0]. Node with highest PageRank gets 1.0, lowest gets 0.0.

### Step 4: Apply boost

For each BM25 result:

```
node_id = result["local_id"]
if node_id in percentile:
    boost = 1.0 + percentile[node_id]  # [1.0, 2.0]
    result["score"] = round(result["score"] * boost, 4)
    result["_hybrid_boost"] = round(boost, 3)
else:
    result["_hybrid_boost"] = 1.0
```

Results are re-sorted by new score after boost is applied.

### Terminal output

When `--hybrid` is active, add boost info to the result line:

```
  1. [Method] method_rlhf (score: 0.852, boost: 1.8x, paper: paper_a)
```

When `--json` is active, each result gets `_hybrid_boost` field. The `score` field contains the boosted score.

## Performance

- PageRank: O(V + E) for power iteration, negligible for hundreds of nodes
- Percentile computation: O(V log V) for sorting, negligible
- Boost application: O(R) where R = result count
- Graph is loaded only when `--hybrid` is active
- No caching needed at this scale

## Edge Cases

| Scenario | Behavior |
|---|---|
| Graph is empty (no edges) | All boosts = 1.0, degrades to pure BM25 |
| Node in BM25 results but not in graph | boost = 1.0 |
| Single node in graph | percentile = 0.5, boost = 1.5 for that node |
| `--hybrid` not set | Existing behavior unchanged |

## Files Touched

| File | Change |
|---|---|
| `cli/commands.py` | Add `--hybrid` flag to `query_cmd`; compute PageRank + apply boost after BM25 |
| `tests/test_query.py` | Add tests for hybrid mode |

## Testing

- `--hybrid` flag adds `_hybrid_boost` to JSON output
- Boost is in [1.0, 2.0] range
- Central node gets higher boost than leaf node
- Node not in graph gets boost 1.0
- Results are re-sorted by boosted score
- `--hybrid` not set → behavior unchanged (backward compat)
