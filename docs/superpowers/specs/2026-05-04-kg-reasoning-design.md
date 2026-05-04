# KG Reasoning Upgrade: 3-Layer Reasoning Stack

## Scope

Three-layer reasoning upgrade based on papers 2202.07412, 2306.08302, 2511.11017:

- **Layer 1**: TransE embeddings for entity/relation vectors, link prediction, similarity search
- **Layer 2**: Logic+embedding fusion — path confidence scoring, hybrid closure
- **Layer 3**: LLM Agent — conversational graph reasoning with tool use

## Non-Goals

- Full KGE training pipeline (no negative sampling beyond random, no complex models like RotatE/ComplEx yet)
- OWL/DL reasoning
- Real-time embedding updates

---

## 1. Architecture

```
drbrain reason "question"        ← Layer 3: LLM Agent
       ↓
drbrain closure --mode hybrid    ← Layer 2: Symbolic + Embedding
       ↓
GraphEngine.learn_embeddings()   ← Layer 1: TransE
```

---

## 2. Layer 1: TransE Embeddings

### API

```
GraphEngine:
  .learn_embeddings(dim=128, epochs=100, lr=0.01) → None
  .entity_embedding(label) → np.ndarray | None
  .relation_embedding(rel) → np.ndarray | None
  .predict_link(head, relation, top_k=10) → list[(tail, score)]
  .similar_entities(label, top_k=10) → list[(label, similarity)]
```

### Algorithm

TransE: `h + r ≈ t` in vector space. Score: `||h + r - t||`.

Training: for each edge (h, r, t), generate negative sample (h, r, t') where t' is a random tail. Margin ranking loss.

### Storage

Embeddings stored in new table `embeddings(label TEXT, vec BLOB)` as numpy binary. Loaded on demand.

### CLI

```bash
drbrain embed             # train embeddings (once, or --retrain)
drbrain embed --info      # show embedding stats (dim, entity count)
```

---

## 3. Layer 2: Logic + Embedding Fusion

### Closure Enhancement

Current closure returns edges with hard confidence (1.0 or section-aware decay). Add embedding-based confidence:

```
For each inferred edge (h, r, t):
  embedding_score = ||h_emb + r_emb - t_emb||  →  [0, 1] normalized
  final_confidence = 0.5 * rule_confidence + 0.5 * embedding_score
```

### Path Confidence

For path rules (e.g., extends → addresses⁻¹ → supersedes_address):
```
path_score = composition of relation embeddings along the path
closer to target relation embedding → higher confidence
```

### CLI

```bash
drbrain closure --mode hybrid    # embedding-weighted inference
```

---

## 4. Layer 3: LLM Agent Reasoning

### Command

```bash
drbrain reason "How do Transformer and GNN relate in knowledge graph construction?"
```

### Agent Design

LLM has access to these tools:
- `search_concepts(query, limit=5)` — BM25 search
- `get_neighbors(node, hops=1, direction="both")` — graph traversal
- `find_path(src, dst)` — shortest path
- `get_embedding_scores(head, relation, candidates)` — rank by embedding

LLM iteratively calls tools, reasons about results, forms hypotheses.

### Implementation

Uses existing graph query functions. No new graph code — purely orchestration via LLM tool-calling loop in a new `extractor/reasoner.py`.

---

## 5. Files Touched

| File | Change |
|---|---|
| `graph/engine.py` | Add `learn_embeddings`, `entity_embedding`, `relation_embedding`, `predict_link`, `similar_entities` |
| `graph/embedding.py` | New: TransE implementation |
| `cli/commands.py` | Add `embed_cmd` (Layer 1), enhance `closure_cmd` with `--mode hybrid` (Layer 2), add `reason_cmd` (Layer 3) |
| `extractor/reasoner.py` | New: LLM Agent with tool-calling loop |
| `storage/database.py` | Add `embeddings` table |
| `tests/test_embedding.py` | New: embedding tests |
| `tests/test_reasoner.py` | New: reasoner tests |

## 6. Testing

### Layer 1
- TransE training converges (loss decreases)
- predict_link returns valid entities
- similar_entities returns reasonable results
- Embeddings are persisted/loaded correctly

### Layer 2
- closure --mode hybrid produces weighted edges
- Path confidence scores in [0, 1]
- Backward compat: closure without --mode behaves as before

### Layer 3
- reason_cmd calls LLM with proper tool definitions
- Tool results are correctly parsed and fed back
- LLM can chain multiple tool calls
- Graceful handling when no results found
