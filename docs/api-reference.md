# API Reference

Module-level reference of key public functions and classes. For architecture and design
rationale, see [Architecture](architecture.md).

---

## Graph Engine

`src/drbrain/graph/engine.py`

### `GraphEngine`

Directed property graph with load/save/traverse operations.

```python
class GraphEngine:
    def __init__(self)
    def add_edge(src, dst, relation, confidence=1.0, paper_id=None, section=None, node_id=None)
    def get_neighbors(node: str, hops: int = 2) -> set[str]
    def traverse(start, direction="forward", max_hops=3) -> list[tuple[str, str, str]]
    def load_from_db(db: Database, paper_ids: set[str] | None = None) -> None
    def persist_to_db(db: Database) -> None
    def get_concepts_by_node(conn, node_id: str) -> list[dict]
    def get_section_context(conn, concept_label: str) -> dict | None
    def get_section_contexts_batch(conn, labels: list[str]) -> dict[str, dict]
    def traverse_with_sections(conn, start_label: str, max_hops: int = 2) -> list[dict]
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_neighbors` | `set[str]` | All nodes within `hops` edges |
| `traverse` | `list[tuple]` | (src, relation, dst) triples along path |
| `load_from_db` | `None` | Populate graph from concepts + edges tables |
| `persist_to_db` | `None` | Save graph state back to database |
| `get_section_context` | `dict | None` | Section info for a concept label |
| `traverse_with_sections` | `list[dict]` | Path traversal with section provenance |

---

## Graph Embeddings (TransE)

`src/drbrain/graph/embedding.py`

### `TransEEmbedding`

Translating Embeddings model for link prediction and entity similarity.

```python
class TransEEmbedding:
    def __init__(self, dim: int = 128, epochs: int = 100, lr: float = 0.01, margin: float = 1.0)
    def train(graph, init_entities=None, init_relations=None) -> None
    def entity_embedding(label: str) -> list[float]
    def relation_embedding(rel: str) -> list[float]
    def score(head: str, relation: str, tail: str) -> float  # cos(h+r, t)
    def predict_link(head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]
    def similar_entities(label: str, top_k: int = 10) -> list[tuple[str, float]]
```

| Method | Description |
|--------|-------------|
| `train` | SGD training on all triples. `dim=128, epochs=100` by default. |
| `score` | Cosine similarity between `head+relation` and `tail`. Higher = more likely. |
| `predict_link` | Given head + relation, find most likely tails. |
| `similar_entities` | Cosine similarity over entity vectors. |

---

## Rule Closure Engine

`src/drbrain/graph/engine_closure.py`

### `ClosureEngine`

Symbolic and hybrid rule-based inference over the knowledge graph.

```python
class ClosureEngine:
    def closure(graph, db, workspace=None, mode="symbolic") -> list[dict]
    def closure_incremental(seed_nodes: set[str]) -> list[dict]
    def closure_with_sections(conn) -> tuple[list[dict], dict]
    def ground_rules(min_confidence: float = 0.5) -> list[dict]
    def detect_research_seeds(db=None) -> list[dict]
```

| Method | Description |
|--------|-------------|
| `closure` | Run all 8+4 inference rules. Mode: `symbolic` or `hybrid`. |
| `closure_incremental` | Run closure only on nodes reachable from given seeds. |
| `closure_with_sections` | Closure with section-provenance tracking. |
| `ground_rules` | Materialize transitive rules as concrete triples (t-norm). |
| `detect_research_seeds` | Find stale problems, unaddressed gaps, debate zones, technology cliffs, confidence collapse. |

---

## LLM Client

`src/drbrain/extractor/llm_client.py`

Functions for calling LLMs via litellm with automatic fallback.

```python
def call_with_fallback(prompt, models, system_prompt="", max_tokens=16384) -> dict | None
def call_text_with_fallback(prompt, models, system_prompt="", max_tokens=4096) -> str | None
def call_with_messages(messages, models, temperature=0.3, max_tokens=16384, tools=None) -> dict | None

async def acall_with_fallback(prompt, models, system_prompt="", max_tokens=16384) -> dict | None
async def acall_text_with_fallback(prompt, models, system_prompt="", max_tokens=4096) -> str | None
async def acall_with_messages(messages, models, temperature=0.3, max_tokens=16384, tools=None) -> dict | None
```

| Function | Returns | Used by |
|----------|---------|---------|
| `call_with_fallback` | `dict` (JSON) or `None` | Build pipeline stages |
| `call_text_with_fallback` | `str` or `None` | Simple generation (workflow synthesis) |
| `call_with_messages` | `dict` (with `content` + `tool_calls`) or `None` | ReasonerAgent |
| `acall_*` variants | Same as sync | Async contexts (SessionAgent, Agent) |

All functions iterate through the `models` list on failure. Returns `None` if all models exhausted.
Cache is keyed by SHA256 of (model, system_prompt, prompt, max_tokens). Disabled when `temperature > 0`.

---

## Build Pipeline Agents

`src/drbrain/extractor/agent.py`

5-stage LLM extraction pipeline agents.

```python
class BuildAgent(ABC):
    def __init__(self) -> None
    async def run(db, paper_id, input_data, config) -> AgentOutput
    def _build_prompt(input_data: AgentInput) -> str          # subclass override
    def _validate_output(raw: dict) -> dict                    # subclass override
    def _is_complete(db, paper_id) -> bool                     # idempotency check

class OntologyAgent(BuildAgent)     # Stage 1: ontology extension
class EntityAgent(BuildAgent)       # Stage 2: entity extraction (10-way concurrent)
class RelationAgent(BuildAgent)     # Stage 3: relation extraction
class CorefAgent(BuildAgent)        # Stage 4: coreference resolution
class RefineAgent(BuildAgent):      # Stage 5: iterative refinement
    def set_snapshot(concepts, relations) -> None
```

Each agent has idempotency via `build_stages` table — stages skip if already completed.

---

## SessionAgent

`src/drbrain/extractor/session_agent.py`

Persistent DB-backed agent for multi-turn reasoning across CLI invocations.
For a full guide, see [Sessions](sessions.md).

```python
class SessionAgent:
    def __init__(self) -> None
    def create_session(db, *, title="", system_prompt="", models=None) -> str
    def load_session(db, session_id, *, graph=None, models=None, closure_context="") -> bool
    def delete_session(db, session_id) -> bool

    async def ask(db, question, graph=None, models=None, closure_context="") -> str
    async def chat(db, graph=None, models=None) -> None  # interactive loop

    def inject_context(context: str, label: str = "") -> None
    async def reason_bidirectional(db, question, graph, models) -> dict
```

| Method | Description |
|--------|-------------|
| `create_session` | Returns session ID. Inserts row in `agent_sessions`. |
| `load_session` | Restores full message history from `agent_messages`. |
| `ask` | Single-turn question. Appends to session history. |
| `chat` | Interactive loop. Type `/exit` to stop. |
| `inject_context` | Insert a context message (e.g. build summary) into the session. |
| `reason_bidirectional` | LLM forms hypothesis, validates against KG, revises. |

---

## RAPTOR Tree

`src/drbrain/extractor/raptor.py`

Recursive semantic tree summarization (Sarthi et al., 2401.18059).

```python
async def build_raptor_tree(db, paper_id, leaf_vectors, config, models) -> list[dict]
```

Internal helpers:
```python
def _bic_gmm(x: list[list[float]], k: int) -> float     # BIC scoring for cluster count
def _gmm_cluster(x: list[list[float]], max_k: int = 15) -> list[list[int]]
def _umap_reduce(x: list[list[float]], n_components: int = 5) -> list[list[float]]
async def _summarize_cluster(texts, models, context) -> str
```

---

## Workflow Engine

`src/drbrain/reasoning/base.py`

Base classes for structured reasoning workflows. For a full guide, see [Workflows](workflows.md).

```python
@dataclass
class WorkflowContext:
    db: Any              # Database connection
    graph: Any           # GraphEngine
    models: list[dict]   # LLM model configs
    question: str        # User question
    cache: ApiCache | None
    results: dict[str, Any]
    def get(step_name: str, default=None) -> Any

class WorkflowStep(ABC):
    name: str = ""
    requires_llm: bool = False
    def run(ctx: WorkflowContext) -> Any: ...  # abstract

class ReasoningWorkflow:
    name: str = ""
    description: str = ""
    steps: list[WorkflowStep]
    def execute(ctx: WorkflowContext) -> dict[str, Any]
```

Registry:
```python
def register_workflow(name: str) -> callable     # decorator
def get_workflow(name: str) -> ReasoningWorkflow  # lookup + instantiate
def list_workflows() -> list[dict[str, str]]       # name + description for all
```

---

## Search

### BM25

`src/drbrain/query/bm25.py`

```python
class BM25Search:
    def __init__(self)
    def add_document(doc_id: str, text: str) -> None
    def build(k1: float = 1.5, b: float = 0.75) -> None
    def search(query: str, top_k: int = 20) -> list[tuple[str, float]]

def tokenize(text: str) -> list[str]
def build_bm25_index(db: Database, k1=1.5, b=0.75) -> BM25Search
```

| Method | Description |
|--------|-------------|
| `add_document` | Index a document (concept, argument, paper). Call before `build()`. |
| `build` | Compute IDF and doc vectors. Must call before `search()`. |
| `search` | Returns `[(doc_id, score), ...]` sorted by BM25 score descending. |
| `build_bm25_index` | Build from all concepts + arguments + papers in DB. |

### PageIndex Tree Retrieval

`src/drbrain/query/tree_retrieval.py`

```python
async def query_by_structure(db, paper_id, query, models) -> list[dict]
async def query_by_structure_hybrid(db, paper_id, query, models) -> list[dict]

def query_cross_paper(db, query, models, paper_ids=None) -> list[dict]
def tree_traversal_search(db, query, top_k=10) -> list[dict]
```

| Function | Description |
|----------|-------------|
| `query_by_structure` | LLM-guided branch/leaf selection on one paper's tree. |
| `query_by_structure_hybrid` | LLM navigation + vector similarity pre-filtering. |
| `query_cross_paper` | Collapsed-tree cosine similarity across all papers. |
| `tree_traversal_search` | Two-stage RAPTOR traversal: layer descent → collapsed fallback. |

---

## Embedding Service

`src/drbrain/services/embedding.py`

Text embeddings for tree nodes (PageIndex + RAPTOR).

```python
def _embed_batch(texts: list[str], cfg: EmbedConfig | None = None) -> list[list[float]]
def build_tree_vectors(paper_dir: Path, cfg, db) -> int       # sync
async def build_paper_tree_vectors(db, paper_id, config) -> int  # async (includes RAPTOR)
def search_tree(db, query, top_k=10, paper_id=None) -> list[dict]
```

| Function | Description |
|----------|-------------|
| `_embed_batch` | Route to local or openai-compat backend based on config. |
| `build_tree_vectors` | Embed PageIndex leaf nodes for one paper. Returns count. |
| `build_paper_tree_vectors` | Full pipeline: PageIndex + RAPTOR + embeddings. |
| `search_tree` | Cosine search over `tree_vectors` table. Optional `paper_id` filter. |

---

## Database

`src/drbrain/storage/database.py`

SQLite with WAL mode, schema migrations, and concept/edge CRUD.

```python
class Database:
    def __init__(self, path: str = "data/drbrain.db")
    def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor
    def executemany(sql: str, seq: list[tuple]) -> sqlite3.Cursor
    def commit() -> None
    def close() -> None
```

Key CRUD methods (conceptual — see source for full signatures):

| Area | Methods |
|------|---------|
| Papers | `insert_paper`, `get_paper`, `get_all_papers`, `delete_paper`, `update_paper_status` |
| Concepts | `insert_concept`, `get_concepts`, `get_concepts_by_type`, `delete_concepts_for_paper` |
| Edges | `insert_edge`, `get_edges`, `get_edges_for_paper`, `delete_edges_for_paper` |
| Arguments | `insert_argument`, `get_arguments`, `get_arguments_by_type` |
| Aliases | `insert_alias`, `get_aliases`, `resolve_alias` |
| Embeddings | `insert_embedding`, `get_embeddings`, `get_all_entities` |
| Stats | `get_stats` |
