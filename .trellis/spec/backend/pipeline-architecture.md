# Pipeline Architecture

## Overview

Two-phase pipeline: lightweight ingest (PageIndex tree structuring) + 5-stage graph build (2511.11017-inspired agent-based KG construction). The synthesis point: PageIndex's document tree serves as the seed ontology for 2511.11017's KG construction pipeline.

## Design Constraints

- **TOC-first, not blank-slate**: Academic papers have author-crafted TOC. Ontology is upgraded from tree structure, not built from zero. Do not apply 2511.11017's "ontology from scratch" pattern to papers with extractable TOC.
- **node_id everywhere**: Every extracted concept, relation, and triple must carry `node_id` back to source tree node. No provenance = no trust.
- **Cross-domain TBox**: Unlike 2511.11017's single-category e-commerce domain, DrBrain handles arbitrary academic disciplines. TBox 6 types (Concept, Method, Dataset, Metric, Phenomenon, Theory) are the universal upper ontology.
- **Vectors augment retrieval, LLM does reasoning**: Text embeddings serve as pre-filters and candidate expansion in retrieval pipelines. They never make reasoning decisions — all inference, navigation, and judgment is LLM-driven. `provider=none` disables vectors entirely; the system must function correctly on pure LLM + BM25.

## Synthesis Architecture

### Dimension 1: Tree → Proto-Ontology

Tree.json chapter hierarchy is the seed for ontology extension, not raw text.

**Before (current)**:
```
Stage 1: LLM reads tree.json → suggests subcategories under TBox 6 types
```

**After (target)**:
```
Stage 1: tree.json TOC hierarchy → LLM upgrades section headings to ontology classes
         Section 3.1 "Methods"     → Method rdfs:subClassOf Concept
         Section 3.1.1 "Datasets"  → Dataset rdfs:subClassOf Method
         Section 4.2 "Metrics"     → Metric rdfs:subClassOf Concept
```

**Contract**:
```python
async def tree_to_ontology(
    tree: list[dict],        # tree.json nodes with title, level, children
    tbox: dict[str, str],    # TBox 6 type definitions
    model: str,
) -> OntologyExtension:
    """Map TOC hierarchy to ontology classes.

    Returns OntologyExtension with:
      - classes: list[OntologyClass]  (name, parent_tbox_type, tree_node_id)
      - subclasses: list[SubclassRelation]  (child, parent)
      - suggested_tbox_type per class
    """
```

**Key rule**: Never discard author structure. If TOC has `3.1 → 3.1.1 → 3.1.1.1`, ontology must preserve this hierarchy. LLM can add cross-links but cannot remove parent-child edges from the tree.

### Dimension 2: node_id Provenance Chain

Every entity, relation, and triple carries full provenance back to source document.

**Provenance chain**:
```
RDF triple → concept → tree node → section → paper
  (subject, predicate, object)
  .source_node_id = "3.1.1.2"
  .source_section = "3.1.1 Training Dataset"
  .paper_id = "2306.08302"
```

**Contracts**:
```python
class ProvenanceMixin:
    """Required fields on all extracted entities."""
    node_id: str           # tree.json leaf node ID
    section: str           # human-readable section path (e.g. "3.1.1 Training Dataset")
    paper_id: str          # source paper
    extraction_stage: int  # which build stage created this (1-5)

class Concept(ProvenanceMixin):
    label: str
    tbox_type: str
    confidence: float

class Relation(ProvenanceMixin):
    source_label: str
    target_label: str
    relation_type: str
    source_node_id: str    # where the relation was extracted (may differ from source concept's node_id)
```

**DB changes**: `edges` table gains `node_id` and `section` columns. `concepts` table already has `node_id` and `section` — validate they are always populated.

### Dimension 4: Agent-Based Stages

Each build stage runs as a dedicated agent with independent system prompt, input/output contract, and error boundary. Agents communicate through structured intermediate artifacts (tree.json, ontology.json, concepts.json, relations.json), not raw LLM context.

**Agent contracts**:
```python
class BuildAgent:
    """Base for 5 build-stage agents."""
    name: str                    # "ontology", "entities", "relations", "coreference", "refine"
    system_prompt: str           # role-specific instructions
    input_schema: type           # pydantic model for input validation
    output_schema: type          # pydantic model for output validation

    async def run(self, input_data: BaseModel) -> BaseModel:
        """Execute agent with retry + idempotency guard.

        1. Check DB for existing stage output → skip if complete
        2. Build prompt from system_prompt + input_data
        3. Call LLM via acall_with_fallback
        4. Validate output against output_schema
        5. Persist to DB with status=complete
        6. Return validated output
        """

class OntologyAgent(BuildAgent):
    name = "ontology"
    # Stage 1: tree.json → ontology extension (Dimension 1)

class EntityAgent(BuildAgent):
    name = "entities"
    # Stage 2: per-leaf-node concept extraction with node_id provenance (Dimension 2)

class RelationAgent(BuildAgent):
    name = "relations"
    # Stage 3: cross-section relation extraction with node_id provenance

class CorefAgent(BuildAgent):
    name = "coreference"
    # Stage 4: deduplicate concepts across sections

class RefineAgent(BuildAgent):
    name = "refine"
    # Stage 5: self-review, output corrections with refinement_diff
```

**Inter-agent protocol**: Agents do not share LLM context. Each receives structured input (pydantic model), returns structured output. The orchestrator (`build_graph_from_tree`) handles sequencing and passes data between agents via DB.

### Dimension 5: RAPTOR-Tree Integration (PageIndex → RAPTOR → Retrieval → Reasoning)

The full pipeline: PageIndex tree.json → `build_tree_vectors` (embed leaf nodes) → `build_raptor_tree` (GMM cluster + LLM summarize) → `tree_vectors` + `tree_summaries` tables → hybrid retrieval + reasoner tools + isomorphism enrichment.

**Architectural Principle**: Vectors augment retrieval only (pre-filter / candidate expansion). All reasoning is LLM-driven. Vectors never make reasoning decisions.

#### 1. Scope / Trigger
- Trigger: `drbrain embed --tree` or `drbrain build --tree`
- Layers: storage (tree_vectors/tree_summaries), extraction (RAPTOR), query (hybrid), reasoning (reasoner/isomorphism)

#### 2. Signatures

```python
# services/embedding.py — bridge function
async def build_paper_tree_vectors(
    paper_dir: Path,
    db_path: Path,
    embed_cfg: EmbedConfig | None = None,
    llm_models: list[dict] | None = None,
) -> int:
    """Build PageIndex tree vectors + RAPTOR recursive summaries for a single paper.
    Returns total vectors+summaries created.
    RAPTOR step is skipped gracefully if llm_models is empty/None.
    """

# query/tree_retrieval.py — hybrid retrieval
async def query_by_structure_hybrid(
    question: str,
    paper_dir: Path,
    db_path: Path,
    models: list[dict],
    cfg: EmbedConfig | None = None,  # None → pure LLM, no vectors
    top_k: int = 5,
) -> list[dict] | None:
    """LLM-primary tree navigation with optional vector pre-filtering.
    Returns [{node_id, title, content, source}] where source ∈ {llm, vector, llm+vector}.
    """

# extractor/reasoner.py — agent tool
def _get_raptor_summaries(self, paper_id: str) -> list[dict]:
    """Return RAPTOR summaries from tree_summaries table.
    Returns [{node_id, paper_id, summary_text, source_node_ids, tree_layer}].
    """

# extractor/isomorphism.py — enrichment
def enrich_isomorphisms_with_raptor(
    mappings: list[IsomorphicMapping],
    db: Database,
) -> list[IsomorphicMapping]:
    """Add raptor_source_context / raptor_target_context to each mapping."""
```

#### 3. Contracts

**tree_vectors table**:
| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT | PageIndex node ID or `raptor_{paper}_L{layer}_{uuid}` |
| paper_id | TEXT | Source paper local_id |
| embedding | BLOB | float32 array |
| content_hash | TEXT | SHA-256 prefix for incremental update |
| tree_layer | TEXT | `pageindex` or `raptor_L1` / `raptor_L2` ... |

**tree_summaries table**:
| Column | Type | Description |
|--------|------|-------------|
| node_id | TEXT | `raptor_{paper}_L{layer}_{uuid}` |
| paper_id | TEXT | Source paper local_id |
| summary_text | TEXT | LLM-generated cross-section summary |
| source_node_ids | TEXT | JSON array of child node_ids (provenance chain) |
| tree_layer | INTEGER | RAPTOR layer depth (1, 2, 3...) |

**IsomorphicMapping**:
| Field | Type | Description |
|-------|------|-------------|
| source_domain | str | First concept label |
| target_domain | str | Second concept label |
| shared_structure | str | Relation signature description |
| confidence | float | Jaccard×0.7 + label_sim×0.3 |
| raptor_source_context | list[dict] | RAPTOR summaries for source concept's papers |
| raptor_target_context | list[dict] | RAPTOR summaries for target concept's papers |

#### 4. Validation & Error Matrix

| Condition | Behavior |
|-----------|----------|
| embed.provider = "none" | Skip all vector generation, return 0 |
| Paper has < 3 PageIndex nodes | Skip RAPTOR (need ≥3 for GMM clustering) |
| GMM finds ≤1 cluster | Stop RAPTOR recursion at current layer |
| LLM summarization fails | Log warning, skip that cluster, continue |
| llm_models is empty/None | Skip RAPTOR, build_tree_vectors only |
| RAPTOR raises exception | Log warning, PageIndex vectors already stored |
| cfg=None in hybrid query | Pure LLM navigation, source="llm" |
| tree_vectors dimension mismatch | Log warning, skip that row |
| No tree_summaries for paper | Return empty list (not error) |
| Concept not in any paper | raptor_source_context = [] |

#### 5. Good / Base / Bad Cases

**Good**: Paper with 30+ PageIndex sections, local embedding, full LLM chain. `build_paper_tree_vectors` produces PageIndex embeddings + 3-layer RAPTOR tree with 10+ summaries. Hybrid query finds 5 sections (3 LLM-selected, 2 vector-boosted). Isomorphism enriched with relevant cross-section context.

**Base**: Paper with 5 PageIndex sections. RAPTOR produces 1-2 summaries (single layer). Hybrid query works as pure LLM (too few vectors to help). Isomorphism has empty or minimal RAPTOR context — structural similarity only.

**Bad**: No LLM models configured. `build_paper_tree_vectors` produces PageIndex vectors only, RAPTOR skipped. Hybrid query delegates to pure LLM (cfg=None path). Isomorphism mappings have no RAPTOR context. System functions correctly but without semantic enrichment.

#### 6. Tests Required

- Unit: `build_paper_tree_vectors` mocked → verify both `tree_vectors` pageindex layer AND `tree_summaries` rows exist
- Unit: `build_paper_tree_vectors` with empty llm_models → verify only PageIndex vectors, no RAPTOR
- Unit: `query_by_structure_hybrid` with cfg=None → verify source="llm", no vector augmentation
- Unit: `_get_raptor_summaries` with populated tree_summaries → verify correct ordering by tree_layer
- Unit: `enrich_isomorphisms_with_raptor` with no RAPTOR data → verify empty context lists
- Integration: full `build_paper_tree_vectors` → `query_by_structure_hybrid` → verify sections returned
- Integration: `enrich_isomorphisms_with_raptor` → verify raptor_source_context non-empty for known concepts

#### 7. Wrong vs Correct

##### Wrong
```python
# Building RAPTOR separately from PageIndex — vectors double-computed, provenance lost
build_tree_vectors(db_path, paper_dir, embed_cfg)  # embeds all leaf nodes
build_raptor_tree(paper_dir, db_path, embed_cfg, models)  # re-embeds same nodes
# → Duplicate embeddings, different content_hash, tree_layer collision risk
```

##### Correct
```python
# Bridge function: PageIndex once → RAPTOR reuses stored vectors for first layer
count = await build_paper_tree_vectors(paper_dir, db_path, embed_cfg, llm_models)
# → build_tree_vectors stores pageindex layer
# → build_raptor_tree embeds only RAPTOR summary nodes (raptor_L* layers)
```

##### Wrong
```python
# Pure vector retrieval — LLM has no say, black-box results
results = search_tree(query, db_path, top_k=5)
sections = [get_node_content(md_path, structure, r["node_id"]) for r in results]
# → No LLM reasoning, no tree navigation, no fallback when vectors misrank
```

##### Correct
```python
# Hybrid: LLM navigation PRIMARY, vectors AUXILIARY
sections = await query_by_structure_hybrid(
    question, paper_dir, db_path, models, embed_cfg
)
# → LLM navigates tree structure, selects candidates
# → Vectors pre-filter narrows search space
# → cfg=None gracefully degrades to pure LLM
```

### Good / Base / Bad Cases

**Good**: Paper with well-structured TOC, 30+ sections, clear methodology. Tree → Ontology produces 15+ classes with correct TBox mapping. All concepts carry node_id. Agent retry recovers from transient LLM failure on Stage 3.

**Base**: Paper with minimal TOC (only H1 headings). Tree → Ontology produces 3-5 classes. Remaining ontology elements come from LLM zero-shot (2511.11017 fallback path). node_id granularity is coarse but present.

**Bad**: PDF without extractable TOC (scanned document). Tree extraction falls back to LLM segmentation. Stage 1 must flag this as degraded mode. node_id points to auto-generated segments, not author sections.

### Tests Required

- Unit: `tree_to_ontology` with mock tree.json → verify class hierarchy preservation
- Unit: `Concept`, `Relation` models → verify `node_id` is required (pydantic ValidationError on missing)
- Unit: `BuildAgent.run` idempotency → verify skip when DB has complete output
- Unit: `BuildAgent.run` recovery → verify resume from failed stage
- Integration: full `build_graph_from_tree` with real tree.json → verify all concepts have node_id
- Integration: interrupt mid-pipeline, re-run → verify no duplicate ontology entries
- Integration: paper without TOC → verify degraded mode flag

### Wrong vs Correct

#### Wrong
```python
# Stage 1: LLM builds ontology from scratch, ignoring tree.json structure
prompt = "Suggest ontology classes for this paper"
ontology = await llm.call(prompt, paper_text)
# → Loses author structure, no connection to source sections
```

#### Correct
```python
# Stage 1: LLM upgrades tree.json TOC to ontology classes
tree = load_tree_json(paper_id)
ontology = await ontology_agent.run(OntologyInput(
    tree_nodes=tree,
    tbox=TBOX,
    paper_id=paper_id,
))
# → Preserves author hierarchy, each class linked to tree node_id
```
