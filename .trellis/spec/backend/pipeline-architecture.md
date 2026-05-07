# Pipeline Architecture

## Overview

Two-phase pipeline: lightweight ingest (PageIndex tree structuring) + 5-stage graph build (2511.11017-inspired agent-based KG construction). The synthesis point: PageIndex's document tree serves as the seed ontology for 2511.11017's KG construction pipeline.

## Design Constraints

- **TOC-first, not blank-slate**: Academic papers have author-crafted TOC. Ontology is upgraded from tree structure, not built from zero. Do not apply 2511.11017's "ontology from scratch" pattern to papers with extractable TOC.
- **node_id everywhere**: Every extracted concept, relation, and triple must carry `node_id` back to source tree node. No provenance = no trust.
- **Cross-domain TBox**: Unlike 2511.11017's single-category e-commerce domain, DrBrain handles arbitrary academic disciplines. TBox 6 types (Concept, Method, Dataset, Metric, Phenomenon, Theory) are the universal upper ontology.

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
