# Research: Visualization Approaches for DrBrain Knowledge Graph

- **Query**: What visualization approaches suit a Python CLI academic knowledge graph tool?
- **Scope**: mixed (internal code audit + external library/ecosystem research)
- **Date**: 2026-05-09

## Current State

DrBrain has minimal visualization. Only two CLI commands emit diagrams, both using Mermaid `graph TD` for lineage trees:

| Command | File | Mechanism |
|---|---|---|
| `evolve --mermaid` | `src/drbrain/cli/analysis_commands.py:188` | `format_tree(trees, mermaid=True)` |
| `descendants --mermaid` | `src/drbrain/cli/analysis_commands.py:230` | `format_tree([tree], mermaid=True)` |

The Mermaid generation is in `src/drbrain/graph/genealogy.py`:

- `_to_mermaid()` (line 308): emits `graph TD` header, calls recursive builder
- `_mermaid_nodes()` (line 747): recursive node/edge generation. Nodes get labels + years + tooltips. Edges get relation labels + provenance.

No Graphviz, D3, Cytoscape, Plotly, or any other visualization backend exists.

## Research Findings

### 1. Mermaid Expansion

Mermaid supports ~20 diagram types. The ones relevant to knowledge graphs:

| Diagram Type | KG Use Case | Effort | Notes |
|---|---|---|---|
| `graph TD/LR` | Hierarchical lineage, descendant trees | Already done | Only TD direction used today |
| `mindmap` | Concept hierarchy, topic breakdown | Low | Mermaid syntax is simple indentation; maps well to tree structures |
| `timeline` | Concept/idea evolution over years | Low | Maps directly to year-annotated genealogy nodes |
| `quadrantChart` | 2D concept positioning (novelty vs impact, etc.) | Low-Med | Need numeric axes; could map confidence/citation-count |
| `classDiagram` | Ontology/type hierarchy, concept classes and relationships | Med | Rich syntax for inheritance, composition, associations |
| `stateDiagram` | Concept lifecycle (emerging, established, declining) | Low | Simple state-transition syntax |
| `flowchart LR` | Horizontal graph overview (already have TD) | Low | Just change direction header |
| `erDiagram` | Entity-relationship view | Med | Maps to concepts as entities, relations as... relations |

**Key constraints of Mermaid:**
- Text-based: limited to ~50-100 nodes before unreadable
- No search, no filtering, no interactive zoom
- Rendered externally (CLI dumps text, user pastes into live editor or renders via mermaid-cli)
- No compound/hierarchical nodes (parent-in-child containment)
- Edge routing is automatic, limited control
- Cannot style individual nodes with custom colors/sizes by data attribute

**Verdict:** Mermaid is a great terminal-first, zero-dependency quick view. Push as far as `mindmap` + `timeline` + `quadrantChart`. Beyond ~80 nodes, text diagrams break down. This is when a real renderer is needed.

### 2. Static Graph Visualization: Graphviz vs Matplotlib

#### Graphviz

- **Layout engines**: `dot` (hierarchical), `neato` (spring/force-directed), `fdp` (force-directed, scalable), `sfdp` (very large graphs, multiscale), `circo` (circular), `twopi` (radial)
- **Output**: SVG, PDF, PNG, EPS at publication resolution
- **Python binding**: `pip install graphviz` (pure Python, calls system Graphviz binary)
- **Academic usage**: Graphviz is the backend for many academic tools. Citespace and VOSviewer use custom Java renderers, but dot/neato layouts are the standard for publication figures.
- **Strengths**: Publication-quality typography (fontconfig), excellent edge routing, subgraph clustering, HTML-like labels for rich nodes
- **Weaknesses**: Static only (no interactivity), layout computation can be slow on 500+ nodes with `dot`, requires system Graphviz install (apt/brew)

#### networkx + matplotlib

- **Layouts**: spring, kamada-kawai, spectral, circular, shell, spiral
- **Strengths**: No external installs (pure Python), quick prototyping, good for debugging
- **Weaknesses**: Low visual quality without heavy customization, no built-in edge routing for dense graphs, text overlap common, no PDF vector output quality
- **Best for**: Quick internal checks, under 50 nodes

**Verdict:** Graphviz `neato` or `sfdp` for overview graphs, `dot` for hierarchical views. networkx+matplotlib only for debugging/quick checks. Graphviz output is the standard for academic figures.

### 3. Interactive Web Visualization

| Library | Visual Quality | Interactivity | Large Graph? | Learning Curve | Maturity |
|---|---|---|---|---|---|
| **Cytoscape.js** | High | Rich (expand/collapse, compound nodes, tooltips, search) | Good (2000+ nodes with cose layout) | Medium | Very high (used by STRING, BioCyc, NDEx) |
| **D3.js force** | Very High (customizable) | Unlimited (custom) | Moderate (500-1k nodes) | Steep | Very high |
| **vis.js (vis-network)** | Good | Good defaults (physics, clustering, navigation) | Moderate | Low-Med | Moderate (less active recently) |
| **Sigma.js v2** | High (WebGL) | Good (spatial exploration) | Excellent (10k+ nodes) | Medium | Moderate (v2 rewrite, modern) |

#### Detailed analysis:

**Cytoscape.js (recommended for interactive exploration)**
- Purpose-built for graph/network data, not a general viz library
- Key features: compound nodes (papers contain concepts), expand/collapse subtrees, edge bend points, grid/guide alignment, built-in layouts (cose, cola, dagre, klay, spread, concentric, circle, grid, breadthfirst)
- Plugin ecosystem: popper tooltips, expand-collapse, edgehandles, navigator
- JSON-based style language (mapping data attributes to visual properties)
- Used by: STRING protein networks, BioCyc metabolic pathways, NDEx, many bio/academic tools
- License: MIT
- v3.30+ is current stable

**D3.js force-directed (recommended for highly custom UIs)**
- Maximum flexibility, but you build everything
- d3-force module: centering, collision, link, many-body (repulsion), positioning forces
- Can create unique interactions not possible in Cytoscape
- Better for bespoke exploratory tools where standard graph interaction isn't enough
- License: ISC

**vis.js vis-network (good for quick interactive HTML export)**
- Simpler API than both above
- Good defaults: Barnes-Hut physics, clustering, smooth zoom/pan
- Less active maintenance post-2020 (community fork visjs/vis-network)
- pyvis wraps this, making it trivial to generate from Python
- License: MIT/Apache 2.0

**Sigma.js v2 (best for large graphs)**
- WebGL rendering, handles 10k+ nodes smoothly
- Uses graphology library for graph data model
- v2 is a clean rewrite, good architecture
- Less rich "out of the box" than Cytoscape
- Better for spatial exploration of large graphs than detailed interaction
- License: MIT

#### Python Web Framework Pairing

| Framework | Effort | Fit for Graph UI | Notes |
|---|---|---|---|
| **FastAPI + HTMX** | High | Excellent | Most flexible, best performance, production-grade. Use FastAPI for data API, HTMX for progressive enhancement. Serve Cytoscape/D3 HTML page. |
| **Streamlit** | Very Low | Moderate | Quickest prototype. Limited control over JS interop. Good for internal tools, not publication. |
| **Dash (Plotly)** | Medium | Good | Callback model can get complex with graph interactions. Plotly's network graph is basic. Would embed Cytoscape.js via custom component. |
| **Panel** | Medium | Good | Integrates with Bokeh. Good for data apps. Less graph-specific. |

**Verdict:** For medium-term (Tier 2/3), Cytoscape.js + FastAPI is the sweet spot. Cytoscape.js is purpose-built for exactly this kind of data (academic concept/relation networks), has compound nodes for paper-in-concept hierarchy, and has the richest out-of-the-box graph interaction features. FastAPI provides a clean data API layer. For quick wins, pyvis (wraps vis.js) generates interactive HTML with one function call.

### 4. Python Graph Visualization Ecosystem

| Library | Interactive | Output | Effort | Best For |
|---|---|---|---|---|
| **pyvis** | Yes (HTML, in-browser) | Standalone HTML | Very Low | Quick interactive graphs, prototyping |
| **python-graphviz** | No (static) | SVG, PDF, PNG, DOT | Low | Publication figures, static exports |
| **networkx + matplotlib** | No (static) | PNG, basic SVG | Very Low | Quick checks, debugging |
| **plotly** | Yes (browser or notebook) | HTML, PNG | Low-Med | Dashboards, moderate interactivity |
| **bokeh** | Yes (browser or server) | HTML, PNG | Medium | Data apps, dashboards |
| **igraph python** | No (static, via matplotlib/cairo) | PNG, SVG, PDF | Medium | Large graph analysis + basic viz |

#### pyvis (recommended for quick interactive export)
```python
from pyvis.network import Network
net = Network(height="750px", width="100%", directed=True)
net.add_node(1, label="Concept A", title="tooltip text")
net.add_edge(1, 2, label="supports")
net.show("graph.html")  # opens browser
```
- Generates self-contained HTML with vis.js
- Physics simulation, drag, zoom, search built-in
- Filtering by node/edge attributes via UI
- Custom styling by group
- Limitation: single HTML file, no server-side state, no custom JS interactions without post-hoc editing

#### python-graphviz (recommended for static export)
```python
import graphviz
dot = graphviz.Digraph(engine='neato')
dot.node('A', 'Concept A', style='filled', fillcolor='lightblue')
dot.edge('A', 'B', label='supports')
dot.render('graph', format='png')
```
- Multiple engines: `dot`, `neato`, `fdp`, `sfdp`, `circo`, `twopi`
- Publication-quality typography
- Requires `graphviz` system package (apt install graphviz / brew install graphviz)

## Tiered Recommendation

### Tier 1: Quick Wins (1-3 days, zero new deps)

**Expand Mermaid output** to cover more use cases with no dependency cost:

1. Add `--direction LR` flag to existing commands for horizontal layouts
2. Add `mindmap` output mode: `drbrain hierarchy <concept> --mermaid-mindmap` -- renders concept/sub-concept tree as mindmap
3. Add `timeline` output mode: `drbrain evolution <concept> --mermaid-timeline` -- shows concept instances over years
4. Add global `--mermaid-type` flag: `--mermaid-type mindmap|timeline|flowchart|stateDiagram`
5. Add `drbrain graph mermaid <id>` subcommand that dumps full subgraph as Mermaid flowchart

**Files to create/modify:**
- `src/drbrain/graph/genealogy.py`: add `_to_mindmap()`, `_to_timeline()`, `_to_flowchart()` functions
- `src/drbrain/cli/analysis_commands.py`: add `--mermaid-type` option
- `src/drbrain/cli/graph_commands.py`: new `mermaid` subcommand

### Tier 2: Medium (1-2 weeks, light deps)

**pyvis interactive HTML export** -- single-command interactive exploration:

1. Add `drbrain graph interactive <id>` that generates pyvis HTML with:
   - Physics simulation (Barnes-Hut)
   - Drag, zoom, search bar
   - Color by concept type (green=method, blue=theory, red=findings)
   - Node size by degree/centrality
   - Edge thickness by confidence
   - Click to show node details (paper, section, confidence)
2. Add `drbrain workspace explore <ws>` that exports full workspace graph

**Dependencies:** `pyvis` (add to pyproject.toml)

**Graphviz static export** -- publication-ready figures:

3. Add `drbrain graph export <id> --format png|svg|pdf --engine dot|neato` using python-graphviz:
   - `dot` engine for hierarchical views (default for sub-50 node trees)
   - `neato` engine for force-directed overview (for larger/denser graphs)
   - HTML-like labels with tooltips
   - Subgraph clusters for paper boundaries

**Dependencies:** `graphviz` python package + system Graphviz

### Tier 3: Ambitious (3-6 weeks, significant effort)

**Cytoscape.js + FastAPI web exploration UI:**

1. New CLI command: `drbrain explore` launches FastAPI server
2. Web UI with Cytoscape.js displaying:
   - Full graph with force-directed layout (cose-bilkent)
   - Compound nodes: papers contain their extracted concepts
   - Expand/collapse compound nodes
   - Search with highlight (fade non-matching nodes)
   - Filter panel: by concept type, confidence threshold, paper
   - Click node: sidebar with details (label, type, definition, paper, section, confidence)
   - Edge labels with relation type + provenance
   - Export current view as PNG/SVG
   - URL state encoding for sharing views
3. REST API endpoints (FastAPI):
   - `GET /api/graph/{paper_id}` -- full paper subgraph
   - `GET /api/search?q=...` -- search concepts with graph neighborhood
   - `GET /api/concept/{id}/neighborhood` -- k-hop neighborhood
   - `GET /api/workspace/{ws}/graph` -- workspace full graph

**Dependencies:** `cytoscape` npm package (served as static asset), `fastapi`, `uvicorn`

**Alternative: Sigma.js v2** if scaling to 1000+ node graphs is a hard requirement. Cytoscape.js starts to slow at ~2000 nodes; Sigma.js handles 10k+ via WebGL.

## Caveats / Not Found

- **Mermaid CLI rendering**: Currently DrBrain dumps Mermaid text to stdout. If we want PNG/SVG from Mermaid, we'd need `mermaid-cli` (Node.js) or the Mermaid Ink API. Not researched deeply -- likely Tier 2 addition.
- **Cytoscape.js vs Cytoscape desktop**: Cytoscape.js is the JS library. Cytoscape Desktop is a separate Java app. Only the JS library is relevant here.
- **Plotly network graphs**: Plotly's built-in network trace (`go.Scatter` with edge mode) is basic. It can convert networkx graphs but the visual quality and interactivity are far below Cytoscape.js. Omitted from recommendations.
- **igraph python**: igraph is excellent for graph analysis (faster than networkx for large graphs) but its visualization is limited to static matplotlib/Cairo output. It's an alternative analysis backend, not a visualization solution.
- **vis-network maintenance**: The original almende/vis repository has been archived. The community fork visjs/vis-network is maintained but less active. pyvis pins an older version. This is a risk for Tier 2 pyvis recommendation -- acceptable for quick prototype but not for Tier 3 production UI.
- **Streamlit graph components**: Streamlit has `streamlit-agraph` (wraps vis.js) and `streamlit-cytoscape` community components. These could accelerate Tier 2 prototyping compared to FastAPI from scratch, at the cost of less control over the UI.
