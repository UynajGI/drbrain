# Research: Knowledge Graph Export Formats

- **Query**: What graph export formats should DrBrain support?
- **Scope**: External (format standards, tool compatibility, library support)
- **Date**: 2026-05-09

## Context

DrBrain stores a knowledge graph in SQLite with tables: `concepts` (id, label, type, description, confidence, section, node_id, paper_id), `edges` (source_id, target_id, relation, source_paper, weight, confidence), `arguments`, `aliases`, `embeddings`. The graph engine (`src/drbrain/graph/engine.py`) uses NetworkX `MultiDiGraph` in-memory. NetworkX (`>=3.4`) is already a project dependency. The current `export` module (`src/drbrain/storage/export.py`) only handles per-paper metadata (BibTeX, RIS, Markdown). No graph structure export exists.

## Findings

### Format Comparison Matrix

| Format | Tool Compatibility | Python Library | Complexity | Pros | Cons |
|---|---|---|---|---|---|
| **GraphML** | Gephi, Cytoscape, yEd, NetworkX | NetworkX (built-in) | LOW | Mature XML standard; rich typed attributes; wide viz tool support | XML verbose; no native dynamic/hierarchy support |
| **GEXF** | Gephi (native), NetworkX, sigma.js | NetworkX (built-in) | LOW | Gephi-native; dynamic graphs; hierarchical nodes/edges; rich attributes | Less universal than GraphML; Cytoscape requires plugin |
| **GML** | Cytoscape, Gephi, yEd, Tulip | NetworkX (built-in) | LOW | Simple human-readable text; good for manual inspection | Limited attribute types; no standard namespace |
| **CSV node+edge** | Universal (Excel, Python, R, any tool) | `csv` (stdlib) | LOW | Dead simple; universally parseable; no deps; can split node/edge tables | No standard schema; no metadata; no hierarchy; ad-hoc |
| **vis.js/D3 JSON** | D3.js, vis.js, ECharts, web dashboards | `json` (stdlib) | LOW | Web-native; directly embeddable; simple `{nodes:[], edges:[]}` shape | No standard; no typed attributes; visualization-only |
| **Cytoscape JSON (cyjs)** | Cytoscape.js, Cytoscape desktop | `json` (stdlib) | LOW-MED | Web-native; rich visual style support; layout hints | Cytoscape-specific; non-trivial structure |
| **NetworkX GPickle** | NetworkX only | NetworkX (built-in) | LOW | Fastest roundtrip; preserves all Python objects; zero config | Python-only; not portable; pickle security risk; no viz tool support |
| **Neo4j Cypher dump** | Neo4j Browser, Neo4j Desktop | Custom | MEDIUM | Importable to Neo4j; queryable after import; APOC supports batch | Neo4j-specific; need string escaping; large dumps slow to import |
| **Pajek (.net)** | Pajek, VOSviewer, CiteSpace | NetworkX (built-in) | LOW | Lightweight; used by bibliometric tools (VOSviewer, CiteSpace) | Niche; limited attributes; dated format |
| **JSON-LD** | Schema.org consumers, web crawlers, semantic web tools | `rdflib`, `pyld` | HIGH | W3C standard; semantic web compatible; schema.org vocabulary; machine-readable linked data | Need ontology mapping; no viz tools consume natively; `rdflib` not in deps |
| **RDF/Turtle** | SPARQL endpoints, triple stores, Protégé | `rdflib` | HIGH | W3C standard; SPARQL-queryable; rich ontologies (SKOS, Dublin Core, PROV-O) | Steep learning curve; no viz tool support; `rdflib` not in deps; overkill for internal use |

### Visualization Tool Input Formats

| Tool | Supported Input Formats |
|---|---|
| **Gephi** | GEXF (native), GraphML, GML, CSV, Pajek, UCINET, Tulip, spreadsheets |
| **Cytoscape** | GML, GraphML, XGMML, SIF, CSV/TSV, Excel, JSON, CX (Cytoscape Exchange) |
| **D3.js force layout** | Any JSON structure; typically `{nodes: [{id, ...}], links: [{source, target, ...}]}` |
| **vis.js network** | JSON `{nodes: [{id, label, ...}], edges: [{from, to, ...}]}` via DataSet/DataView |
| **Neo4j Browser** | Cypher LOAD CSV, Cypher CREATE statements, APOC import (GraphML, JSON), neo4j-admin import |
| **yEd** | GraphML (native), GML, Excel |
| **VOSviewer** | Tab-delimited text, GML, Pajek (.net), VOSviewer map/network format |

### Academic Tool Export Formats

| Tool | What It Exports | Relevant Formats |
|---|---|---|
| **OpenAlex API** | REST JSON (Works, Authors, Concepts, Institutions, Topics) | JSON objects with IDs; graph is implicit through ID references; no bulk graph format |
| **Semantic Scholar API** | REST JSON (Papers, Citations, References, Authors) | JSON; citation graph implicit in `citationCount`, `citations[]`, `references[]` fields |
| **VOSviewer** | Maps, networks, clusters from bibliometric data | Tab-delimited text, GML, Pajek (.net), JSON (map format), image export |
| **CiteSpace** | Co-citation networks, burst detection, timelines | Internal format; exports to Pajek (.net), GML; uses WoS/Scopus/PubMed input |
| **Connected Papers** | Web-only derivative graph | No export; renders similarity graph interactively |
| **Litmaps / ResearchRabbit** | Web-only | No standard export; some CSV/JSON download of paper lists |

**Key insight**: The bibliometric tool ecosystem (VOSviewer, CiteSpace) converges on **GML** and **Pajek (.net)** for graph interchange. OpenAlex / Semantic Scholar use REST JSON where the graph is implicit in ID references -- not a bulk graph dump format. Connected Papers and similar tools do not export graph structure at all.

### Python Library Details

| Format | Library | Notes |
|---|---|---|
| GraphML | `networkx.readwrite.graphml` | `nx.read_graphml()` / `nx.write_graphml()`. Supports typed attributes. |
| GEXF | `networkx.readwrite.gexf` | `nx.read_gexf()` / `nx.write_gexf()`. Supports dynamic, hierarchy. |
| GML | `networkx.readwrite.gml` | `nx.read_gml()` / `nx.write_gml()`. String labels only. |
| Pajek | `networkx.readwrite.pajek` | `nx.read_pajek()` / `nx.write_pajek()`. |
| GPickle | `networkx.readwrite.gpickle` | `nx.read_gpickle()` / `nx.write_gpickle()`. |
| CSV | `csv` (stdlib) | Custom writer; trivial. |
| JSON (vis.js/D3) | `json` (stdlib) | Custom writer; trivial. |
| Cytoscape JSON | `json` (stdlib) | Custom writer; needs `cyjs` structure mapping. |
| Neo4j Cypher | Custom | Generate `CREATE` / `MERGE` statements with proper escaping. |
| JSON-LD | `rdflib` (6.x) | `rdflib-jsonld` plugin. Need to define context mapping. |
| RDF/Turtle | `rdflib` (6.x) | `g.serialize(format='turtle')`. Need to build RDF graph first. |

**Dependency status**: NetworkX is already in `pyproject.toml`. `rdflib` is not, and adds significant weight (~10MB) for a format with no viz tool support. Recommend avoiding `rdflib` dependency unless JSON-LD / RDF is explicitly requested.

## Recommendation

### Phase 1 (immediate, low complexity)

1. **CSV node+edge tables** -- Universal interchange. Every tool ingests it. Zero new dependencies. Can export concepts table as nodes CSV and edges table as edges CSV. Perfect for ad-hoc analysis in Python/R/Excel.
2. **GraphML** -- Gephi + Cytoscape + yEd compatibility. NetworkX handles serialization. Covers the desktop visualization use case. XML-based with typed attributes.

### Phase 2 (medium complexity)

3. **vis.js/D3 JSON** -- Web embedding. Simple `{nodes: [...], edges: [...]}` shape. Enables embedding interactive graphs in dashboards or HTML reports.
4. **GEXF** -- Gephi-native format with dynamic/hierarchical support. NetworkX handles it. Better than GraphML for temporal or layered graphs.

### Phase 3 (niche, only if requested)

5. **Neo4j Cypher dump** -- For users who want to import into Neo4j for SPARQL/Cypher queries.
6. **JSON-LD/RDF** -- Academic linked data. Only if semantic web interoperability is a stated requirement. Adds `rdflib` dependency.

### Implementation Notes

- Since `GraphEngine` already wraps `nx.MultiDiGraph`, most format writers can call `nx.write_<format>(self.graph, path)` directly after augmenting node/edge attributes from the DB.
- For CSV: a custom writer is cleaner than NetworkX's `write_edgelist` / `write_adjlist` since DrBrain needs typed attributes (confidence, section, node_id, relation type).
- For vis.js JSON: trivial `json.dumps({"nodes": [...], "edges": [...]})` with node/edge dicts built from DB queries.
- The existing `export` CLI subcommand structure (`src/drbrain/cli/export_commands.py`) can be extended with `drbrain export graph --format <fmt>`.

## Caveats / Not Found

- No existing graph export code in DrBrain to build on. The current `export.py` is metadata-only (BibTeX/RIS/Markdown).
- Gephi GEXF vs GraphML preference: Gephi actively develops GEXF and considers it the modern replacement for GraphML. However, GraphML has broader tool support (Cytoscape, yEd, etc.).
- The exact attribute schema for DrBrain concepts (which fields beyond label/type/confidence to include) needs design -- this research covers format choice, not schema.
