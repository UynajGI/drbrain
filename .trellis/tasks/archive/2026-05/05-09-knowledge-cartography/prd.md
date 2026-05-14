# brainstorm: knowledge-cartography (Phase 1: KG Export)

## Goal

Unlock the knowledge graph from SQLite. Add `drbrain export --format graphml` and `--format csv` to export the full graph structure (concepts + edges), making it ingestible by Gephi, Cytoscape, yEd, D3.js, and any other graph tool.

## Requirements

### R1: CSV node+edge export
- `drbrain export --format csv` dumps `nodes.csv` + `edges.csv`
- Nodes: id, label, type, description, confidence, section, paper_id
- Edges: source_id, target_id, relation, source_paper, weight, confidence
- Workspace-scoped via `--workspace` flag (filters to papers in workspace)

### R2: GraphML export
- `drbrain export --format graphml` produces valid GraphML
- Typed attributes (confidence as double, paper_id as string, etc.)
- Workspace-scoped via `--workspace` flag
- Validated: `networkx.read_graphml()` roundtrip

### R3: Integration with existing export command
- `--format` flag extends existing `drbrain export` command
- Existing formats (bib, ris, md) unchanged
- `--output` flag controls output path (default: stdout or auto-named file)

## Acceptance Criteria

- [ ] `drbrain export --format csv` produces nodes.csv + edges.csv with correct columns
- [ ] `drbrain export --format graphml` produces valid GraphML roundtrippable via NetworkX
- [ ] `drbrain export --format csv --workspace NAME` scopes to workspace papers
- [ ] `drbrain export --format graphml --output /path/out.graphml` writes to specified path
- [ ] Empty graph produces valid empty GraphML (no crash)
- [ ] Existing `drbrain export <paper_id> --format bib|ris|md` still works
- [ ] Tests cover: basic export, workspace scope, empty graph, roundtrip validation

## Definition of Done

- Tests added for CSV + GraphML export
- Lint / typecheck green
- No breaking changes to existing export command
- Zero new dependencies (NetworkX already in pyproject.toml)

## Decision (ADR-lite)

**Context**: Graph is locked in SQLite. No structured export exists. Need to feed external viz tools.

**Decision**: Phase 1 = CSV + GraphML only. Extend existing `export` command with `--format csv|graphml`. Phase 2-4 (Mermaid expansion, Graphviz, pyvis, controversy command, web UI) deferred.

**Consequences**:
- Zero new dependencies (NetworkX `write_graphml()` + stdlib `csv`)
- GraphML is the universal interchange format for Gephi/Cytoscape/yEd
- CSV is the universal fallback for any tool
- `--workspace` flag enables subgraph export without needing full-graph filtering

## Out of Scope

- JSON-LD / RDF / GEXF / Pajek / Neo4j dump (future formats)
- Mermaid expansion to more commands
- Graphviz / pyvis / D3.js / Cytoscape.js visualization
- `drbrain controversy` standalone command
- Interactive web UI / dashboard
- Advisor/advisee author lineage

## Technical Notes

- **Graph engine**: `src/drbrain/graph/engine.py` wraps `nx.MultiDiGraph` — direct `nx.write_graphml()` call
- **Export module**: `src/drbrain/storage/export.py` currently has BibTeX/RIS/Markdown per-paper only
- **CLI entry**: `export_cmd` at `src/drbrain/cli/export_commands.py:21-72`
- **Graph stats**: 2513 nodes, 26110 edges — GraphML output ~5-10MB for full graph
- **Workspace scoping**: `src/drbrain/storage/workspace.py` has `list_papers(ws_name)` for filtering
- **No new deps**: NetworkX `>=3.4` already in `pyproject.toml`

## Research References

- [`research/kg-export-formats.md`](research/kg-export-formats.md) — Format comparison: GraphML + CSV best first choices
- [`research/visualization-approaches.md`](research/visualization-approaches.md) — Deferred: Mermaid/Graphviz/pyvis tiers
- [`research/controversy-graph-design.md`](research/controversy-graph-design.md) — Deferred: controversy command design
