# Analyze Enhancement: Subset Selection + LLM Insights

## Scope
- Add 4 paper subset selection modes to `drbrain analyze`
- Add LLM-generated insights to each report section
- Add cross-paper insight section (method migration)

## CLI
```bash
drbrain analyze --papers p1,p2         # specific papers
drbrain analyze --query "transformer"  # BM25 search
drbrain analyze --workspace myws       # workspace
drbrain analyze --discover "question"  # LLM graph discovery
drbrain analyze                        # all papers (existing)
```

## Implementation
- Modify `analyze_cmd` to handle new flags
- Add `_discover_papers(question, db, graph)` using ReasonerAgent
- Add `_generate_insights(report)` for LLM commentary on each section
- Add cross-paper section: `_find_method_migrations(concepts, edges)`
