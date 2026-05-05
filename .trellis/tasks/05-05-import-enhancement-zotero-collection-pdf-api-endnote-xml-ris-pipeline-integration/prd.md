# Import Enhancement

## Context
Extend zotero_import.py with scholaraio patterns: collection filtering, PDF attachment detection,
creator parsing, Zotero Web API, Endnote XML/RIS, pipeline integration, collection→workspace.

## Requirements
- T1: Zotero local SQLite — collection filter, creator parsing, PDF detection, list_collections
- T2: Zotero Web API — pyzotero integration with PDF download
- T3: Endnote XML/RIS parsing with PDF extraction
- T4: Pipeline integration — dry-run, dedup, batch embed+index, collection→workspace

## Success
- Tests pass, CLI works, no regressions
