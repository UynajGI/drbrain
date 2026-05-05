# Translate & WS Refactor — scholaraio Patterns

Based on scholaraio's battle-tested translate.py and workspace.py implementations.

## Translate

### Current State (77 lines)
- Section/paragraph-based chunking, no placeholder protection
- Sequential translation (async but one-at-a-time)
- No resume, no retry, no language detection, no skip logic

### Target State

```
services/translate.py
├── detect_language(text) → str          # CJK/Kana/Hangul + Latin stopwords
├── validate_lang(lang) → str            # regex [a-z]{2,5}
├── _split_into_chunks(text, size) → []  # placeholder-protected (code, $$, $, images)
├── _translate_chunk(text, lang, config) → str
├── _translate_chunk_with_retry(...)     # exponential backoff, 5 attempts
├── _translate_chunk_resilient(...)      # timeout → subdivide → retry
├── translate_paper(paper_dir, config)   # full workflow
│   ├── skip: no raw.md / empty / same lang / already translated
│   ├── _load_or_init_state (resume)
│   ├── ThreadPoolExecutor (concurrency)
│   ├── persist prefix to output on each completion
│   └── return TranslateResult(path, ok, partial, skip_reason)
├── batch_translate(papers_dir, config)  # multi-paper, split concurrency budget
└── _record_translation_meta(...)        # write meta.json translations field
```

### Key Differences from scholaraio
| Aspect | scholaraio | DrBrain |
|--------|-----------|---------|
| LLM client | `requests` + `call_llm()` | litellm async `acall_text_with_fallback()` |
| LLM call | sync | wrap in `asyncio.run()` for chunk-level |
| Config | `Config` object | `config.yaml` dict |
| Paper dir | `meta.json` (paper metadata) | `raw.md` (MinerU output) |
| Output path | `paper_{lang}.md` | same |
| Portable export | `translation-bundles/` + images | skip for now (YAGNI) |

### Tasks
1. **T1** — `_split_into_chunks()` with placeholder protection (code, $$...$$, $...$, images)
2. **T2** — `detect_language()` + `validate_lang()` 
3. **T3** — `_translate_chunk_with_retry()` (exponential backoff) + `_translate_chunk_resilient()` (timeout→subdivide)
4. **T4** — `translate_paper()` full workflow: skip logic + state persistence + resume + ThreadPoolExecutor
5. **T5** — Integrate into `translate_cmd` CLI + `_record_translation_meta`
6. **T6** — Tests + E2E verification

## Workspace

### Current State (142 lines)
- Basic CRUD (create/add/remove/list/show/delete)
- Workspace.yaml with name/description/created
- refs/papers.json with local_id + added_at
- No validation, no atomic writes, no rename, no migration

### Target State

```
storage/workspace.py
├── validate_workspace_name(name) → bool    # no path traversal
├── _read_papers(ws_dir) → list[dict]      # atomic read
├── _write_papers(ws_dir, entries)          # tmp→rename atomic write
├── create_workspace(name, ...)             # + schema_version in yaml
├── add_papers(name, local_ids, ...)         # batch resolve + dedup
├── remove_papers(name, local_ids, ...)      # remove by id
├── rename_workspace(old, new)              # NEW
├── list_workspaces(root)                   # unchanged
├── get_workspace(name)                     # show detail
├── delete_workspace(name)                  # unchanged
└── load_workspace_papers(name)             # unchanged
```

### Tasks
1. **T7** — `validate_workspace_name()` + atomic `_write_papers()` (tmp→rename)
2. **T8** — Add `schema_version: 1` to workspace.yaml on create
3. **T9** — `rename_workspace()` CLI command
4. **T10** — Tests + E2E verification
