# 整合 PageIndex-RAPTOR-tree 全链路贯通

## Goal

PageIndex tree 和 RAPTOR 的代码各自独立存在但未贯通。本次任务将断层全部接上，使 PageIndex→RAPTOR→tree_vectors→tree_summaries→检索→推理形成完整闭环。

## Status: DONE

Phase 1 + Phase 2 全部完成。86 tests pass, 0 lint errors.

## Decision (ADR-lite)

**Architectural Principle**: 向量只增强检索（pre-filter / candidate expansion），推理由 LLM 完成。向量不做推理决策。

## Changes

### Phase 1: 数据生产 + 检索接通

| PR | 文件 | 改动 |
|----|------|------|
| PR1 | `services/embedding.py` | 新增 `build_paper_tree_vectors()` — `build_tree_vectors` + `build_raptor_tree` 桥接 |
| PR1 | `cli/commands.py` | `embed --tree` 自动追加 RAPTOR 递归聚类 |
| PR2 | `cli/commands.py` | `query --paper` 替换为 `query_by_structure_hybrid`（LLM primary + vector auxiliary） |

### Phase 2: 推理消费者

| PR | 文件 | 改动 |
|----|------|------|
| PR3 | `extractor/reasoner.py` | 新增 `get_raptor_summaries` agent tool + handler + dispatch |
| PR4 | `extractor/isomorphism.py` | `IsomorphicMapping` 新增 `raptor_source_context`/`raptor_target_context` |
| PR4 | `extractor/isomorphism.py` | 新增 `enrich_isomorphisms_with_raptor()` 函数 |

### New data flow

```
drbrain build --tree
  → build_tree_vectors()     # PageIndex embeddings → tree_vectors
  → build_raptor_tree()      # GMM cluster + LLM summarize → tree_summaries + tree_vectors

drbrain query "..." --paper <id>
  → query_by_structure_hybrid()  # LLM primary + vector auxiliary

ReasonerAgent.reason("...")
  → get_raptor_summaries tool    # Fetches cross-section summaries

isomorphism.find_isomorphic_patterns() + enrich_isomorphisms_with_raptor()
  → structural patterns + RAPTOR semantic context
```

## Tests Added

- `test_build_paper_tree_with_raptor_integration` — build_tree_vectors + build_raptor_tree 全链路
- `test_get_raptor_summaries_handler` — reasoner tool handler
- `test_get_raptor_summaries_empty_for_unknown_paper` — empty paper edge case
- `test_reason_dispatches_raptor_summaries` — tool dispatch in reason loop
- `test_enrich_isomorphisms_with_raptor_context` — RAPTOR context on mappings
- `test_enrich_isomorphisms_without_raptor_data` — graceful when no RAPTOR exists

## Out of Scope

* 新 embedding 模型集成
* FAISS 索引替换
* reasoning 模块全部改造（仅 isomorphism 试点）
