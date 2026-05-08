# brainstorm: 架构健康 — 死代码清理、commands.py 拆分

## Goal

1. 删除 3 个已确认的死代码符号
2. 将 `commands.py`（4331 行）拆分为 8 个扁平命令文件 + 1 个共享辅助文件
3. `main.py` 简化为薄注册层

## Requirements

### 死代码清理

删除以下 3 个无任何引用的符号：

| 符号 | 文件 | 行号 |
|------|------|------|
| `_embed_signature` | `src/drbrain/services/embedding.py` | 42 |
| `expand_citations_oa` | `src/drbrain/extractor/citation.py` | 402 |
| `_pick_main_pdf` | `src/drbrain/services/zotero_import.py` | 656 |

**保留** `ExtractionError` / `StorageError`（公共 API 异常层次结构的一部分，有测试覆盖）。

### commands.py 拆分

采用与 `graph_commands.py` 一致的扁平文件模式：

```
cli/
├── ingest_commands.py    # 摄取管道 (34-691)
├── query_commands.py     # 核心查询 (916-1738)
├── export_commands.py    # 导出与数据操作 (1739-2262)
├── check_commands.py     # 系统检查 (2263-2879)
├── ws_commands.py        # 工作区管理 (2881-3034)
├── repair_commands.py    # 修复与导入 (3035-3366)
├── build_commands.py     # 构建/嵌入管道 (3367-3696)
├── analysis_commands.py  # 分析命令 (3697-4331)
├── _common.py            # 共享私有辅助函数
├── graph_commands.py     # (已有，不变)
├── main.py               # 薄注册层
├── setup.py
└── dependencies.py
```

**共享辅助函数** → `_common.py`：
- `_resolve_workspace_papers`
- `_resolve_node_type`
- `_show_actor`
- 其他被跨模块引用的私有函数

**`graph_commands.py` 修改**：import 从 `drbrain.cli.commands` 改为 `drbrain.cli._common`

**`main.py` 重构**：从 ~130 行缩减到 ~50 行。从 8 个命令模块导入，每个模块 ~5-10 个函数。`ws` 作为 typer 子应用注册（与 `graph` 一致）。

## Decision (ADR-lite)

**Context**: commands.py 4331 行，单文件承载全部 CLI 命令实现。
**Decision**: 扁平文件拆分（8 个命令文件 + `_common.py`），遵循 `graph_commands.py` 先例。
**Consequences**:
- `main.py` import 从 1 个模块变为 8 个，但每个模块职责清晰
- 共享辅助函数提取到 `_common.py`，`graph_commands.py` 的 import 路径需更新
- 命令函数签名不变 → 测试无需修改
- 异常类保留（公共 API）

## Acceptance Criteria

* [ ] `_embed_signature`、`expand_citations_oa`、`_pick_main_pdf` 已删除
* [ ] `_embed_signature` 在 `services/embedding.py` 中的调用点已清理
* [ ] `commands.py` 已删除，替换为 8 个命令文件 + `_common.py`
* [ ] `graph_commands.py` 的 import 更新为从 `_common` 导入
* [ ] `main.py` 简化为 ~50 行薄注册层
* [ ] `ws` 作为 `app.add_typer(ws_app, name="ws")` 注册
* [ ] 所有现有测试通过
* [ ] `drbrain --help` 输出与拆分前完全一致
* [ ] 所有子命令 `--help` 输出与拆分前完全一致

## Out of Scope

* `ExtractionError` / `StorageError` 删除
* `GraphEngine`、`Database`、`ReasonerAgent` 类拆分
* 测试文件拆分
* `.trellis/scripts/` 下的死代码
* services/ 目录重组

## Implementation Plan

1. **死代码清理**：删除 3 个符号 → 验证：tests pass
2. **提取 `_common.py`**：移动共享辅助函数 → 验证：`graph_commands.py` 和 `commands.py` 都能 import
3. **创建 8 个命令文件**：按行范围移动命令组 → 验证：每个文件语法正确
4. **重构 `main.py`**：更新 import 为 8 个模块 + 注册 ws typer → 验证：`drbrain --help` 不变
5. **删除 `commands.py`**：最终清理 → 验证：full test suite pass

## Technical Notes

* 报告: `reports/report_2026-05-08T04-23-14.md`
* `graph_commands.py` 第 11 行需要修改: `from drbrain.cli.commands import _resolve_node_type, _resolve_workspace_papers` → `from drbrain.cli._common import ...`
* `main.py` 第 10-54 行为 45 个命令函数的 import 块
* 共享辅助函数清单: `_resolve_workspace_papers`, `_resolve_node_type`, `_show_actor`, `_log_error`, `_move_to_pending`, `_save_paper_artifacts`, `_enrich_doi_from_crossref*`, `_fetch_citations_interested`, `_print_analyze_report`, `_render_landscape`, `_export_paper_to_meta`, `_enrich_tree_with_sections`, `_apply_mined_rules`, `_match_pattern`, `_extend_chain`, `_ingest_single_paper`, `_check_and_merge_duplicates`, `_merge_papers`
