# p1-split-large-files

## Goal

拆分 4 个超 1000 行的核心文件，按职责切分到独立模块。纯重构，零行为变更，全部测试通过为唯一验收标准。

## Scope (MVP — sequential)

按以下顺序拆，每个独立 PR/commit，前一个绿后再做下一个：

### 1. `graph/engine.py` 1117 行 → 3 文件

**GraphEngine 类承载**：闭包推理 + 嵌入训练/查询 + 图遍历。

**拆分方案**：
- `graph/engine.py` — 保留 `GraphEngine` 类骨架 + NetworkX 图管理 + BFS 遍历
- `graph/engine_closure.py` — 闭包规则（8 symbolic + 4 embedding）+ `closure()` / `closure_incremental()` / `apply_path_rules()`
- `graph/engine_embeddings.py` — `learn_embeddings` / `entity_embedding` / `predict_link` / `similar_entities` / `_ensure_embeddings`

**机制**：用 mixin 或 composition。倾向 mixin（保持 `GraphEngine` 单一对象，避免改外部调用）：

```python
# graph/engine.py
from .engine_closure import ClosureMixin
from .engine_embeddings import EmbeddingsMixin

class GraphEngine(ClosureMixin, EmbeddingsMixin):
    # 骨架：图管理 + BFS 遍历
```

### 2. `parser/pageindex_parser.py` 1032 行 → 3 文件

- `parser/pageindex_parser.py` — markdown→tree 主入口 + `md_to_tree_with_fallback()`
- `parser/toc_extract.py` — TOC 提取三模式（header / PDF outline / LLM segmentation）
- `parser/toc_validate.py` — `validate_and_fix_tree()` + LLM 修复循环

### 3. `graph/genealogy.py` 1000 行 → 6 文件

- `graph/genealogy/common.py` — 共享 utils (`_get_concept_provenance`, etc.)
- `graph/genealogy/evolve.py` — concept lineage tree
- `graph/genealogy/descendants.py` — academic offspring
- `graph/genealogy/landscape.py` — domain timeline
- `graph/genealogy/paradigm.py` — paradigm shift detection
- `graph/genealogy/transfers.py` — cross-domain migration
- `graph/genealogy/__init__.py` — re-export 全部公开 API（保 backward compat）

### 4. `cli/_common.py` 987 行 → 3 文件

- `cli/_common/__init__.py` — re-export
- `cli/_common/context.py` — `_build_closure_context`, BM25 上下文构建, ask/reason context 组装
- `cli/_common/format.py` — 表格、mermaid、tree render
- `cli/_common/io.py` — 文件 IO atomic writes、JSON/YAML 读写

## Requirements

- [ ] 全部现有测试通过（`uv run pytest` 非集成）
- [ ] Lint clean (`uv run ruff check .`)
- [ ] 向后兼容：所有 `from drbrain.X import Y` 路径仍可用（通过 re-export）
- [ ] 无行为变更：纯重构，不修复 bug，不改进性能
- [ ] 每个文件拆分作为独立 commit，便于回滚

## Acceptance Criteria

- [ ] engine.py 拆完，`GraphEngine` API 不变，graph engine 测试全部通过
- [ ] pageindex_parser.py 拆完，parser 测试全部通过
- [ ] genealogy.py 拆完，genealogy/evolve/landscape 等 5 个 CLI 命令测试通过
- [ ] _common.py 拆完，所有 CLI 测试通过
- [ ] git diff 显示纯移动（`git diff --stat` 看插入/删除行数基本平衡）

## Out of Scope

- P2 性能优化（增量闭包、增量索引）
- P2 测试覆盖改进
- P3 高级 KGE 模型
- 重命名任何公开 API
- 修复任何"顺便发现"的 bug（记下来，单独任务）

## Technical Notes

### 风险

- **Mixin 多继承冲突**：若 ClosureMixin 和 EmbeddingsMixin 都有 `__init__` 需 super()。建议都不要 `__init__`，全部用 GraphEngine 已有属性
- **import cycle**：拆 genealogy 时 common.py 不能 import 子模块；子模块 import common 用 relative import
- **CLI `_common.py` 路径**：很多 cli/*.py 都 `from ._common import X`，拆为包后 `from ._common import X` 仍可用（包的 `__init__.py` re-export）

### 工具

- `git mv` 保留历史
- 跑测试 + lint 在每个 commit 前
- 不要 batch 4 个文件一起改，分 4 轮

## Decisions

- **GraphEngine 拆分机制**: Mixin 多继承（ClosureMixin + EmbeddingsMixin）✅
- **推进节奏**: 一次一个，逐步推进。先 engine.py，绿后再下一个 ✅
