# 全局复杂度审计（2026-09-13）

## 扫描范围

使用 Lizard 1.24.0 扫描 `src/` 与 `scripts/` 下的 Python 代码，排除 `vendor` 与 `.venv`：

```bash
lizard -l python src scripts --exclude '*/vendor/*' --exclude '*/.venv/*' -C 10 -L 300 -w
```

该阈值用于发现问题，不作为 CI 阻断门槛。扫描得到 412 条告警；提高到 `CCN >= 25` 或 `NLOC >= 200` 后仍有 77 条高风险告警。

## 最高风险热点

| 位置 | NLOC | CCN | 判断 |
|---|---:|---:|---|
| `src/drbrain/rag/indexer.py`（175 行附近） | 310 | 99 | RAG 索引入口同时承担编排、分支和持久化，优先拆分 |
| `src/drbrain/cli/check_commands.py`（53 行附近） | 441 | 89 | CLI 检查逻辑过度集中，应按检查域拆成独立函数 |
| `src/drbrain/loop/director.py::run` | 425 | 83 | 研究循环总控，生命周期、恢复、执行和结算耦合在一起 |
| `scripts/pipeline/load_ingest_cache.py`（112 行附近） | 278 | 81 | 批处理加载与错误恢复混杂，影响可重入性 |
| `src/drbrain/cli/setup.py`（250 行附近） | 327 | 67 | 初始化向导包含过多环境分支 |
| `src/drbrain/loop/workflow.py::compute` | 202 | 64 | 实算门、作业状态和结果写回需要分层 |
| `src/drbrain/cli/_helpers/db_ingest.py::_ingest_single_paper` | 318 | 60 | 单篇摄取函数覆盖下载、解析、元数据和入库多个阶段 |
| `scripts/pipeline/build.py`（354 行附近） | 220 | 59 | 管线阶段与并发/缓存策略耦合 |
| `src/drbrain/rag/sql_retrie.py`（464 行附近） | 160 | 59 | SQL 检索分支复杂，建议抽出查询策略和结果适配 |
| `src/drbrain/loop/workflow.py::critique` | 202 | 49 | 讨论、证据和状态转换混在一个节点 |

另外，`workflow.py` 的 `verify`、`_retrieve_rag_evidence`、`_classify_verification`，以及 `query/tree_retrieval.py` 的结构检索函数也处于高复杂度区间。

## 结论

复杂度主要集中在三类边界：

1. **编排入口**：`loop/director.py` 与 `loop/workflow.py` 把状态机、策略、持久化和外部调用放在同一函数。
2. **数据管线入口**：RAG 索引、PDF 摄取、批处理加载函数同时处理阶段控制、错误恢复和 I/O。
3. **CLI 聚合入口**：`check`、`setup`、`build`、`repair` 等命令把用户交互、配置解析和业务逻辑集中在命令函数。

这不是单纯的“函数都太长”问题，而是层边界没有在入口处收敛。继续增加功能会扩大回归面，也会让单层测试难以隔离。

## 建议顺序

1. **P0：拆 `director.run`**：分成运行上下文恢复、周期调度、节点执行、结算/持久化四个协作者；保留一个薄的生命周期入口。
2. **P0：拆 `workflow` 高复杂度节点**：将检索、门禁判定、作业适配、证据写回和事件发射分别抽出；用 dataclass 传递节点上下文，避免继续增加参数。
3. **P1：拆 RAG 索引与 SQL 检索**：把 source adapter、chunk/index policy、持久化和失败记录分开，确保单个来源失败时可以独立入库或重试。
4. **P1：拆 `db_ingest` 与 pipeline loaders**：每个阶段返回明确的 typed outcome，编排层只负责顺序、重试和汇总。
5. **P1：拆 CLI 聚合函数**：命令函数保留参数解析和输出，检查/初始化/构建逻辑下沉到 service 层。
6. **P2：为复杂度建立基线**：将 `CCN >= 25` 或 `NLOC >= 200` 作为报告项；新增代码不得提高热点函数复杂度，重构后逐步下调基线。

## 不建议现在做的事

不要为了让数字变小而机械切函数，也不要把所有复杂函数直接改成大量微型包装。优先围绕真实边界（状态、I/O、策略、持久化、适配器）拆分，并为每个边界补单元测试。
