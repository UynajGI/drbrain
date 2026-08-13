# Research Loop 设计（编排闭环 · 最后一部分）

> 三合一架构第三层：**RAG（读文献）+ 插件（跑模型/软件）+ loop（编排闭环）**。
> 定调（2026-08-13）：**确定性骨架 + 借 AutoScientists 闭环语义**。
> 参考：mims-harvard/AutoScientists（自组织 agent 团队 + 提案互评 + 共享成败）。

## 一、定位

- **RAG 层**（已做）：文献理解 —— LlamaIndex 检索/融合/Agent + Epistemic Layer。
- **插件层**（已做）：Model-as-Tool 接口 —— 模型与软件统一走 `Plugin` 协议（`src/drbrain/plugins/`）。
- **loop 层**（本文）：用 LlamaIndex `Workflow` 状态图把检索、抽取、Gap、假设、验证、报告串成**可循环 pipeline**，验证通过沉淀回 KG + 模型。

三者关系：loop 是骨架，RAG 与插件是它的两只手——一只读文献、一只跑模型/软件，loop 负责编排、状态流转与自进化。

## 二、编排框架

LlamaIndex `Workflow`（`@step` + pydantic `Event` + `Context.store`），与 RAG 层同依赖，避免 LangGraph State ↔ LlamaIndex Node 的二次映射；`Evidence` 直接定义为 pydantic `Event`。

## 三、确定性骨架（10 节点）

```
[任务规划] → [文献检索] → [文献筛选] → [PDF解析] → [知识抽取]
    → [实体规范化] → [跨文献融合] → [Gap识别] → [证据核验] → [报告生成]
      ↑                                   ↑
      └──── 检索不足/证据不足 → 回检索/回抽取（条件循环）
```

每个节点 = 一个 `@step`；节点间用 pydantic `Event`（= 输入/输出 Schema）交换，非自由群聊；节点内部 LLM 做局部自主决策。

## 四、AutoScientists 闭环语义（新增，落在骨架之上）

借鉴 AutoScientists 的 loop 语义，不改确定性骨架：

1. **假设提出**：`Gap识别` 之后从 gap 生成候选假设（可验证、带实验条件）。
2. **互评（critique-before-compute）**：花算力前先对假设互评打分，淘汰低质/不可验证假设。
3. **实验验证（双路）**：
   - 推理路（RAG）：检索证据 → LLM 推理；
   - 计算路（插件）：调模型（GBDT/GNN）+ 调软件（LAMMPS/DFT）。
4. **共享成败**：验证结果（成功/失败 + 证据）写入共享记忆（KG claim/evidence + `answer_records`），避免重复探索，支撑多假设并行。

## 五、节点映射（骨架 + 闭环语义）

| 骨架节点 | 闭环语义 | 实现 |
|---|---|---|
| `Gap识别` | 假设提出 | 从 gap 生成 `Hypothesis` 列表（pydantic Event） |
| （新增）`假设互评` | 互评 | 对 `Hypothesis` 打分/淘汰，低分不进算力 |
| `证据核验` | 实验验证（双路） | RAG 证据 + 插件调用（模型/软件） |
| （闭环）`沉淀` | 共享成败 | 写 KG + 增量重训 + 共享记忆 |

> 新增两个节点（`假设互评`、`沉淀`）让 10 节点骨架扩展为 12 节点，但骨架的确定性先后顺序不变——AutoScientists 语义是「节点语义」而非「自由群聊」。

## 六、共享结构化状态 + 证据对象

```python
ResearchState {
  task, candidates, parsed, entities,
  evidence: [Evidence],   # 证据对象，全图共享货币
  hypotheses,             # AutoScientists：候选假设（互评前后）
  gaps, verified, predictions, report
}

Evidence { paper_id, page, snippet, value, unit, conditions, provenance, authority }
```

对接 Epistemic Layer（答案绑证据 + provenance/authority），「每个结论可追溯」成为默认。

## 七、双路召回 + 插件（既调模型又调软件）

- **模型插件**：flatband/GBDT/GNN（`backend="inprocess"`，joblib 权重）。
- **软件插件**：LAMMPS/DFT（`backend="subprocess"`，如 `research/plugins/lammps_plugin.py`）。
- 统一经 `PluginRegistry.discover(plugin_dir)` 加载，`to_llamaindex_tools()` 桥进 FunctionAgent；失败映射到 `RetrievalStatus.SOURCE_UNAVAILABLE` → abstain 不编造。

## 八、闭环沉淀

核验通过 → `record_claim`/`record_evidence`（schema 现役）→ 增量重训模型 → 下次研究用更全知识 + 更准模型。

## 九、实施顺序

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | `Workflow` 骨架 + 10→12 节点 + `ResearchState` + 条件循环 | loop 可跑通（端到端空跑） |
| P1 | `Evidence` 贯穿全图 + 假设互评（AutoScientists 语义） | 满足「文献溯源」评审 |
| P2 | 双路召回（插件接入模型+软件）+ 闭环沉淀 | 验证结果写回 KG/模型 |
