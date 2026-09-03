# Research Loop 设计（编排闭环 · 最后一部分）

> 三合一架构第三层：**RAG（读文献）+ 插件（跑模型/软件）+ loop（编排闭环）**。
> 定调（2026-08-13）：**确定性骨架 + 借 AutoScientists 闭环语义**。
> 参考：mims-harvard/AutoScientists（自组织 agent 团队 + 提案互评 + 共享成败）。
> **实现演进（2026-08-15）**：本文的「12 节点骨架 + T1-T6 对齐」之上，讨论层已落地（`loop/discussion.py` 多 critic 并发 + 非作者门 + queue claim + roster + Mode Selector），现役 13 节点 + 4 角色。**完整现役架构见 [docs/drbrain-architecture.md](drbrain-architecture.md)。**

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
- **软件插件**：LAMMPS/DFT（`backend="subprocess"`，外部插件，如 `lammps_plugin.py`）。
- 统一经 `PluginRegistry.discover(plugin_dir)` 加载，`to_llamaindex_tools()` 桥进 FunctionAgent；失败映射到 `RetrievalStatus.SOURCE_UNAVAILABLE` → abstain 不编造。

## 八、闭环沉淀

核验通过 → `record_claim`/`record_evidence`（schema 现役）→ 增量重训模型 → 下次研究用更全知识 + 更准模型。

## 九、实施状态（2026-08-13 已全部完成）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | `Workflow` 骨架 + 12 节点 + `ResearchState` + 条件循环 | ✅ |
| P1 | 假设互评（AutoScientists 语义）+ 结构化 JSON 输出 | ✅ |
| P2 | 双路召回（插件）+ 闭环沉淀（settle → claims 表） | ✅ |

实现相对设计的增量：4 个自由节点（retrieve / gap / 互评 / 核验）是 **agent-backed**（内部跑 `build_node_agent` + `run_agent` 的 FunctionAgent，非手写 stub）；agent 经 `run_agent_json` 返回结构化 JSON；新增 **通用 MCP 接入**（`rag/mcp_tools.py`，`build_agent(mcp_servers=...)` 连接任意 stdio MCP server）；闭环沉淀落 `settle` 节点写回 KG `claims` 表。

**autoresearch 集成测试增量（2026-08-13，`tests/integration/` + `run_flatband.py`）**：为让三层体系脱离人工辅助自主运转，`retrieve` 改为**确定性路径**（agent 蒸馏关键词 → 本地 KG 插件 `search_papers` 直接检索，不再解析 agent 自由文本）；`report` 增加**确定性模板回退**（agent 报告优先，空则落结构化模板）；新增**本地 KG data 插件**（`local_kg_search.py`，读真实 schema `papers`/`fulltext`/`concept_nodes`/`concept_cooccurrence`/`paper_citations`）；`search_papers` 用**逐步短语匹配**（多词 query 逐词退让命中）；`build_bm25_index` 加**按库路径缓存**；loop 每步超时 45s→600s（agent 节点多轮 LLM）。

## 十、持续研究内核（AutoScientists 语义，2026-08-13 补齐）

对齐 AutoScientists 的「24 小时持续研究一个课题」内核，而非一次性流水线：

- **`ResearchDirector`（`loop/director.py`）**：把 12 节点 Workflow 当作**一轮 cycle**，反复运行直到停滞。工作区用**文件分离**（镜像 AutoScientists，非单一 JSON）：`champion.md`（冠军结论）/ `dead_ends.md`（已否定）/ `knowledge/patterns.md`（winning patterns + dead ends + exhausted axes）/ `results/cycle-NNN.md`（逐轮证据）/ `run.json`（仅运行态：cycles/no-gain/adaptations，可续跑）。
- **可证伪假设（falsifiable）**：`Hypothesis` 带 `prediction`（什么证据支持）+ `falsification`（什么证据证伪），`identify_gaps` 提出、`verify` 判定。
- **三态分类（KEEP/DISCARD）**：`verify` 返回 `verified`（证据支持→champion）+ `falsified`（证据证伪→dead ends）+ `predictions`；未判定者**留作 unresolved**（下轮带 prior_context 重试），不误判为死路。`settle` 落 `claims`：verified→`Conclusion`、falsified→`Rejected`（负结论也是知识）、prediction→`Prediction`，并各写 `evidence` 行。
- **Phase 4 停滞转向（adapt）**：连续 N 轮无新结论 → 记录耗尽方向为 dead end、重置 no-gain 计数、**转向继续**（非直接停），直到 `max_adaptations` 次转向才停。
- **共享记忆跨轮注入**：每轮把「已确认结论 + 已否定假设」作为 `prior_context` 注入 `identify_gaps`。
- **自扩展（agent 自写代码）**：`run_python` 插件——agent 遇到没有现成工具的缺口时**自己写 Python 现算**，泛化到任意方向（预置插件只是种子，不是全集）。
- **域无关原则**：loop 节点的 prompt **不写死任何领域工具名/领域术语**，领域知识经 **task（运行时）+ skill（`read_skill` 读 `skills/*/SKILL.md`）** 注入——换研究方向不改源码。材料学依赖（ase/gpaw）在 `[dependency-groups] materials` 独立隔离，不进主依赖。

**T1-T6 对齐 AutoScientists（2026-08-14）**：在「持续研究内核」之上，按 `docs/autoscientists-gap-report.md`（源码取证差距）与 `docs/autoscientists-alignment-plan.md`（原子书）补齐了**角色分化与互验闭环**——T1 critic/verifier 独立 role prompt（`loop/roles.py`）；T2 verify 消费 critique 分数；T3 Supports/Refutes/Orthogonal 计数代码化（证伪不再 LLM 自由裁量）；T4 实算门（`run_python(mode=async)` 落盘 + `job_id` 作业文件校验，杜绝编造数值）；T5 proposal 评审门（低分 DISCARD）；T6 查重门；T7 per-agent 记忆（critic/verifier 跨轮记忆文件 + 注入）+ janitor 复核（实算作业超时标记）。现状以 `docs/loop-current-state.md` 为准。


