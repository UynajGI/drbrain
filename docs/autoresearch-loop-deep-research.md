# Autoresearch Loop 深度调研与生产化方向

> 调研日期：2026-08-25
> 范围：DrBrain `src/drbrain/loop/`、RAG、Model-as-Tool、MCP；AutoScientists、AutoDiscovery（arXiv:2507.00310）及相关开源系统。
> 约束：SQLite-only；公共 API、CLI、配置和现有持久化文件只增不改；本报告本身不修改运行代码。
>
> 实施更新（2026-08-25）：PR 1 已加入独立的 SQLite/WAL durable run ledger。下文的「当前缺口」均保留为调研基线；其中 run-level/cycle-boundary 账本缺口已部分关闭，节点 checkpoint、工具轨迹与 evidence lineage 仍未实现。

## 1. 结论

DrBrain 现有 loop 已经不是空壳：它有 13 节点 LlamaIndex Workflow、analyst/critic/compute/verifier 四角色、非作者评论门、研究队列、跨轮记忆、KEEP/DISCARD、实验日志和整轮 trace。RAG、图工具、插件和 MCP 也已经可以组装进 `FunctionAgent`。

但它目前更接近“有确定性顺序的单进程研究 pipeline”，还不是可长期运行的生产级 autoresearch runtime。最关键的缺口不是再增加角色或 prompt，而是：

1. 已有 cycle 级 durable run ledger，但还没有节点级 checkpoint/resume；崩溃后不能从未完成节点续跑。
2. RAG、LLM、插件和 MCP 调用没有统一的 durable intent/observation 轨迹。
3. 证据、假设、实验、工具调用和产物之间缺少稳定 ID 与结构化 lineage。
4. 权限、预算、审批、幂等和副作用恢复没有收敛到一个执行网关。
5. AutoDiscovery 式搜索策略尚未与真实性验证解耦。

目标架构应拆成两层：

```text
确定性研究内核
  状态机 / SQLite 账本 / policy / budget / approval / checkpoint / lineage
                     ↑ 只接受结构化 proposal
可替换策略层
  Analyst / Critic / Compute / Verifier / RAG / Tools / MCTS / Surprise
```

LLM 和 agent 只能提出动作、评价证据或选择候选；普通代码负责校验状态、权限、预算和 schema，并决定是否真正执行和迁移状态。

## 2. DrBrain 当前真实状态

### 2.1 已实现

| 能力 | 当前实现 | 评价 |
|---|---|---|
| 确定性主流程 | `loop/workflow.py` 的 13 个 `@step` | 主链清晰，可保留 |
| 角色分工 | analyst、critic、compute、verifier 使用不同 system prompt | 有角色语义，但仍共享同一进程与工具面 |
| 讨论门 | `MessageBoard` + 多 critic + 非作者评论后入队 | 已实现 AutoScientists 的关键语义 |
| 工作队列 | `ResearchQueue` claim/version | 进程内有效，尚非 durable lease |
| 多轮编排 | `loop/director.py` | 有 champion、dead ends、adaptation、stagnation |
| Durable run ledger（PR 1） | `loop/store.py`、`state.py`、`transitions.py` | 新 run 的 cycle/step/attempt/event 进 SQLite/WAL；旧工作区文件保持兼容投影 |
| 人类可读产物 | `task.md`、`champion.md`、`dead_ends.md`、`results/*.md` | 应继续兼容输出 |
| 整轮审计 | `traces/cycle-NNN.json`、`logs/experiments.jsonl`、`sessions.jsonl` | 只能审计已完成轮次 |
| RAG 工具 | `rag.agent._build_retrieval_tool()` 暴露 `search_documents` | 能检索 chunk，但 loop 的默认检索仍主要走 `search_papers` |
| 图工具 | `build_agent()` 装配图查询与 `kg_validate` | 可直接复用 |
| Model-as-Tool | `PluginRegistry.to_llamaindex_tools()` | 有 schema、timeout、状态和基础 provenance |
| MCP | allowlist、trusted 标记、调用 timeout | 基础边界存在 |

### 2.2 关键断点

1. `ResearchState` 仍主要存在于 LlamaIndex `Context.store`。PR 1 已在 cycle 边界记录 step/attempt 与可重放状态快照，但节点中途崩溃仍没有 `checkpoint_cursor`，不能跳过已完成节点（PR 2）。
2. `Evidence` 只有 paper/page/snippet/value 等字段，没有稳定 `evidence_id`、内容 hash、RAG generation、producer attempt 或 claim relation。
3. `Hypothesis`、`Verification` 没有稳定 ID；同一语句被当作关联键，难以处理改写、分支和多次实验。
4. 插件虽生成 `plugin/version/input/timestamp` evidence，但没有 `tool_call_id`、代码 digest、idempotency key、side-effect class、approval 或 artifact lineage。
5. MCP 信任策略存在，但 loop 通过 `build_agent()` 时没有显式要求 `require_trusted_mcp=True`；生产 loop 应 fail closed。
6. `MessageBoard` 与 `ResearchQueue` 仍是单轮进程内对象。PR 1 的 ledger 只记录 cycle/step/attempt 生命周期，尚不持久化未完成 claim 或讨论状态。
7. 新 run 已以 `ledger.sqlite3` 作为可事务更新、可重放的 cycle 级 canonical ledger；`run.json`、Markdown、cycle JSON 和 JSONL 保留为兼容投影。工具、证据与 claim 的 durable 表仍在后续阶段。
8. LLM 生成的报告可以覆盖确定性模板，但报告中的 claim 尚未逐条绑定 evidence。

## 3. 外部参考结论

### 3.1 AutoScientists

取证版本：[`c71a923`](https://github.com/mims-harvard/AutoScientists/commit/c71a92343b9a488ed10134be805845b9473ad18f)，2026-05-28 初次公开提交。仓库没有 tag/release，也没有可识别的 LICENSE；只借鉴机制，不复制源码或协议文本。

它最值得借鉴的是：

- proposal → non-author review → queue → claim → result 的完整轨迹；
- 乐观并发、stale claim、pending result 恢复；
- monitor/analyst/GPU runner 职责隔离；
- champion、dead ends、exhausted axes 与跨会话记忆；
- 多 seed/noise gate 后才晋升 champion。

不应照搬：

- 用 Markdown prompt 充当状态机；
- agent 直接写 canonical state；
- 多份 workspace 文件和日志共同承担事实源；
- 依靠角色说明阻止越权；
- 未明确授权的实现复制。

### 3.2 AutoDiscovery（arXiv:2507.00310）

论文当前为 [v3（2026-02-12）](https://arxiv.org/abs/2507.00310v3)，已接收 NeurIPS 2025。核心机制是：

- 明确的 hypothesis → experiment plan → programmer → executor → analyst → reviewer/reviser FSM；
- 程序实现最多 6 次修复，实验方案最多 1 次修订；
- 用实验前后 LLM belief 的 Beta 分布变化计算 Bayesian surprise；
- MCTS + UCT + progressive widening 在固定预算内平衡探索与利用；
- 对语义重复假设做 embedding + LLM HAC 去重。

论文在 21 个数据集、每次 500 个假设预算下报告，比基线多产生 5%–29% 的 LLM-surprising discoveries；约 67% 也被领域专家认为 surprising。但作者同时明确报告：错误安装依赖会破坏环境、祖先超过约 150 层会污染上下文、模型会生成数据不存在的属性、LLM surprise 不等于人类 novelty。

因此 DrBrain 可把 surprise、novelty、utility、expected information gain 作为调度优先级，但不能把它们当作 `verified`。科学结论仍必须来自可复现的实验结果、数据校验、文献证据和独立 verifier。

### 3.3 其他项目

| 项目 | 许可/版本状态 | 借鉴点 | 不整栈采用的原因 |
|---|---|---|---|
| [Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) | MIT；取证 `d9017d9` | 阶段门、角色分工、human-in-loop、checkpoint | pickle checkpoint 不适合长期兼容；执行边界不足 |
| [OpenHands software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) | MIT；取证 `750b14f` | append event log、action/observation、unmatched action、branch/replay、风险确认 | 通用软件 agent runtime 太重，缺科学语义 |
| [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | 受限 source-code license；取证 `96bd516` | parent-child journal、tree search、阶段 checkpoint、多 seed | 许可证受限；pickle 与直接代码执行风险；强绑定 ML |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | README 声明 MIT；取证 `228791f` | 固定预算、不可修改评测器、keep/discard、隔离分支 | `git reset + TSV` 不是生产状态系统 |
| [AutoDiscovery 源码](https://github.com/allenai/autodiscovery-neurips) | 未发现项目级 LICENSE；取证 `66aa2d5` | FSM、MCTS 节点、per-node JSON、resume | 动态执行/装包风险；无审批和副作用模型 |

## 4. 目标架构

### 4.1 核心对象

```text
ResearchTask
  └─ ResearchRun  ── pinned policy / model / tool registry / RAG generation / budget
       ├─ Step ── Attempt ── ToolCall ── Observation
       │                      ├─ Artifact
       │                      └─ Evidence
       ├─ Hypothesis ── Experiment ── Verification
       └─ Approval

Append-only RunEvent → deterministic projections → current state
```

建议使用 SQLite WAL，不引入外部数据库。事件、当前投影、预算结算和 outbox 在同一事务内提交；现有 Markdown、JSON 和 JSONL 继续作为兼容投影输出，不再作为唯一事实源。

### 4.2 最小数据模型

| 实体 | 必要字段 |
|---|---|
| `ResearchRun` | `run_id`、task、status、state_version、parent_run_id、policy/model/tool manifests、RAG generation、budget、timestamps |
| `Step` | `step_id`、run_id、kind、dependencies、status、priority、lease owner/expiry、input/output bundle |
| `Attempt` | `attempt_id`、step_id、attempt_no、status、failure class、idempotency key、environment digest、seed、checkpoint cursor |
| `RunEvent` | run-local sequence、entity/version、event type、actor/principal、trace ID、payload、timestamp |
| `ToolCall` | tool/plugin/version/code digest、normalized args、side effect、principal、policy decision、approval、external operation ID、status |
| `Observation` | tool call、status、structured result、stdout/stderr digest、usage、result artifact、timestamps |
| `Artifact` | content hash、media type、size、producer attempt、parent artifacts、schema/version、storage URI |
| `Evidence` | immutable ID、source、content hash、RAG generation、document/chunk、query、retriever/rank/score、trust、producer attempt |
| `ClaimEvidence` | claim/hypothesis ID、evidence ID、`supports/refutes/orthogonal/insufficient`、evaluator、confidence |
| `Approval` | subject、requested capability、payload hash、policy rule、requester、approver、decision、scope、expiry |
| `BudgetLedger` | reservation/settlement、tokens、money、wall time、CPU/GPU、tool/RAG/model calls、artifact bytes |

### 4.3 状态机

```text
Step:
PENDING → READY → CLAIMED → RUNNING
RUNNING → WAITING_APPROVAL → RUNNING
RUNNING → SUCCEEDED | FAILED | TIMED_OUT | UNKNOWN
UNKNOWN → RECONCILING → SUCCEEDED | FAILED | MANUAL_REVIEW

ToolCall:
PROPOSED → POLICY_CHECKED
POLICY_CHECKED → APPROVAL_REQUIRED | AUTHORIZED | REJECTED
AUTHORIZED → EXECUTING → SUCCEEDED | FAILED | TIMED_OUT | UNKNOWN
```

`UNKNOWN` 不能省略。进程崩溃时，外部调用可能已经成功但 observation 尚未写入；此时必须按 `external_operation_id/idempotency_key` reconcile，禁止盲目重试。

### 4.4 LlamaIndex 的位置

当前依赖是 `llama-index-core>=0.14.23`。LlamaIndex 已提供 `Context.to_dict()/from_dict()`、`WorkflowCheckpointer.run_from()`、human-in-the-loop event 和 instrumentation；这些能力适合用作 workflow 执行与 checkpoint payload，但不能取代 DrBrain 的持久账本：

- checkpoint 自身要写入 SQLite，并绑定 run/step/attempt/version；
- 恢复前校验 workflow、policy、tool registry、model 和 RAG generation 是否兼容；
- replay 默认只重建投影，不重新执行外部副作用；
- LlamaIndex event/span ID 映射到 DrBrain trace ID，而不是成为业务主键。

参考：[Workflow checkpointing](https://github.com/run-llama/llama_index/blob/v0.14.6/docs/examples/workflow/checkpointing_workflows.ipynb)、[human-in-the-loop](https://github.com/run-llama/llama_index/blob/v0.14.6/docs/src/content/docs/framework/understanding/agent/human_in_the_loop.md)、[instrumentation](https://github.com/run-llama/llama_index/blob/v0.14.6/docs/src/content/docs/framework/module_guides/observability/instrumentation.md)。

## 5. Tool、RAG 与插件如何进入 loop

### 5.1 统一 ToolBroker

所有 graph tool、RAG、Model-as-Tool 和 MCP 都必须经过同一条执行链：

```text
Agent 提交 ToolProposal
→ schema validation
→ policy/capability evaluation
→ budget reservation
→ optional approval
→ durable ToolCall intent
→ executor
→ durable Observation + Artifact/Evidence
→ budget settlement
→ state transition
```

现有 `Plugin` 只增可选字段，保持构造兼容：

- `code_digest`
- `side_effect = pure | read | write | irreversible`
- `required_capabilities`
- `allowed_paths/domains/datasets`
- `secret_refs`
- `max_output_bytes`、`cost_hint`
- `supports_idempotency/reconcile/cancel`
- `sandbox_profile`
- `approval_policy`

Model-as-Tool 额外记录 provider/model、prompt-template digest、temperature、seed、token usage 和 response schema。模型工具不会因为是“模型”而获得更高证据权威。

### 5.2 RAG Evidence Plane

- 每个 run 固定一个或多个 `index_generation_id`，运行中不静默切换。
- RAG 返回内容按不可信数据处理；文档文本无权触发工具。
- 每个 hit 记录 query、filter、retriever、rank、score、document/chunk、generation、content hash。
- `search_documents` 只增 `evidence_id` 和 `index_generation_id` 字段，保留现有字段。
- claim/hypothesis 必须通过 `ClaimEvidence` 绑定支持、反驳或不足证据。
- RAG 默认只有 read capability；ingest、delete、publish generation 是独立 write capability。

### 5.3 角色工具面

| 角色/节点 | 默认能力 |
|---|---|
| planner/analyst | RAG read、graph read；不能执行有副作用工具 |
| critic | 只读输入 bundle；需要外部证据时提交 retrieval proposal |
| experiment planner | 查询 tool manifests；只生成实验计划 |
| compute | 经审批的 compute/plugin；限制路径、网络、预算和输出 |
| verifier | 读取 evidence/artifact；不可修改原实验产物 |
| reporter | 只读已结算 claim/evidence；不得自行补证据 |
| monitor/director | lease、reconcile、budget、pause/cancel；不生成科学结论 |

生产 loop 对 MCP 必须显式 `require_trusted_mcp=True`，且每个 server 使用非空 allowlist。

## 6. 开放式搜索

MCTS 不是第一阶段基础设施，而是 `SearchPolicy`：

```text
SearchPolicy.select(frontier, budget, scores) -> hypothesis_id
SearchPolicy.expand(parent, context) -> proposals
SearchPolicy.observe(hypothesis_id, verification) -> reward update
```

第一版保留 FIFO/priority baseline；之后并列增加 greedy、beam、MCTS progressive widening。`surprise`、novelty、utility、evidence gap 和 expected information gain 是独立评分维度，最终 `Verification.status` 不读取这些调度分数。

所有 search node 都持久化 parent、children、visit count、reward components、budget cost、hypothesis/experiment IDs。长路径上下文只通过检索式摘要构造，不把全部祖先对话塞回 prompt。

## 7. 分阶段实施

### Phase 1：Durable Kernel

新增 SQLite 表、repository、状态迁移服务、run/step/attempt/event、CAS lease、checkpoint 和 crash-resume。先用 fake executor 验证状态与恢复，不接真实副作用工具。

验收：任意步骤 kill 后可恢复；无重复完成；同一事件流重放得到相同投影；旧 `run.json`/Markdown/JSONL 继续生成。

### Phase 2：Tool/Plugin Gateway

新增 ToolBroker、policy、approval、intent/observation、budget、idempotency 与 artifact；现有 plugin/MCP/graph tools 通过 adapter 接入。

验收：read/write/irreversible 权限可区分；未授权调用不进入 executor；UNKNOWN 调用进入 reconcile；production MCP fail closed。

### Phase 3：RAG Evidence Plane

固定 run 的索引代际；检索 hit 生成 Evidence；建立 claim→evidence 关系和引用校验。

验收：每个结论能回到 generation/chunk/query/tool call；旧 RAG 输出仍是新输出的子集。

### Phase 4：规范科研闭环

将现有 13 节点迁移到 durable step contracts：

```text
retrieve → hypothesize → non-author critique → experiment plan
→ policy/approval → execute → analyze → independent verify → settle
```

验收：失败、拒绝、证据不足和验证冲突都有明确状态；所有实验、输出和结论都有 lineage。

### Phase 5：开放式搜索与并发

引入 pluggable search policy、MCTS/PW、多分支并行、resource lease、champion CAS 和多 seed/noise gate。

验收：在相同预算下与 FIFO/greedy/beam 消融；收益按有效且可复现 discovery 计算，不按调用次数或 surprise 单项计算。

### Phase 6：长期运行与治理

增加 pause/resume/cancel、人工审批入口、审计导出、成本看板、策略迁移与跨 run 知识蒸馏。

## 8. 第一条实现分支的边界

第一条分支只做 Phase 1，且主体必须是架构代码，不是测试堆积：

- 新增 durable store schema 与 repository；
- 新增 run/step/attempt/event 模型和 transition service；
- 接入 LlamaIndex checkpoint serialization；
- director 以兼容方式双写 ledger 与现有文件投影；
- 新增只读查询 API/CLI，不修改已有命令签名；
- 测试只用于证明 crash-resume、CAS、replay 和兼容性。

明确不做：MCTS、多 agent 动态扩容、UI、外部数据库、任意 Python 执行、旧 API 删除或重命名。

## 9. 评测

系统可靠性与科研质量分开评测。

系统指标：非法迁移、orphan/unmatched ToolCall、重复副作用、crash 恢复率、stale lease 回收、审批绕过、证据完整率、replay 一致性、有效发现成本。

科研指标：experiment feasibility、implementation faithfulness、evidence support precision、hypothesis uniqueness、falsification rate、多 seed 可复现率、verifier/专家一致性、fixed-budget discovery、RAG ablation、search-policy ablation。

## 10. API 兼容性

所有升级遵守 additive-only：

- 新增表、字段、事件类型、可选参数、命令和只读视图；
- 保留现有 `ResearchLoopWorkflow`、`ResearchDirector`、CLI 参数和 workspace 文件；
- 旧文件成为 ledger 的兼容投影，格式原字段不删除、不改名；
- `Plugin` 新字段全部有默认值；现有外部插件无需修改即可运行；
- 新结果可以增加 IDs、trace、evidence 和 budget 字段，旧字段语义不变；
- MCTS、approval、durable mode 先以 opt-in 配置发布，稳定后再讨论默认值。

## 11. 参考

- [AutoScientists](https://github.com/mims-harvard/AutoScientists)
- [AutoDiscovery paper](https://arxiv.org/abs/2507.00310v3)
- [AutoDiscovery source](https://github.com/allenai/autodiscovery-neurips)
- [Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)
- [OpenHands software-agent-sdk](https://github.com/OpenHands/software-agent-sdk)
- [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [ResearchAgent](https://arxiv.org/abs/2404.07738v2)
