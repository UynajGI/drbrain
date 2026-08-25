# Autoresearch Loop 生产化实施计划

> 日期：2026-08-25
>
> 依据：[Autoresearch Loop 深度调研与生产化方向](autoresearch-loop-deep-research.md)
>
> 约束：SQLite-only；公共 API、CLI、配置和现有持久化文件只增不改。

## 1. 目标

把现有单进程、整轮落盘的 research loop 升级为可长期运行的可靠闭环：

```text
检索证据
→ 提出可证伪假设
→ 非作者评审
→ 制定实验计划
→ 工具策略与审批
→ 执行
→ 独立核验
→ KEEP / DISCARD / INSUFFICIENT
→ 结论、证据和产物沉淀
```

完成后必须满足：

1. 任意步骤中断后可以从最近的可靠 checkpoint 恢复，不重复已完成的副作用。
2. 每个 run、step、attempt、tool call、evidence、artifact 和 claim 都有稳定 ID。
3. 每条结论都能追溯到检索片段、工具调用、实验产物和核验结果。
4. RAG、图工具、Model-as-Tool 和 MCP 经过统一权限、预算、记录和失败处理。
5. Agent 只能提出动作，状态迁移、权限判定、预算结算和 KEEP/DISCARD 由代码控制。
6. 现有 `ResearchDirector`、`ResearchLoopWorkflow`、插件接口和 workspace 文件继续工作。

## 2. 范围锁定

### 2.1 本计划包含

- SQLite durable run ledger。
- run/step/attempt 状态机、lease、checkpoint 和恢复。
- 统一 ToolBroker，以及插件/MCP/图工具的策略边界。
- RAG evidence binding 与索引代际固定。
- 假设、实验、证据、产物和结论 lineage。
- 人工审批、预算、暂停/恢复/取消和审计查询。
- 现有 13 节点 workflow 的兼容迁移。
- 故障恢复、可追溯性和科研质量验收。

### 2.2 本计划不包含

- AutoDiscovery 式搜索策略。
- MCTS、beam search、progressive widening 或 surprise reward。
- 动态扩展 agent 数量或自组织团队调度。
- 外部数据库、消息队列或分布式调度器。
- 整套引入 OpenHands、AutoScientists 或其他 agent runtime。
- 内容寻址对象存储；产物继续使用现有文件系统，只记录必要校验值。
- 任意执行模型生成代码或运行时任意安装依赖。
- UI。

上述内容在本计划全部完成并通过验收前不得进入实现分支。

## 3. 已有基础与必须修复的断点

### 3.1 保留

- `ResearchLoopWorkflow` 的 13 节点顺序和 Pydantic event。
- analyst、critic、compute、verifier 四角色。
- `MessageBoard`、非作者评论门和 `ResearchQueue` 语义。
- director 的 champion、dead ends、adaptation 和 stagnation。
- `Plugin` / `PluginResult` / `PluginRegistry`。
- RAG `search_documents`、图工具和 MCP allowlist/timeout。
- `run.json`、`champion.md`、`dead_ends.md`、`results/`、`knowledge/`、`traces/` 和 `logs/`。

### 3.2 修复

| 断点 | 目标状态 |
|---|---|
| `ResearchState` 主要在进程内 | 每步输入、输出和状态持久化 |
| 只在整轮结束后写 trace | 每次 step/attempt/tool call 都有 durable 记录 |
| queue/board 是内存对象 | claim、review 和 pending 可恢复 |
| 工具调用无统一 ID | intent 与 observation 一一对应 |
| 外部调用崩溃后结果不确定 | `UNKNOWN → RECONCILING`，禁止盲重试 |
| 证据靠文本或语句关联 | 稳定 ID + 结构化 supports/refutes/insufficient |
| RAG 代际未绑定 run | run 创建时固定 generation |
| agent 获得宽泛工具面 | 每个 step 明确 capability allowlist |
| 多份文件承担事实源 | SQLite ledger 为事实源，文件是兼容投影 |

## 4. 固定架构决策

### 4.1 存储位置

在 `ResearchDirector.run_dir` 下新增一个独立 SQLite 文件：

```text
workspace/autoresearch/
├── ledger.sqlite3
└── <topic>/
    ├── run.json
    ├── champion.md
    ├── dead_ends.md
    ├── results/
    ├── traces/
    ├── logs/
    ├── jobs/
    └── artifacts/
```

选择独立 ledger，而不是修改知识库主库：

- loop 目前允许 `db=None`，不能把运行能力绑定到知识库连接。
- run_dir 已是 autoresearch 生命周期边界。
- 一个 WAL 数据库可以协调同一 run_dir 下多个 topic。
- 归档或迁移 autoresearch workspace 时不会污染知识图谱 schema。

### 4.2 事实源与兼容投影

- `ledger.sqlite3` 是新 run 的 canonical state。
- 现有 Markdown、JSON 和 JSONL 路径继续生成，字段不删除、不改名。
- ledger 事务提交后再更新文件投影；投影写入中断时，下次启动按 `last_projected_event` 重建。
- 已存在但没有 ledger 的旧 topic，在首次运行时从现有文件生成一条 `legacy_snapshot_imported` 事件；不改写历史文件。

### 4.3 最小数据模型

Phase 1–4 只实现以下表，不提前增加搜索树或分布式调度表：

| 表 | 责任 |
|---|---|
| `research_runs` | topic、状态、版本、固定配置、预算、投影游标 |
| `research_steps` | 节点、依赖、状态、lease、输入/输出引用 |
| `research_attempts` | 重试次数、失败分类、checkpoint、环境与 seed |
| `research_events` | run 内单调序号、actor、事件、payload、trace |
| `research_checkpoints` | LlamaIndex Context 快照及兼容 manifest |
| `research_attempt_progress` | 内部节点开始/安全边界；阻止中断外部副作用节点被自动重放 |
| `research_tool_calls` | intent、policy、状态、外部 operation ID、usage |
| `research_approvals` | 危险调用的请求与决定 |
| `research_artifacts` | 文件路径、类型、大小、校验值、生产 attempt |
| `research_evidence` | RAG/tool/experiment evidence 元数据 |
| `research_claim_evidence` | claim 与 evidence 的支持、反驳或不足关系 |

SQLite 约束至少包括：

- `UNIQUE(run_id, event_seq)`
- `UNIQUE(step_id, attempt_no)`
- 一个 step 同时最多一个未过期 lease
- 一个 tool intent 最多一个 terminal observation
- claim/evidence 关联不可指向其他 run 的私有对象

### 4.4 状态机

```text
Run:
CREATED → RUNNING → PAUSED | SUCCEEDED | FAILED | CANCELLED
PAUSED → RUNNING | CANCELLED

Step:
PENDING → READY → CLAIMED → RUNNING
RUNNING → WAITING_APPROVAL → RUNNING
RUNNING → SUCCEEDED | FAILED | TIMED_OUT | UNKNOWN
UNKNOWN → RECONCILING → SUCCEEDED | FAILED | MANUAL_REVIEW

ToolCall:
PROPOSED → AUTHORIZED | APPROVAL_REQUIRED | REJECTED
APPROVAL_REQUIRED → AUTHORIZED | REJECTED
AUTHORIZED → EXECUTING → SUCCEEDED | FAILED | TIMED_OUT | UNKNOWN
UNKNOWN → RECONCILING → SUCCEEDED | FAILED | MANUAL_REVIEW
```

所有迁移由 `TransitionService` 执行。Agent、plugin handler 和文件 projector 不能直接更新状态列。

## 5. 实施顺序

每个 PR 必须可独立合并；后一个 PR 不反向扩大前一个 PR 的范围。

### PR 1：Durable Run Ledger

目标：建立可靠事实源，不改变现有 workflow 行为。

**状态（2026-08-25）：已合入 PR #21。** 实现范围严格停在 cycle 边界：`research_runs`、`research_steps`、`research_attempts`、`research_events` 进入独立 SQLite/WAL ledger；未完成 cycle 在下一次启动时记为 `unknown`，不会伪装成节点级 checkpoint/resume。

主要改动：

- 新增 `loop/store.py`：SQLite 连接、WAL、schema version 和事务。
- 新增 `loop/state.py`：run/step/attempt 状态与允许迁移。
- 新增 `loop/transitions.py`：唯一状态写入口。
- director 创建或加载 run，并把 cycle 边界写入 ledger。
- 新增兼容 projector，继续输出现有文件。
- 旧 workspace 首次运行只导入摘要，不重写历史。

验证：新增 `tests/test_loop_ledger.py` 覆盖非法迁移、投影中断后的重放、legacy snapshot 一次性导入和中断 cycle 的 `unknown` 审计；既有 director、discussion 与 agent-loop 测试保持通过。

验收：

- 新 run 能完整写入 ledger 和旧文件。
- 非法迁移被拒绝且不产生半写状态。
- ledger 提交后文件投影失败，重启能重建投影。
- 现有构造函数和导入路径不变。

明确不做：工具调用记录、RAG evidence、审批、workflow step 恢复。

### PR 2：Step Checkpoint 与 Crash Resume

目标：从整轮恢复升级为步骤级恢复。

**状态（2026-08-25）：已实现，待 CI 与 PR 审查。** Context 以 JSON 持久化到独立 ledger；内部节点开始与完成边界均留有审计。`compute` 中断会转入人工审查，而不会从旧 checkpoint 自动重放外部副作用。

主要改动：

- 新增 `loop/checkpointing.py`。
- 使用 LlamaIndex Context JSON serialization 保存 checkpoint；不使用 pickle。
- checkpoint 绑定 run、step、attempt、workflow version、model manifest、tool manifest 和 RAG generation。
- step 运行前写 attempt，成功后同事务结算 step/event/checkpoint metadata。
- 增加 lease 领取、续租、过期回收和单 writer 约束。
- 将 workflow 自身持有的 message board / research queue（含 hypothesis）一起纳入 checkpoint。
- director 优先恢复未完成 run；无 ledger 时保持现有文件恢复路径。

验收：

- 在每个节点前后故障注入，恢复后不重跑已成功节点。
- 同一 step 不会被两个 worker 同时完成。
- checkpoint manifest 不兼容时进入明确失败或人工处理，不静默加载。
- 完成 run 的最终 Markdown/JSONL 与无中断运行一致。

明确不做：真实外部副作用的自动恢复。

### PR 3：统一 ToolBroker

目标：让图工具、插件、Model-as-Tool 和 MCP 共用一个受控执行边界。

主要改动：

- 新增 `loop/tool_broker.py`：proposal、schema 校验、policy、intent、执行、observation。
- 新增 `loop/policy.py`：step capability、side-effect 和审批规则。
- `build_agent()` 只增加可选的 broker/policy 参数；默认值保持旧行为。
- `Plugin` 只增加带默认值的可选元数据：side effect、capabilities、code/version digest、idempotency/reconcile/cancel、resource scope。
- 旧 `PluginRegistry.call()` 保持可用；durable loop 通过 broker 调用。
- autoresearch 的 MCP 显式要求 trusted server 和非空 tool allowlist。
- 每个 workflow step 声明可见工具，而不是共享全部工具面。

默认策略：

| 调用类型 | 默认处理 |
|---|---|
| pure/read | 预算允许时自动执行 |
| write | 需要显式 allowlist；无幂等信息时需要审批 |
| irreversible | 必须审批，不自动重试 |
| unspecified legacy plugin | 旧直接调用保持兼容；durable loop 拒绝，直到显式分类 |

验收：

- 每次执行都先有 durable intent，再有 observation。
- policy 拒绝的调用不会进入 handler。
- read 调用超时可以按策略重试；write/irreversible 超时进入 `UNKNOWN`。
- secret 不写入 event、prompt、artifact 或日志。
- 旧插件无需修改仍可通过原 API 调用。

### PR 4：RAG Evidence Plane

目标：让 loop 的文献检索结果成为可验证、可引用的 evidence，而不是 prompt 文本。

主要改动：

- run 创建时固定 `index_generation_id`。
- `search_documents` 结果只增 `evidence_id`、generation、document/chunk locator 和 content checksum。
- 新增 `EvidenceBundle`，记录 query、filters、retriever、rank、score 和 tool call。
- `ResearchState.evidence` 兼容保留，同时写入 ledger evidence。
- 假设、verification 和报告使用稳定 claim/evidence ID 关联。
- RAG 内容按不可信数据处理；检索文本不能改变工具权限或系统指令。

验收：

- 任意报告 claim 可回到具体 generation、document、chunk、query 和 rank。
- run 期间发布新索引不会改变该 run 的检索代际。
- 没有足够 evidence 的 claim 保持 `insufficient`，不能被报告器升级为 verified。
- 旧 RAG/tool 结果字段保持不变，新字段为纯新增。

### PR 5：检索—假设—评审链 durable 化

目标：迁移 workflow 前半段，不一次性重写 13 节点。

迁移节点：

```text
plan_task → retrieve → filter → parse_pdf → extract
→ normalize → fuse → identify_gaps → critique
```

主要改动：

- 每个节点声明 input/output schema、允许工具、最大 attempt 和 retry class。
- proposal、critic review、queue item 使用稳定 ID。
- 非作者评论门由 TransitionService 校验，不只依赖 prompt 或内存 board。
- in-memory MessageBoard/ResearchQueue 保留为本轮视图，canonical state 写 ledger。
- 重复假设检查使用结构化历史和稳定 ID；文本相似只用于候选匹配，不直接删除记录。

验收：

- crash 后 proposal/review/queue 状态可恢复。
- 未达到非作者门的假设不能进入 compute。
- 同一 proposal 不会重复入队。
- 前半段无需 compute 插件也能稳定完成并留下 evidence trace。

### PR 6：实验—核验—沉淀链 durable 化

目标：完成可靠科研闭环。

迁移节点：

```text
compute → verify → settle → report
```

主要改动：

- 实验计划、代码/配置、环境、seed、输入和输出作为 artifact 记录。
- compute 必须经过 ToolBroker；不允许 workflow 直接调用 handler。
- verifier 只读已结算 evidence/artifact，不能修改实验产物。
- KEEP/DISCARD/INSUFFICIENT 由结构化规则决定；LLM 只能提交计数、评价或建议。
- champion 晋升使用 expected version 做 CAS；结果接近噪声区时要求配置的复验门。
- report 只消费已结算 claim/evidence；agent 文本不能新增未登记结论。

验收：

- 从 claim 可追到 hypothesis、experiment、attempt、tool call、artifact 和 evidence。
- compute 中断不会重复提交已执行的写操作。
- verifier 与 compute 角色隔离。
- 无数值产物或 evidence 的实验不能晋升 champion。
- 旧 champion/dead ends/results 文件继续生成。

### PR 7：运行治理与终态验收

目标：让闭环可操作、可停止、可检查。

主要改动：

- 增加只读 run/status/trace 查询 API。
- 增加 pause、resume、cancel、approve/reject 操作；现有 API 不改。
- 预算控制：tokens、wall time、tool/RAG/model calls、CPU/GPU time、attempts。
- 清理旧的“文件是 canonical state”内部假设，但保留兼容 projector。
- 增加审计导出和运行摘要。
- 同步架构、配置、API 和运维文档。

验收：

- pause 后没有新 step 被领取；resume 从 checkpoint 继续；cancel 不删除证据。
- 超预算进入可解释终态，不继续调用模型或工具。
- 所有 terminal run 都有完整 audit summary。
- 全量兼容测试证明旧公共接口仍是新接口的子集。

## 6. 兼容迁移规则

### 6.1 Python API

- 保留 `ResearchDirector` 和 `ResearchLoopWorkflow` 的现有导入路径。
- 现有参数不重命名、不删除；新增参数全部 keyword-only 且有默认值。
- `Plugin` 新字段有默认值；现有第三方插件无需修改。
- 旧 `PluginRegistry.call()` 行为保持，ToolBroker 是新增入口。

### 6.2 配置

新增配置放在独立 `autoresearch` 段，全部有兼容默认值。首期 durable mode 可 opt-in；启用后也不改变已有字段含义。

### 6.3 持久化文件

- 旧路径和旧字段保留。
- 新 ID、trace、evidence、budget 信息只能追加。
- 兼容 projector 必须幂等；不得通过删目录或覆盖历史来修复状态。
- ledger migration 只前进，不自动降级或销毁数据。

## 7. 验收矩阵

| 维度 | 必须证明 |
|---|---|
| 状态正确性 | 非法迁移为 0；同一 step 不能重复结算 |
| 恢复 | 每个节点故障注入后能恢复；完成结果与无故障运行等价 |
| 副作用 | orphan/unmatched tool call 可发现；重复副作用为 0 |
| 并发 | lease/CAS 冲突有确定结果；stale lease 可回收 |
| 证据 | verified claim 的 evidence coverage 为 100% |
| RAG | run 内索引代际不变化；每个 hit 可定位到 chunk |
| 工具安全 | 未授权调用执行数为 0；production MCP 全部 trusted + allowlisted |
| 预算 | 超预算后新增调用数为 0 |
| 兼容 | 旧 Python API、插件、文件路径和原字段全部保留 |
| 科研质量 | feasibility、faithfulness、可复现率和 verifier 一致性有固定基线 |

测试是以上架构能力的验收证据，不作为独立交付目标。每个 PR 的代码变更必须先形成实际运行路径，再补相应故障与兼容测试。

## 8. 完成定义

本计划只有同时满足以下条件才算完成：

- PR 1–7 全部合并且 CI 通过。
- 至少一个真实任务使用 RAG 和一个受控工具完成全闭环。
- 运行中途强制终止后恢复成功，没有重复工具副作用。
- 最终报告中的每条 verified claim 都能沿稳定 ID 回到 evidence 和 artifact。
- 旧调用方、旧插件和旧 workspace 仍可使用。
- 运维者可以查询、暂停、恢复、取消和审计 run。

完成这些之后，才单独评估 AutoDiscovery 式搜索策略；它不会反向改变本计划建立的状态、工具、证据和治理边界。
