# DrBrain Research Loop — 现状总结（2026-08-14，以磁盘当前内容为准）

> ⚠️ **已过时（2026-08-15）**：本文写于讨论层（`loop/discussion.py` 消息板 MessageBoard + 非作者门 + queue claim）实现之前。所述「12 节点 + 同一个 system prompt 换 user_msg」已被「13 节点 + 4 角色（analyst/critic/compute/verifier 各自独立 system prompt）+ 多 critic 并发讨论层」取代。**现役权威见 [docs/drbrain-architecture.md](drbrain-architecture.md) 第 5 章。**

> 本文只描述**代码现状**，不评判目标状态、不美化。所有结论带 `文件:行号` 证据。
> 涉及四个文件：
> - `src/drbrain/loop/director.py`（ResearchDirector 连续循环）
> - `src/drbrain/loop/workflow.py`（12 节点 Workflow）
> - `src/drbrain/loop/events.py`（ResearchState / Hypothesis 数据类）
> - `src/drbrain/rag/agent.py`（build_agent / FunctionAgent 构建）

---

## 0. 一句话概括

DrBrain 的 research loop 是**一条固定顺序的 12 节点流水线**（LlamaIndex `Workflow`），由 `ResearchDirector` 在外层反复跑这条流水线直到停滞。它不是多个独立 agent 互相对话/互相验证——而是**每轮（cycle）执行一次流水线**，其中 5 个节点内部各自**临时构建一个无状态的 FunctionAgent**（同一个 `build_agent` 模板、同一个 system prompt、同一套工具），用**不同的 `user_msg` 文本**来驱动不同节点的行为。agent 之间的唯一「记忆」是共享的 `ResearchState` 结构化状态 + director 注入的 prior_context 文本。

---

## 1. 12 节点结构

节点定义在 `workflow.py`，顺序为（行号 = 节点函数定义处）：

| # | 节点 | 类型 | 证据 | agent 是否介入 |
|---|------|------|------|----------------|
| 1 | `plan_task` | 确定性 | `workflow.py:286` | 否 |
| 2 | `retrieve` | 半 agent | `workflow.py:296` | 是（仅用于「把任务提炼成检索 query」，检索本体走确定性插件） |
| 3 | `filter` | 确定性（含回检索循环） | `workflow.py:326` | 否 |
| 4 | `parse_pdf` | 确定性（纯拷贝） | `workflow.py:336` | 否 |
| 5 | `extract` | agent-backed | `workflow.py:344` | 是 |
| 6 | `normalize` | 确定性（纯拷贝） | `workflow.py:369` | 否 |
| 7 | `fuse` | 确定性（纯拷贝） | `workflow.py:377` | 否 |
| 8 | `identify_gaps` | agent-backed | `workflow.py:385` | 是 |
| 9 | `critique` | agent-backed | `workflow.py:441` | 是 |
| 10 | `verify` | agent-backed | `workflow.py:476` | 是 |
| 11 | `settle` | 确定性（写 KG） | `workflow.py:512` | 否 |
| 12 | `report` | agent-backed（含确定性模板兜底） | `workflow.py:585` | 是 |

### 1.1 确定性节点（纯代码，6 个 + retrieve 的检索路径）

- `plan_task`：只把 `StartEvent.task` 写进 `state.task`（`workflow.py:286-294`）。
- `filter`：候选为空且未超 `MAX_RETRIEVE_ATTEMPTS`（=3）就发 `RetrieveAgain` 回检索，否则放行（`workflow.py:326-334`）。
- `parse_pdf` / `normalize` / `fuse`：**三个纯转发节点**，只把上一节点输出拷进 state，无任何计算：
  - `parse_pdf`：`state.parsed = ev.selected`（`workflow.py:340`）
  - `normalize`：`state.entities = ev.entities`（`workflow.py:372`）
  - `fuse`：`state.entities = ev.entities`（`workflow.py:380`）
  - 即「解析/规范化/融合」目前是占位（P0 骨架），没有真实逻辑。
- `settle`：把 verified/falsified/predictions 写回 KG 的 `claims` 表（`_persist_claims`，`workflow.py:527-573`），并写证据行（`_record_evidence_for`，`workflow.py:575-596`）。

### 1.2 agent-backed 节点（内部跑 FunctionAgent）

真正会构建并调用 agent 的节点是：`retrieve`（部分）、`extract`、`identify_gaps`、`critique`、`verify`、`report`。

### 1.3 agent 是怎么构建的 —— **同一个 build_agent 模板，换 prompt，不是每个节点一个定制 agent**

- 每个 agent-backed 节点都调用 **同一个** `self.build_node_agent()`（`workflow.py:206-222`），它内部只做一件事：

```python
def build_node_agent(self, *, plugins_dir=None):
    if self._cfg is None:
        return None
    from drbrain.rag.agent import build_agent
    return build_agent(self._cfg, self._db, graph=self._graph,
                       plugins_dir=..., mcp_servers=self._mcp_servers)
```

- 而 `build_agent`（`rag/agent.py:597-654`）的 system prompt 是**写死的常量**：

```python
BASE_SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning assistant. "
    "Use the provided tools to explore the graph and answer questions. "
    "Explain your reasoning step by step."
)   # rag/agent.py:74-78
...
system_prompt = BASE_SYSTEM_PROMPT   # rag/agent.py:642
```

- 结论：**没有每个节点一套独立 system prompt / 独立记忆**。所有节点 agent 共享同一句通用 system prompt + 同一套工具（7 个图工具 + 可选 `kg_validate` + 可选 `search_documents` + 外部插件），节点间的差异**完全靠传入的 `user_msg` 文本**区分（例如 extract 让它「只返回 `{"entities": [...]}`」，critique 让它「只返回 `{"hypotheses": [{"statement","score"}]}`」）。
- **无状态**：每次 `build_node_agent()` 都是 `FunctionAgent(...)` 全新实例，不传 `session_id`、不共享 chat_history（对比 `rag/agent.py` 里 `reason_llamaindex` 支持 session 恢复，但 loop 节点**没有**用这套 session 恢复）。所以节点 agent 之间、以及跨 cycle 之间，agent 层面没有共享记忆。

---

## 2. 工作区文件（谁写、存什么）

工作区布局见 `director.py:87-97` 的 docstring。**所有文件都由 director 写**，agent 从不碰这些文件（`_log_experiment` docstring 明说「written ONLY by the director (agents never touch it)」，`director.py:275-282`）。

| 文件 | 存什么 | 写者 | 证据 |
|------|--------|------|------|
| `task.md` | 研究任务（bootstrap，只写一次） | director `_save_state` | `director.py:180-183` |
| `champion.md` | 已确认结论（frontmatter `count`，正文 `- [cycle N] 结论`） | director `_save_state` | `director.py:185-190` |
| `dead_ends.md` | 已否定假设（frontmatter `count`，正文 `- 假设`） | director `_save_state` | `director.py:192-197` |
| `knowledge/patterns.md` | 三层：已验证结论 + 已否定假设 + 已耗尽方向（no-gain 轮次/转向次数） | director `_save_state` | `director.py:200-215` |
| `results/cycle-NNN.md` | 单轮证据：KEEP / DISCARD / 预测 / 假设 / 报告 | director `_save_cycle_result` | `director.py:227-240` |
| `run.json` | 纯运行时续跑状态：cycles / consecutive_no_gain / adaptations / 时间戳 | director `_save_state` | `director.py:217-225` |
| `logs/experiments.jsonl` | 每轮结局（KEEP/NO_GAIN）的**唯一规范日志**，停滞检查读它 | director `_log_experiment` | `director.py:273-295` |
| `logs/sessions.jsonl` | 每次 director 运行一条（session_id/cycles/champion/rejected/adaptations/status） | director `_log_session` | `director.py:297-315` |
| `traces/cycle-NNN.json` | 单轮完整 `ResearchState` 序列化（`rs.model_dump()`） | director `_save_cycle_trace` | `director.py:242-267` |
| `jobs/` | agent 自写后台作业落点（`DRBRAIN_RUN_DIR`，`run_python mode=async`） | 环境变量注入，director `run()` | `director.py:399-402` |

说明：
- `champion.md` / `dead_ends.md` / `knowledge/patterns.md` / `results/cycle-NNN.md` 是**语义对象文件**（镜像 AutoScientists），不是单个 JSON blob；`run.json` 是唯一存运行计数的文件。
- 从文件恢复内存状态的逻辑在 `_load_state`（`director.py:141-175`）：champion 通过正则 `^- \[cycle (\d+)\] (.*)$` 从 champion.md 解析，dead_ends 从 `- ` 列表解析。
- `results/` 在 `_load_state` 里被注释为「reconstructed lazily」（`director.py:168`），实际 `results: []` 是空列表重建，只供报告用途。

---

## 3. falsifiable hypotheses 怎么实现

### 3.1 字段存在

`Hypothesis` 数据类（`events.py:30-45`）确实带三个语义字段：

```python
class Hypothesis(BaseModel):
    statement: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    prediction: str = ""        # 可观察预测：什么证据会支持该假设
    falsification: str = ""     # 证伪标准：什么证据会放弃该假设
    score: float = 0.0
    status: str = "proposed"    # proposed → critiqued → confirmed | falsified
```

### 3.2 提出阶段

`identify_gaps`（`workflow.py:385-439`）要求 agent 按结构返回：

```python
'{"gaps": [...], "hypotheses": [{"statement": "...", '
 '"prediction": "什么证据会支持它", "falsification": "什么证据会证伪它", '
 '"conditions": {}}]}'
```

agent 输出被解析成 `Hypothesis(statement=..., prediction=..., falsification=...)`（`workflow.py:407-417`）。确定性兜底假设也带 prediction/falsification（`workflow.py:428-436`）。

### 3.3 怎么判证伪 —— **prompt 级指令，不是代码级判定**

关键事实：`falsification` 字段被写入 state、被打印进报告模板（`workflow.py:186-188`），但**没有任何代码拿「证据」去对 `h.falsification` 做匹配/判定**。实际证伪判定发生在 `verify` 节点，靠的是让 agent 自行分类：

```python
# verify 的 user_msg（workflow.py:486-491）
"核验以下假设：用你手上的检索工具找文献证据，用计算/模型工具或"
"自写代码做数值验证。对每个假设判定：verified（证据支持其 prediction）、"
"falsified（证据证伪）、或无法判定则不入两类。"
```

即「prediction / falsification」字段是**喂给 LLM 的提示语义**，证伪是 agent 的自由裁量，不是确定性比对。

---

## 4. 三分类 KEEP / DISCARD —— `director._absorb`

判定逻辑在 `director.py:340-380`（`_absorb`），docstring 自述 AutoScientists 三分类：

```python
verified = list(rs.verified) if rs else []
falsified = list(rs.falsified) if rs else []
...
champion_statements = {c["statement"] for c in state["champion"]}
rejected = set(state["rejected"])

new_champion = [v for v in verified if v not in champion_statements]
new_rejected = [f for f in falsified if f not in rejected and f not in champion_statements]
```

- **verified → champion（KEEP）**：`state["champion"].append({"statement": v, "cycle": cycle_no, "confidence": 1.0})`（`director.py:360-361`）
- **falsified → dead_ends（DISCARD）**：`state["rejected"].extend(new_rejected)`（`director.py:362`）
- **neither（unresolved）→ 不入两类**：既不是 champion 也不是 dead end，只是「证据不充分」，下一轮可借 prior_context 再试（docstring `director.py:343-347`）。

外加一条与三分类正交的停滞标记（见 §5）。

注意：`_absorb` 只看 `rs.verified` / `rs.falsified` 两个字符串列表，**不读** `h.score`（critique 分数）也不读 `h.status` 细分——分类依据是 verify 节点产出的三列表，而非 hypothesis 对象状态。

---

## 5. Phase 4 adapt（stagnation → pivot）

触发与动作在 `director.py:382-449`（`run` 循环末尾）。

- 触发条件：`consecutive_no_gain >= stagnation_cycles`（默认 `stagnation_cycles=3`）。
- `no_gain` 计数在 `_absorb` 里更新：`state["consecutive_no_gain"] = 0 if new_champion else state["consecutive_no_gain"] + 1`（`director.py:377`）。即「本轮有新增 champion 才算有进展」。

动作（`director.py:434-448`）：

```python
if state["consecutive_no_gain"] >= stagnation_cycles:
    state["adaptations"] = state.get("adaptations", 0) + 1
    state["rejected"].append(
        f"[stagnation] 连续 {stagnation_cycles} 轮无新结论，当前方向已耗尽"
        f"（第 {state['adaptations']} 次转向）"
    )
    ...
    state["consecutive_no_gain"] = 0     # 计数器重置 → 继续跑
    if state["adaptations"] >= max_adaptations:
        stop_status = "max_adaptations"
        break
```

**如实描述**：这个「pivot」**并没有真正改变研究方向**——它只是：
1. 把一条 `[stagnation] …` 文本塞进 `rejected`（dead_ends）；
2. 把 `consecutive_no_gain` 重置为 0；
3. 循环继续跑（直到 `adaptations >= max_adaptations`，默认 `max_adaptations=2` 才停）。

没有任何代码去改变 topic、检索 query、假设生成方向。默认参数：`max_cycles=10, stagnation_cycles=3, max_adaptations=2`（`director.py:385-388`）。

---

## 6. 验证机制：critique + verify 是「独立 agent 互验」还是「顺序节点」？

**是后者——单 workflow 里的顺序节点，不是互相验证的独立 agent。**

- `critique`（`workflow.py:441-474`）给每个假设打分（0~1），把 `status` 改为 `critiqued`，把分数写进 `state.scores`。
- `verify`（`workflow.py:476-510`）在 critique **之后**顺序执行，独立地再做一次 verified/falsified/predictions 分类。

关键证据：**verify 不使用 critique 的分数**。

- `verify` 开头 `verified = [h.statement for h in ev.hypotheses if h.status == "critiqued"]`（`workflow.py:478`）只按 `status == "critiqued"` 过滤（critique 已经把所有 hypothesis 都置为 critiqued），随后整段被 agent 返回的 JSON 覆盖。`h.score` 从未参与筛选。
- `critique` 的分数只写进 `state.scores`（`workflow.py:470`），**下游没有任何节点读 `state.scores` 来做门槛/排序/裁决**。

所以：
- 两者不是「两个独立 agent 各自独立判断、再比对结果」——它们是同一流水线里先后两个节点；
- 也**不是「后一个校验前一个输出」**——verify 不校验 critique 的分数，而是从头独立再跑一遍（各自都构建全新 agent、各自跑检索/计算）；
- critique 与 verify 的 agent 都是同一个 `build_node_agent()` 模板（同一 system prompt + 同套工具），不存在「评审者」与「被评审者」的角色分工。

---

## 7. 日志（写什么、谁写、什么格式）

三个 JSONL 日志 + traces 目录：

| 日志 | 路径 | 写者 | 格式 / 字段 | 证据 |
|------|------|------|-------------|------|
| `experiments.jsonl` | `run_dir/<topic>/logs/` | director `_log_experiment` | 每轮一条：`cycle, topic, outcome(KEEP/NO_GAIN), champion_before/after, verified, falsified, predictions, hypotheses, started_at, completed_at, duration_seconds` | `director.py:273-295` |
| `sessions.jsonl` | 同上 | director `_log_session` | 每次运行一条：`session_id, topic, started_at, ended_at, duration_seconds, cycles_run, champion_count, rejected_count, adaptations, status` | `director.py:297-315` |
| `llm_calls.jsonl` | **`data/logs/llm_calls.jsonl`（不在 run_dir 工作区内）** | `extractor/llm_client.py` 的 `_log_llm_call` | 每次 LLM 调用一条：`ts, session_id, model, provider, status, prompt_hash, n_messages, tokens_in, tokens_out, duration_ms, error`。**不记录 api_key、不记录 prompt/messages 原文**（只有 hash + 计数） | `extractor/llm_client.py:267-289` |
| `traces/cycle-NNN.json` | `run_dir/<topic>/traces/` | director `_save_cycle_trace` | 单轮完整 `ResearchState`（`rs.model_dump()`，`default=str`） | `director.py:242-267` |

补充要点：
- `llm_calls.jsonl` 的写点是 **LLM 调用层**（`extractor/llm_client.py`），所以 loop 节点 agent 走的每条 `AgentFunctionLLM.achat → acall_with_messages → call_with_fallback` 链路都会**间接**落到这里；它不是 director 写的，也不分 topic，全局一个文件。
- 另外存在 SQLite 表 `llm_calls`（`src/drbrain/metrics.py:17`），与本任务关注的 JSONL 无关，属 metrics 面板的另一套持久化。

---

## 8. agent 之间到底有没有「互相验证 / 互评」？

**没有。** 现状是**固定顺序的单向流水线**：

```
plan → retrieve → filter → parse → extract → normalize → fuse
     → identify_gaps → critique → verify → settle → report
```

- 每个节点只接收上一个节点的类型化事件（`TaskPlanned → Retrieved → Filtered → …`），通过共享 `Context.store` 里的 `ResearchState` 读写，**单向推进、不回退**（唯一的循环是 `filter → RetrieveAgain → retrieve`，且 `MAX_RETRIEVE_ATTEMPTS=3` 有界，`workflow.py:326-334`）。
- 「critique」的中文注释写的是「假设互评」（`workflow.py:441`），但它**不是两个 agent 互评**，而是**单个 agent 给一组假设逐个打分**（`workflow.py:450-460` 让 agent 输出 `{"hypotheses": [{"statement","score"}]}`）。
- 不存在「A agent 审 B agent 的结论、B 再反驳」之类的多 agent 对话/对抗结构；`critique` 和 `verify` 各自独立跑一个全新 agent，结果只通过 state 顺流，无交叉质询。
- 跨 cycle 的「记忆」只有两个来源：
  1. `ResearchState` 结构化状态（单轮内共享，跨轮不持久）；
  2. director 把 champion/dead_ends 拼成 `prior_context` 文本注入下一轮 `StartEvent`（`_build_prior_context`，`director.py:317-325`；`identify_gaps` 读取 `state.prior_context` 注入 prompt，`workflow.py:393-399`）。

---

## 附：与「AutoScientists 语义」的对应情况（如实，不夸大）

| AutoScientists 概念 | 现状落地 | 证据 |
|---------------------|----------|------|
| 多轮连续循环 | ✅ director 外层 while 循环跑 cycle | `director.py:406-449` |
| 文件式任务工作区（champion/dead_ends/patterns/results） | ✅ 有，全部 director 单写 | `director.py:87-97` |
| 可证伪假设（prediction/falsification 字段） | ⚠️ 字段有，但证伪靠 LLM 自由裁量，无代码级比对 | `events.py:30-45`、`workflow.py:476-510` |
| critique-before-compute | ⚠️ 有 critique 节点打分，但分数**不被 verify 消费**，起不到「筛选」作用 | `workflow.py:441-474` |
| 共享成败（shared success/failure） | ✅ champion/dead_ends + prior_context 注入下一轮 | `director.py:317-325` |
| Phase 4 adapt / pivot | ⚠️ 只记一条 stagnation 死路 + 重置计数器，**不真正换方向** | `director.py:434-448` |
| 多 agent 协作 / 互评 | ❌ 无——单流水线、每节点独立无状态 agent、单向推进 | §8 |
