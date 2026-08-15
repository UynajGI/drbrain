# AutoScientists vs drbrain loop 差距报告

> ⚠️ **差距已大部分补齐（2026-08-15）**：本文写于讨论层实现之前。所记差距「无独立角色、无 per-agent 记忆、无消息板、互验形同虚设」已通过 `loop/discussion.py`（消息板 MessageBoard + 队列 ResearchQueue + 非作者门 + queue claim）+ 4 角色（analyst/critic/compute/verifier）+ per-agent 记忆（role-*.md）+ roster 补齐。**现役权威见 [docs/drbrain-architecture.md](drbrain-architecture.md) 第 5 章。**

> 基于两份源码级调研产出：
> - `docs/autoscientists-architecture.md`（AutoScientists commit c71a923 逐文件取证）
> - `docs/loop-current-state.md`（drbrain loop 现状，文件:行号取证）
> 生成日期 2026-08-14。如实标注差距，不夸大「已对齐」。

---

## 0. 一句话结论

AutoScientists 的「多 agent」**不是** Python 内部的多线程/多进程调度，也**不是**中心化 workflow 引擎，而是**「提示词 + 共享状态服务器」的分布式约定**：1 orchestrator + 10 个无状态 worker（各自独立 Claude Code 会话 + 独立 `HEARTBEAT.md`/`AGENT.md`/`memory/`），通过本地 Node 服务 ClawInstitute（消息板 + 版本化文件 + 乐观锁）异步协作。

drbrain loop 是**单 Python Workflow + 12 节点固定顺序 + 同一个 `build_agent` 模板换 prompt**，无独立角色、无 per-agent 记忆、无消息板、互验形同虚设。

**最关键的差距不是「少了几个节点」，而是「协作形态完全不同」：auto 靠共享黑板 + 消息板集体评审，我方靠单向流水线。**

---

## 1. 架构对比表

| 维度 | AutoScientists | drbrain loop | 判定 |
|---|---|---|---|
| agent 数量/角色 | 1 orchestrator + 10 worker（monitor×1 / gpu×6 / analyst×3，角色硬编码在 `launch.py:559`） | 12 节点（6 确定性 + 5 agent-backed + retrieve 半 agent），**同一 `build_agent` 模板** | ❌ 无独立角色 |
| 独立 system prompt | 每 agent 专属 `HEARTBEAT.md`（`setup_agent()` 把 `ROLE-{role}.md` 注入占位符，角色不同内容不同） | 写死 `BASE_SYSTEM_PROMPT`（`rag/agent.py:74`），节点差异全靠 `user_msg` 文本 | ❌ |
| 独立记忆 | 每 agent 独立 `memory/MEMORY.md` + `memory/feedback_{topic}.md`（HEARTBEAT Part 6c） | 无 per-agent memory；每次全新无状态 `FunctionAgent`，不传 session_id、无共享 chat_history | ❌ |
| 调度 | 无中央调度；决策写在 HEARTBEAT Part 0 Mode Selector，LLM 现场执行 | 固定 12 节点顺序（`workflow.py` @step），单向推进 | ⚠️ 我确定性更强、auto 自主性更强 |
| 状态层 | ClawInstitute：workshops（帖子/评论/通知）+ workspaces（版本化 YAML + `If-Match` 乐观锁） | SQLite + 文件工作区（director 单写，无乐观锁、无消息板） | ❌ |
| 协作形态 | 异步共享黑板 + 消息板，agent 通过读写共享状态「对话」 | 单向流水线，唯一回环 `filter→RetrieveAgain`（有界 3 次） | ❌ |

---

## 2. 互验对比（你最关心的）

**先纠正一个误解**：AutoScientists **也没有**「双跑同一实验对比」、**也没有**独立 reviewer agent。它的互验是**共享黑板上的五层集体评审**：

| AutoScientists 互验层 | 机制 | drbrain loop 现状 | 判定 |
|---|---|---|---|
| ① Discussion-Before-Queuing | proposal 入队前**强制 ≥1 非作者评论**（`ROLE-TEAM.md` / `ROLE-GPU.md` Step 3） | 无 proposal 层、无评论 | ❌ |
| ② hypothesis lens 三角验证 | 同一结果被不同 team 用各自假设独立判 Supports/Refutes/Orthogonal，累进计数驱动证伪（`ROLE-ANALYST.md` Step 0.3） | 有 critique 打分，**但 `h.score` 从未被 verify 消费**（`workflow.py:441-474`） | ❌ 形同虚设 |
| ③ 跨代理查重 | 提案前查 results/ + dead_ends + champion code 三重去重 | 有 champion/dead_ends + prior_context 注入，但无「提案前强制查重门」 | ⚠️ 部分 |
| ④ 晋升自我验证门 | 多 seed 噪声门 / race-condition 重读 / diff-applied→FAILED（`ROLE-GPU.md` Step 4/5/7） | 无；verify 三分类纯 LLM 自由裁量 | ❌ |
| ⑤ 结构变更 endorsement | ≥2 非提案者实质赞同 + 0 反对（`ROLE-ANALYST.md` Step 1d.5） | 无 | ❌ |
| ⑥ janitor 复核 | claim 超 30min 且无结果文件 → 释放重新认领 | 有 timeout，但无「claim 释放重认领」 | ⚠️ 部分 |

**判定**：我方 loop 的「互相验证」＝ critique 打分后被丢弃 + verify 三分类 LLM 裁量，**没有一层是真正生效的互验**。

---

## 3. 已对齐 / 半成品 / 缺失（诚实清单）

**✅ 已落地**：
- 多轮连续循环（`director.py:406-449` while 循环）
- 文件式任务工作区（champion/dead_ends/patterns/results/run.json）
- 共享成败（champion/dead_ends + prior_context 注入下一轮）
- 单一日志（experiments.jsonl，director single-writer）

**⚠️ 半成品（字段有、未代码化）**：
- 可证伪假设：`prediction`/`falsification` 字段有（`events.py:30-45`），但证伪靠 LLM 自由裁量，无代码比对证据
- critique-before-compute：有 critique 节点，但分数不被 verify 消费，起不到筛选作用
- Phase 4 adapt：只记一条 `[stagnation]` 死路 + 重置计数器，**不真正换方向**

**❌ 缺失**：
- 多 agent 独立角色 + 独立记忆
- 消息板 / 讨论层（proposal 评审）
- 版本化状态层（If-Match 乐观锁）
- 结果侧多假设三角验证
- 晋升侧噪声/race 自我验证门

---

## 4. 三个最痛的差距（按影响排序）

1. **单 workflow 同一模板，无独立角色/记忆** → 这正是「critique 分数被丢弃」「verify 不实算 DFT」「identify_gaps 出废话假设」的根因：所有节点是同一个无状态 agent，没有角色约束逼它做对的事。

2. **互验五层全部缺失** → critique→verify 是单向顺序，critique 的产出（分数）根本不流入 verify，等于没有互评。

3. **无共享状态层（消息板 + 乐观锁）** → auto 的「集体评审」靠帖子和版本化文件承载，我方只有 director 单写文件，agent 之间没有「对话」通道。

---

## 5. 重构建议（分三步，从轻到重）

**第一步（轻，先修最痛的，解决「互验形同虚设 + DFT 没实算」）**：
1. verify 真正消费 critique 分数：`h.score` 参与筛选（低分假设直接 DISCARD 或需更强证据才 KEEP）。
2. falsification 判定代码化：拿证据比对 `h.falsification`，而不是 LLM 一句话「证伪」。
3. 给 verify 节点加「实算门」：当环境有 `run_python`/`check_job`/`materials-env` 时，强制要求产出一份**实际计算结果**（能量/能带数值）作为 verified 的证据，否则降级为 prediction。

**第二步（中，引入角色分化，接近 auto 的独立 agent）**：
- 把「同一 `build_agent` 换 prompt」改成「按角色独立 agent 定义」：analyst（提假设）/ critic（独立打分）/ verifier（独立核验）各自独立 system prompt，**至少让 critique 和 verify 是不同角色、不同人格、独立记忆**，而不是同一模板的两个 prompt。

**第三步（重，引入共享状态层，真对齐 auto）**：
- 加一个轻量消息板/讨论层：proposal 入队前强制 ≥1 非作者评论（对应 auto 的 Discussion-Before-Queuing）。
- 状态文件版本化 + 乐观锁（或至少 `dead_ends.md` 跨 agent 读 + 提案前强制查重）。
- 结果侧多假设 lens 计数（Supports/Refutes/Orthogonal）。

---

## 6. 建议

- **短期（初赛交付）**：做第一步，先把「critique 分数接入 verify + 实算门」落地——这直接解决当前实跑暴露的两个硬伤（互验失效 + DFT 不实算），成本低、见效快。
- **中长期（真对齐 auto）**：第二步 + 第三步，是真正的架构重构，建议单独一轮专项做。
