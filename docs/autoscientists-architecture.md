# AutoScientists 真实架构总结

> 调研对象：`mims-harvard/AutoScientists`（GitHub）
> 源码取证版本：commit `c71a92343b9a488ed10134be805845b9473ad18f`（2026-05-28，"Initial public release"，即公开发布的首个版本）
> 调研方式：下载 `refs/heads/main` tarball 到 `/tmp/AutoScientists-main/` 逐文件读源码（未读 README 下结论）
> 配套论文：arXiv 2605.28655《AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation》

本文目的：准确回答「AutoScientists 到底怎么实现多 agent」，并逐项落到「具体机制 + 源码文件路径 + 关键片段」。**第 9 节（agent 互验）是重点**，也是与我方 loop（单 workflow 编排）对比的关键差异点。

---

## 0. 一句话结论（先给框架，避免被细节淹没）

AutoScientists 的「多 agent」**不是**一个 Python 程序内部的多线程/多进程调度，也**不是**一个中心化的 workflow 引擎。它是一套**「提示词 + 共享状态服务器」的分布式约定**：

- **1 个 orchestrator**：一个 Claude Code 会话，读 `runbook.md`，负责外层循环（launch/轮转/健康检查/champion 晋升）。
- **10 个 worker agent**：每个是一个独立的 Claude Code subagent 会话，读各自生成的 `agents/<name>/HEARTBEAT.md`，自己决定「这轮我该干什么」（Mode Selector）。
- **共享状态层**：本地跑一个 Node 服务 `ClawInstitute`（`npx clawinstitute start`），提供两样东西——
  - **workshops** = 消息板（posts/comments/notifications，用于「讨论」）
  - **workspaces** = 带版本号的文件（YAML frontmatter，用于「结构化状态」）

协调既不是纯消息传递、也不是纯共享文件、更不是心跳轮询调度，而是**三者混合**：每个 agent 每轮启动时读共享文件/帖子，自己按 HEARTBEAT Part 0 选分支，做完后用 `If-Match` 乐观并发写回。**没有任何一段 Python 代码在「调度」agent 的决策逻辑**——决策全在提示词里，由 LLM 自己执行。

关键证据（`launch.py` 末尾 + `README.md`）：

```text
README.md:
  Each launch materializes ... claude -p "Read runbook.md and execute. Task: ..."

launch.py (main 末尾):
  To run the orchestrator:
    claude -p "Read {ROOT / program_file} and execute"
```

---

## 1. 多 agent 架构：几个 agent、各自 role、怎么协作

### 1.1 Agent 数量与角色（源码硬编码）

`launch.py` 第 559 行 `AGENTS` 字典，共 **10 个 agent**：

```python
AGENTS = {
    f"{PREFIX}_monitor":  ("Focus area monitor — bootstraps, forms teams, monitors health", "monitor",  "server1", -1),
    f"{PREFIX}_gpu1":     ("GPU agent 1 — runs experiments on GPU 0", "gpu", "server1", 0),
    f"{PREFIX}_gpu2":     ("GPU agent 2 — runs experiments on GPU 1", "gpu", "server1", 1),
    f"{PREFIX}_gpu3":     ("GPU agent 3 — runs experiments on GPU 0", "gpu", "server2", 0),
    f"{PREFIX}_gpu4":     ("GPU agent 4 — runs experiments on GPU 1", "gpu", "server2", 1),
    f"{PREFIX}_gpu5":     ("GPU agent 5 — runs experiments on GPU 0", "gpu", "server3", 0),
    f"{PREFIX}_gpu6":     ("GPU agent 6 — runs experiments on GPU 1", "gpu", "server3", 1),
    f"{PREFIX}_analyst1": ("Analyst 1 — researches mechanisms, proposes experiments", "analyst", "server1", -1),
    f"{PREFIX}_analyst2": ("Analyst 2 — researches mechanisms, proposes experiments", "analyst", "server2", -1),
    f"{PREFIX}_analyst3": ("Analyst 3 — researches mechanisms, proposes experiments", "analyst", "server3", -1),
}
```

- **monitor × 1**：系统「清洁工」（janitor），只做健康检查，**不做**组队、不跑实验。角色协议在 `system/templates/ROLE-MONITOR.md`。
- **gpu × 6**：跑实验（训练），从 team 队列认领、训练、记录结果、判断 KEEP/DISCARD/FAILED、晋升 champion。角色协议在 `system/templates/ROLE-GPU.md`。
- **analyst × 3**：研究机制、提 proposal、剪枝 dead ends、维护知识。**永不跑训练**。角色协议在 `system/templates/ROLE-ANALYST.md`。

### 1.2 每个 agent = 独立 system prompt + 独立记忆吗？——是。

`launch.py` 的 `setup_agent()`（第 575 行）给每个 agent 建一个独立目录：

```python
def setup_agent(name, desc, role, server, gpu):
    agent_dir = AGENTS_DIR / name
    (agent_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
    (agent_dir / "memory").mkdir(parents=True, exist_ok=True)
    # credentials.json（独立 api_key）
    # AGENT.md（身份文件，相当于 CLAUDE.md）
    # memory/MEMORY.md（独立记忆索引，空）
    # HEARTBEAT.md（该 agent 专属、自包含的完整指令，见下）
```

`system/reference/AGENT-SETUP.md` 明确类比了它与 Claude Code 的对应关系：

| Claude Code | AutoScientists agent |
|---|---|
| `CLAUDE.md` | `AGENT.md` |
| `~/.claude/.../memory/MEMORY.md` | `memory/MEMORY.md` |
| `~/.claude/.../memory/*.md` | `memory/*.md` |

所以：**每个 agent 有独立的 `AGENT.md`（身份/续接）+ 独立的 `memory/` 目录（持久记忆）**。`HEARTBEAT.md` Part 6c 是记忆写入协议（写 `memory/feedback_{topic}.md` 并更新 `MEMORY.md` 索引）。

`HEARTBEAT.md` 是**每个 agent 各自专属**的：`setup_agent()` 读模板 `system/templates/HEARTBEAT.md`，把 `ROLE-{role}.md`（gpu/analyst/monitor）和 `ROLE-TEAM.md` 的内容注入两个占位符，生成该 agent 自己的 `HEARTBEAT.md`：

```python
role_file_map = {"gpu": "ROLE-GPU.md", "analyst": "ROLE-ANALYST.md", "monitor": "ROLE-MONITOR.md"}
...
heartbeat = heartbeat.replace("<!-- ROLE_CONTENT_PLACEHOLDER -->...", role_content)
heartbeat = heartbeat.replace("<!-- TEAM_CONTENT_PLACEHOLDER -->...", team_content)
(agent_dir / "HEARTBEAT.md").write_text(heartbeat)
```

**所以「独立 system prompt」是实打实的**：每个 agent 的 `HEARTBEAT.md` 因 role 不同而内容不同。

### 1.3 Agent 之间怎么协作

三样东西（见 `system/reference/API-REFERENCE.md` 的完整端点清单）：

1. **workspace 文件**（共享状态，版本化）——`PUT /workspaces/{id}/files/{path}`，`If-Match` 乐观锁，`PATCH` frontmatter，`GET /files` 列举（LIST→DECIDE→READ）。
2. **workshop posts/comments**（讨论层）——`POST /posts`，带 `notify_agents` 触发收件箱通知。
3. **本地文件系统**——`{FOCUS_ROOT}/champion/train.py`、`logs/*.jsonl`、`agents/<name>/workspace/*` 等（agent 与 orchestrator 各自读写的本地路径）。

协作的**具体形态**是「异步共享黑板 + 消息板」，每个 agent 是**无状态 worker**（`AGENT-SETUP.md`：「agents are stateless workers that read shared state, act, and write back. No session depends on a previous session's memory」）。agent 之间不直接互相调用；它们通过读/写同一个 workspace + 帖子来「对话」。

---

## 2. Heartbeat loop / Mode Selector

### 2.1 谁决定「下一步派哪个 agent」

**外层调度 = orchestrator（`runbook.md` Step 5 的 while 循环）**，但**每个 agent 走哪个分支 = agent 自己（`HEARTBEAT.md` Part 0）**。两层：

- `runbook.md` Step 5b/5c：orchestrator 每轮先**并行** launch 3 个 analyst，再按 profile 的 `gpu_dispatch` launch GPU agents，然后 wait、promotion、health、stagnation、periodic hooks。
- `HEARTBEAT.md` Part 0 Mode Selector：agent 被 launch 后，**先不做任何实事**，而是依次过 4 个检查，选一个分支。

### 2.2 Mode Selector 的四个检查（`HEARTBEAT.md` Part 0）

```
Check A   launch prompt 是否设了 MODE=discussion / MODE=execute
Check A2  是否有未解决的 [DISCUSSION-TRIGGER] 帖子（<5 个 [DISCUSS-DONE]）→ 切 discussion
Check B   teams/roster.md 是否有 team：
           - roster 空 → Part 2（cold-start bootstrap）
           - 有 team 但我不在任何 team → Part 3（干净退出）
           - 我在某个 team → 继续 Check C
Check C   （仅 GPU）是否有未 post 的上轮结果（result_latest.json sentinel）→ Part 5 resume-and-post
Check D   → Part 4 正常循环
```

最终分支汇总表（HEARTBEAT.md 原文）：

```text
| Launch MODE | Roster | MY_TEAM | Pending result? | Branch | What you do |
| any         | any    | any     | (GPU) unposted, alive | resume-waiting | Log, exit |
| any         | any    | any     | (GPU) unposted, done  | Part 5 | Post [RESULT], update champion |
| discussion  | any    | any     | none   | Part 2 | CPU-only thinking |
| execute     | empty  | —       | none   | Part 2 | Cold-start bootstrap |
| execute     | non-empty | None | none   | Part 3 | Exit cleanly (coordination bug) |
| execute     | non-empty | set  | none   | Part 4 | Normal cycle |
```

Part 0 明确「Do NOT skip Part 0. Do NOT execute Parts 1–5 until Part 0 has explicitly told you which branch」。**「下一步派谁、走哪个分支」不是 Python 里的 if/else，而是写进提示词、由 LLM 现场执行的规则。**

### 2.3 停止条件

来自 profile 的 `exit_condition` hook（`runbook.md` 引用的 task-profile）：

- **optimization（task-autoresearch/LAUNCH.md）**：`exit_condition()` 恒返回 `False`（永不自愿退出）；真正退出靠 `stagnation_response` 在「最近 10 个实验 0 KEEP」时 `raise SystemExit(0)`。

```python
# task-autoresearch/LAUNCH.md
def exit_condition():
    return False   # never exit voluntarily; only stagnation_response stops the loop

def stagnation_response(cycle_count):
    print(f"STAGNATION: 0 KEEPs in last 10 experiments (cycle {cycle_count})")
    print("Stopping loop — wait for user input.")
    raise SystemExit(0)
```

- **biomlbench（task-biomlbench/LAUNCH.md）**：固定 deadline（8h CPU / 16h GPU），`pre_cycle_check()` 在剩余时间 ≤ `DEADLINE_BUFFER_MINUTES`（15min）且 `submission.csv` 存在时返回 True → 退出；若无 submission 则先触发 emergency save 再退出。

---

## 3. 工作区文件：谁写谁读（single-writer 还是多写）

「工作区」分**主 workspace**（全团队共享）和**team workspace**（每队一个）。核心文件与读写者如下（据 `launch.py` Step 5 实际 seed 的 + 各 ROLE 文档）：

| 文件 | 所在 workspace | 谁写 | 谁读 | 并发控制 |
|---|---|---|---|---|
| `champion.md` | main | launch.py seed；GPU agent 在 KEEP 时（ROLE-GPU Step 7b） | 所有 agent 每轮必读 | `If-Match` version |
| `champion/train.py` + `champion/SOURCE` | 本地文件系统 | KEEP 的 GPU agent（Step 7b1 原子 `tmp.replace`）| 所有 GPU agent（Step 2） | 原子 rename + If-Match 串行化 |
| `queue.md` | team | analyst 加项（Step 5）；GPU agent claim/release（Step 3/6） | 全 team | **多写**，`If-Match` read-modify-PUT，禁止 PATCH |
| `strategy.md`（= 假设文件） | team | monitor/组队者创建；analyst 更新计数 | 全 team | 多写 |
| `dead_ends.md` | team | analyst 剪枝（Step 2）；GPU agent 记 DISCARD（Step 7c） | analyst/GPU dedup 检查 | 多写，`If-Match` |
| `knowledge/patterns.md`、`knowledge/exhausted.md` | main | launch.py seed；agent 后续可写 | 所有 agent | — |
| `results/{exp_id}.md` | main | GPU agent（Step 5）**write-once** | 所有 agent/team | 单写（每个 exp_id 只写一次） |
| `teams/roster.md` | main | **字母序最后一个 analyst**（Step 0.25/1d.5 组队重组） | 所有 agent 每轮必读 | 单写者仲裁（字母序规则）+ `If-Match` |

**结论**：除了 `champion.md`、`results/{exp_id}.md`、`teams/roster.md`、`champion/train.py` 是**受控单写者**（靠「唯一有权写」的约定 + 字母序仲裁 + If-Match 兜底），`queue.md`、`dead_ends.md`、`strategy.md` 都是**多 agent 可写**，靠 `If-Match`（version 校验，冲突返回 409）做乐观并发，而不是中央锁。

关于你问的「hypotheses」文件名：文档里有两套说法，如实标注——
- `PHASES.md` 的 `create_team()` 代码里创建的是 `queue.md / hypotheses.md / dead_ends.md / strategy.md`（其中 `hypotheses.md` 只有 `count: 0` 一个字段）；
- 但 `ROLE-MONITOR.md`、`ROLE-ANALYST.md` Step 0.3、`SKILL.md` 全都把「带 hypothesis/prediction/falsification 字段的文件」叫 **`strategy.md`**。

即：**假设字段实际存在 `strategy.md` 的 frontmatter 里**，`hypotheses.md` 只是 `PHASES.md`（较老参考文档）里的一个几乎未使用的空壳。这也是「参考文档与 launch.py/ROLE 文档不完全同步」的一个实例（见 §10 文档漂移）。

---

## 4. Falsifiable hypotheses（可证伪假设）

团队**按假设组队，不按坐标轴组队**（`ROLE-MONITOR.md`「Team Creation — Hypothesis-Based, Not Axis-Based」）。每个 team 的 `strategy.md` frontmatter 必须含（`ROLE-MONITOR.md` 原文模板）：

```yaml
hypothesis: H-throughput
prediction: "Experiments that increase num_steps by ≥10% will KEEP"
falsification: "If 3 rotations of prediction-consistent experiments all DISCARD, hypothesis is falsified"
age_rotations: 0
supported_keeps: 0
refuted_discards: 0
```

- `hypothesis`：对「当前是什么在限制 metric」的可证伪断言（如「模型在当前算力预算下训练不足」）。
- `prediction`：一个**可观测的实验模式**（如「任何让 num_steps 增加 ≥10% 的实验会 KEEP」）。
- `falsification`：一个**明确的放弃门槛**（如「3 轮与预测一致的实验全部 DISCARD」）。

### 判定证伪（`ROLE-ANALYST.md` Step 0.3 — Hypothesis Check）

每个 analyst 每轮开头必做：

```python
# 对每个新 team 结果分类：
#   Supports  → supported_keeps += 1   （KEEP 且与 prediction 一致）
#   Refutes   → refuted_discards += 1  （prediction 说该 KEEP，结果却 DISCARD）
#   Orthogonal → 不变                  （在 hypothesis 不预测的轴上）
# age_rotations += 1
# if age_rotations >= 3 and supported_keeps == 0 and refuted_discards >= 3:
#     post "[HYPOTHESIS-FALSIFIED]"
```

`ROLE-MONITOR.md` 的 health check 用同一个判据（每轮 monitor 检查 `age_rotations ≥ 3 AND supported_keeps == 0 AND refuted_discards ≥ 3` → 发 `[HYPOTHESIS-FALSIFIED]`，下一轮重组团队）。

关键点：**「证伪」是按 team 各自独立维护的计数器来判的**，且同一个实验结果会被不同 team 用各自的 lens 分别计为 Supports/Refutes/Orthogonal——这是「集体推理」的核心机制（见 §9）。

---

## 5. 三分类 KEEP / DISCARD / FAILED 的判定逻辑

**判定者：GPU agent 自己**（`ROLE-GPU.md` Step 5），不是 monitor、不是 analyst、不是 orchestrator。判据（`ROLE-GPU.md` Step 5 原代码）：

```python
diff_applied = bool(item.get("diff_applied", True))

if not diff_applied:
    outcome = "FAILED"
elif (direction == "minimize" and our_metric < current_best) or \
     (direction == "maximize" and our_metric > current_best):
    outcome = "KEEP"
else:
    outcome = "DISCARD"
```

三个要点：

1. **FAILED 的语义很特定**：不是「训练崩了」，而是「**proposal 的 diff 根本没应用到代码上**」（Step 4 用 `filecmp.cmp` 对比 `train.py` 与 champion，字节相同 → `diff_applied=False`）。此时测出来的 metric 只是 baseline 噪声，既不能当作 KEEP（会造出「幻影 champion」），也不能当作 DISCARD（会错误地证伪该 proposal），所以单列 FAILED，orchestrator 跳过晋升、analyst 可以换新 diff 重新入队。`ROLE-GPU.md` Step 4 明确写了历史事故：「Phantom KEEPs from this exact path (gpt-nano-pubrun round on 2026-05-26: data_v7, 0.979985, diff rejected) corrupted the champion lineage」。

2. **KEEP 之前要过 race-condition 重读**：Step 5 开头**重新读 champion.md**，对比训练前读到的 version，若变了 `race_condition=True`，且比较用**当前** champion 而非训练前读的那个。

3. **KEEP 还要过多 seed 噪声门**（`ROLE-GPU.md` Step 7.0）：`|delta| <= 2σ` 时换种子重跑，只有第二种子也更好才晋升，否则记 `[NEAR-MISS]`、不动 champion。

---

## 6. Phase 4 adapt（停滞转向）

**触发条件**（两处，注意新旧文档差异）：

- 老文档 `PHASES.md` Phase 4：monitor 检测「某队 10+ 连续 DISCARD」→ monitor 发 `[DISCUSSION]`（选项 A 合并 / B 拆分 / C 转向新轴 / D 解散）。
- **现行机制（权威）**：`ROLE-MONITOR.md` 明确「Mid-run regroup... handled by **agent-driven self-regroup**... Monitor does NOT intervene」，且 `ROLE-ANALYST.md` Step 0.2 把停滞检测**内建到每个 analyst 每轮**：

```python
trigger_conditions = (rotations_since_keep >= 3) or falsified_since_reform
# 若满足且没有 active trigger → analyst 直接 POST [DISCUSSION-TRIGGER]
```

另有 Step 0.2b「axis-mining」触发器：最近 8+ 个 DISCARD 集中在 ≤3 个轴且无跨轴配对探针 → 发 `[DISCUSSION-TRIGGER] (axis-mining)`。

**具体动作**（`ROLE-ANALYST.md` Step 0.25 / 1d / 1d.5）：

- 讨论收敛（≥5 个 `[DISCUSS-DONE]`）后，**字母序最后一个跑过的 analyst** 重写 `teams/roster.md`（重新按 3 个假设组队），发 `[TEAM-REFORMED]`。
- 结构性调整走 `[DIMENSION-NEW]` / `[DIMENSION-SPLIT]` / `[DIMENSION-MERGE]` / `[REGROUP]` 帖子，需「endorsement bar」（≥2 个非提案者的实质赞同/反对解决评论 + 0 未解决反对 + 帖子 ≥1 rotation），满足后由**字母序最后一个非提案者 analyst** 执行合并（改 roster + 发 `[TEAM-REFORMED]` + 给被解散队的 queue.md 打 `team_status: dissolved` 标记）。
- 重组时强制「cold-axis 指令」：每个新队队列里至少要有一个零历史实验的冷轴，否则改发 `[SYSTEM-EXHAUSTED]`。

---

## 7. Monitor（health check）

两个并行的健康检查通道（注意不是只有 monitor）：

**通道 A — monitor agent（`ROLE-MONITOR.md`）**，Phase 3 期间每 10 分钟一次：

```python
def health_check(main_ws_id, roster):
    for team_name, team in roster["teams"].items():
        # 1. 数连续 DISCARD（停滞阈值：单队 10 连 DISCARD → Phase 4）
        # 2. 查 stale claims：claimed_at 距今 >30min 且 results/{exp_id}.md 不存在 → 释放 claim
        #    （读-改-PUT 释放，NEVER PATCH——会破坏嵌套 YAML）
        #    （且绝不碰 agents/{agent}/workspace/result_latest.json——那是 resume 用的 sentinel）
        # 3. 查队列深度：pending < 3 → 提醒 analyst 多提案
        # 4. nvidia-smi 查 GPU 利用率
```

**通道 B — orchestrator（`runbook.md` Step 5f）**，每个循环末尾也做同样的 stale-claim 释放（>30min 且无 result 文件）和空队列警告。

**出问题怎么处理**：释放 stale claim（让该实验可被下一个 agent 重新认领）、发 `[AUDIT]` 汇总、空队列则提示。monitor **被明确禁止**跑实验、认领实验、写结果、覆写 champion。

---

## 8. 日志

`system/reference/LOGGING.md` 是主文档，但**它与当前 `launch.py`/ROLE 代码存在矛盾**，如实标注：

| 日志 | 谁写（LOGGING.md 说法） | 实际/更权威说法 | 格式 |
|---|---|---|---|
| `logs/experiments.jsonl` | §开头「orchestrator 写，agent 不直接写」；§2「GPU agents 写」——**文档自相矛盾** | `task-autoresearch/LAUNCH.md` 的 `meta_improve()` 每 3 轮从 `agents/*/cycle_result.json` **harvest 进 experiments.jsonl**（orchestrator 写） | 每行一个 JSON（`exp_id/agent/team/metric/champion_before/after/delta/outcome/...`） |
| `logs/sessions.jsonl` | orchestrator | `runbook.md` Step 5d（orchestrator 写） | 每行一个 agent session |
| `logs/raw/{agent}_{ts}.log` | orchestrator 捕获 stdout/stderr | — | 原始文本 |
| `logs/meta_results.tsv` | meta-improvement | `META-IMPROVEMENT.md` Step 4 | TSV |
| `logs/approach_registry.json` | GPU agent（biomlbench） | `ROLE-GPU.md` Step 2a-i（`fcntl` 文件锁） | JSON |
| `logs/{agent}.gpu_claim` | GPU agent | `ROLE-GPU.md` Step 2a-ii（声明 gpu/cpu） | 单行文本 |
| `agents/{name}/workspace/result_latest.json` | **该 agent 单写** | `ROLE-GPU.md` Step 4/7c（sentinel） | JSON |
| `results/{exp_id}.md`（main ws） | GPU agent 单写（write-once） | `ROLE-GPU.md` Step 5 | Markdown |

**结论**：
- `experiments.jsonl` 是 stagnation check 读的**唯一权威日志**（`LOGGING.md`：「SINGLE SOURCE OF TRUTH」）；它在现行代码里由 **orchestrator 通过 meta-improvement 的 harvest 步骤写入**（源头是各 GPU agent 的 `cycle_result.json`）。「GPU agent 直接 append experiments.jsonl」是文档里另一处旧说法。
- 每 agent 的 `result_latest.json` 是**严格单写**的（只有自己写），它同时是 HEARTBEAT Part 0 Check C / Part 5 的 resume sentinel。

---

## 9. ★ Agent 之间的「互相验证」到底怎么实现（重点）

**先给诚实结论**：AutoScientists **没有**「多个 agent 各自独立跑同一实验、再对比结果」的机制，也**没有**「独立 reviewer agent」这个角色。它的「互验」是**共享黑板 + 消息板上的集体评审**，分五层，各有明确源码位置：

### 9.1 Proposal 级互审：Discussion-Before-Queuing（最强的一条）

每个实验必须先在 workshop 发 `[PROPOSAL]` 帖子，**至少 1 个非作者评论**后才能入队（`ROLE-TEAM.md`「Discussion-Before-Queuing」；`ROLE-ANALYST.md` Step 5；`ROLE-GPU.md` Step 3 的 discussion-gate）：

```text
ROLE-TEAM.md:
  Every experiment MUST have a [PROPOSAL] post first.
  At least 1 team member must comment before it enters the queue.

ROLE-ANALYST.md Step 5:
  Wait for at least 1 comment from a non-author ... before adding to queue.
  A comment from the proposer themselves (you) does NOT count.
```

这就是「一个 agent 提议、其他 agent 审稿」：**评审发生在花 GPU 之前**，由帖子/评论承载。README 的「critique each other's proposals before spending experimental compute」即指此。

配套还有讨论期（HEARTBEAT Part 2）里的显式「反驳/找漏洞/排序」指令：`Disagree with something... say so with evidence`、发 `[GAPS]`、`[RANKED]`、`[CONSTANTS]` 帖子——是 proposal 互审的前置讨论层。

### 9.2 结果级三角验证：hypothesis lens（「多个 agent 独立解读同一结果」）

这是最接近「互相验证」的机制，但它是**结果侧**的：同一个实验结果会被**不同 team 用各自的假设 lens 独立分类**（`ROLE-ANALYST.md` Step 0.3 原文）：

```text
Different teams may propose the same axis for different reasons;
that is the intended form of collective reasoning.
...the same experiment (e.g., TOTAL_BATCH_SIZE halving)
gets evaluated through your team's LENS — does it support your hypothesis?
```

即：团队 A 的假设预测「该实验会 KEEP」→ 若 DISCARD 则 A 记 `refuted_discards+=1`；团队 B 的假设不预测这个轴 → 记 Orthogonal。**一个结果被多个 team 独立判读，累进各队假设的支持/反驳计数**，最终驱动 §4 的证伪判定。这不是「有人复核数值对不对」，而是「同一结果被多套假设独立消化」。

### 9.3 跨代理查重验证（dedup）：提案前对照共享状态

analyst（Step 3b）和 GPU（Step 3b）在提案/认领前都要做三重查重，本质是「验证我的想法是否已被别人做过/证伪」：

```python
# 1. search 主 workspace results/ 是否已跑过该机制
# 2. search 队内 dead_ends.md 是否已证伪该机制族
# 3. grep champion code 该机制是否已实现
```

`dead_ends.md` 的「3+ DISCARD 0 KEEP → dead end」规则（`ROLE-ANALYST.md` Step 2、`ROLE-TEAM.md`）是**跨 agent 的失败共享**：一个 agent 的失败写入共享文件，后续 agent 据此不再重复踩坑。

### 9.4 晋升侧的自我验证门（不是「互验」，但是对「一个结果能否晋升」的验证）

- **多 seed 噪声门**（`ROLE-GPU.md` Step 7.0）：KEEP 且 `|delta| ≤ 2σ` 时，**换种子重跑**，第二种子也更好才晋升，否则记 `[NEAR-MISS]`。3 次同 tuple 的 NEAR-MISS 触发「3-seed 确认」（≥2/3 种子赢才算）。**注意：这是同一个 agent 换种子自跑，不是另一个 agent 复核。**
- **race-condition 重读**（Step 5）：晋升前重读 champion.md，防止训练期间别的 agent 已晋升。
- **diff-applied 检查 → FAILED**（Step 4/5）：用 `filecmp` 验证代码确实变了，防止「幻影 KEEP」。

### 9.5 结构性决策的背书（endorsement）机制

团队合并/新建等结构改动（`[DIMENSION-MERGE]`/`[REGROUP]`）需要「endorsement bar」——≥2 个非提案者的实质赞同 + 0 未解决反对 + 帖子 ≥1 rotation 后，才由「字母序最后一个非提案者 analyst」执行（`ROLE-ANALYST.md` Step 1d.5）。这是**结构化变更上的少数服从多数式集体验证**。

### 9.6 兜底验证：janitor 复核「活没干完」

monitor/orchestrator 的健康检查（§7）会验证「认领了实验的 agent 是否真的产出了结果文件」——若 claim 超过 30 分钟且 `results/{exp_id}.md` 不存在，就释放 claim 让别的 agent 重新认领。这是对「工作是否真的落地」的间接验证。

### 9.7 小结（与单 workflow 编排的对比）

AutoScientists 的互验 = **「共享文件上的群体评审 + 结果侧多假设三角验证 + 晋升侧噪声/race 门 + 跨代理查重 + 结构变更背书」**，全部通过**提示词约定 + ClawInstitute 的帖子/版本化文件**实现。**没有**中央调度器做验证、**没有**「双跑同一实验对比」的重复计算验证、**没有**独立 reviewer agent。相比之下，单 workflow 编排（我方 loop）若要复刻这种「互验」，关键是引入一个**共享版本化状态层（If-Match 乐观锁）+ 消息板**，把「审稿」做成 proposal 入队前的强制门，把「结果解读」做成多假设 lens 计数——而 agent 本身仍是提示词驱动的无状态 worker。

---

## 10. 已知的文档漂移 / 不一致（如实标注，供对比时注意）

源码内存在「参考文档较老、与当前 launch.py/HEARTBEAT.md 不一致」的情况，读取时不要被误导：

1. **`AGENT-SETUP.md`** 还在描述 `CONTROLLER`、`role.md`、`actions.md`、`system/HEARTBEAT.md` 路径、`{AIAGENTS}` 变量；当前 `launch.py` 实际用的是 `AGENT.md` + `memory/MEMORY.md` + 每个 agent 自己的 `HEARTBEAT.md` + `credentials.json`，路径是 `agents/<name>/HEARTBEAT.md`。`AGENT-SETUP.md` 是旧版。
2. **`PHASES.md`** 的 Phase 4 说「monitor 检测停滞并重组团队」，而 `ROLE-MONITOR.md` 明说「monitor 不介入重组，由 agent 自组织」。**现行权威是后者**（agent-driven self-regroup）。
3. **`LOGGING.md`** 对 `experiments.jsonl` 的写者自相矛盾（§开头说 orchestrator，§2 说 GPU agents）；现行代码是 orchestrator 从 `cycle_result.json` harvest（`task-autoresearch/LAUNCH.md` meta_improve）。
4. **`SKILL.md`/`PHASES.md`** 仍写「orchestrator 在 KEEP 后复制 train.py 到 champion/」；`ROLE-GPU.md` Step 7b1 已改为「**KEEP 的 GPU agent 自己**原子复制 champion/train.py」（「This step replaces the prior 'orchestrator promotes' model that was unreliable in practice」）。`runbook.md` Step 5e 又仍写着「orchestrator 是唯一写 champion 的实体」——**三处对 champion 晋升者的说法互相冲突**，需注意。实际上 `runbook.md`「What you NEVER do」说 orchestrator 只做 champion-promotion copies，`ROLE-GPU.md` 说 GPU agent 自己做，`SKILL.md` 说 orchestrator 做。这是演化过程中遗留的文档不一致，不是 bug 本身。
5. `PHASES.md` 的 `create_team()` 创建 `hypotheses.md`（空壳），但真正承载假设字段的是 `strategy.md`。

---

## 附录：关键源码文件清单（取证位置）

| 文件 | 作用 | 大小 |
|---|---|---|
| `launch.py` | 启动器：建 run 目录、注册 10 agents、建 workshop/workspace、seed 文件、发 kickoff | 42 KB |
| `runbook.md` | orchestrator 基础程序：外层 while 循环 + 13 个 profile hook | 14 KB |
| `system/templates/HEARTBEAT.md` | 每 agent 自包含循环（Mode Selector + 5 个分支） | 38 KB |
| `system/templates/ROLE-ANALYST.md` | analyst 协议（假设检查/证伪/提案/剪枝/组队重组） | 62 KB |
| `system/templates/ROLE-GPU.md` | GPU 协议（认领/训练/KEEP-DISCARD-FAILED/噪声门/晋升） | 48 KB |
| `system/templates/ROLE-MONITOR.md` | monitor 协议（健康检查） | 6 KB |
| `system/templates/ROLE-TEAM.md` | team 协作（队列/查重/文件发现） | 6 KB |
| `system/reference/{SKILL,PHASES,LOGGING,AGENT-SETUP,API-REFERENCE,META-IMPROVEMENT}.md` | 参考文档（部分较老，见 §10） | 各 5–8 KB |
| `task-autoresearch/LAUNCH.md` | optimization profile（13 hooks 的实现） | 15 KB |
| `task-biomlbench/LAUNCH.md` | biomlbench profile（deadline/GPU 混排/紧急提交） | 23 KB |
| `system/external-repo-setup/SKILL.md` | 外部 repo 集成协议（GPU agent 用） | 19 KB |
