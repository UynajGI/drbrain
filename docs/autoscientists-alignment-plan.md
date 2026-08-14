# AutoScientists 对齐计划原子书

> 原则：**扬长避短、取其精华去其糟粕**，结合 drbrain 现有环境（FunctionAgent + PluginRegistry + Workflow + SQLite + 16GB db + GPAW + flash + 日志体系）。
> 目标：把 loop 从「单 workflow 同一模板 + 无互验」对齐到「角色分化 + 互验闭环 + 实算门」，**不引入** Claude Code / ClawInstitute / 完全自主调度。
> 参考：`docs/autoscientists-gap-report.md`（差距） + `docs/autoscientists-architecture.md`（源码取证）。

---

## 一、取舍定案

**取（AutoScientists 精华）**：
1. 角色分化——critique 是「批判者」、verify 是「核验者」，各自独立 system prompt + 职责契约，不是同一模板换 prompt。
2. 互验闭环——critique 分数**必须流入** verify；verify 产出结构化 Supports/Refutes/Orthogonal，代码判定，不靠 LLM 一句话。
3. falsification 代码化——拿证据比对 `h.falsification`，计数驱动证伪。
4. 实算验证——晋升（verified）需实际计算结果，不是 LLM 空口。

**弃（AutoScientists 糟粕）**：
1. Claude Code 会话 → 用 LlamaIndex FunctionAgent（已有）。
2. ClawInstitute Node 服务 + 乐观锁 → 用 director 单写 SQLite + 文件工作区（已有）。
3. 完全自主调度（无中央编排）→ 保留确定性 Workflow 骨架（「自由在节点内、纪律在节点间」）。

**结合我的环境**：角色 prompt 用文件/常量定义（领域无关，不硬编码材料学）；工具面用现有 PluginRegistry（run_python/check_job/predict_*/search_*/read_skill）；状态用现有 ResearchState + 文件工作区 + traces 日志。

---

## 二、原子任务书

### 阶段一（核心：角色分化 + 互验 + 实算门）

#### T1 角色分化：critique / verify 独立 role prompt
- **目标**：critique 用「批判者」role（职责：独立打分 + 找漏洞 + 反驳），verify 用「核验者」role（职责：拿证据判定 Supports/Refutes/Orthogonal）。两者 system prompt 不同、职责契约不同。
- **改动**：`src/drbrain/loop/workflow.py` 的 `build_node_agent` 支持按节点传 role；新增 role prompt 定义（放 `src/drbrain/loop/roles.py` 或 `prompts/`，领域无关）。
- **验收**：critique 和 verify 的 system prompt 内容不同，各自职责清晰；不破坏现有测试。

#### T2 verify 消费 critique 分数
- **目标**：`h.score` 流入 verify 的判定输入；低分假设（如 score < 0.5）需更强证据才能 verified，否则降级 prediction/falsified。
- **改动**：`workflow.py` critique→verify 数据流（把 critiqued 假设连同 score 传入 verify）；verify 的判定逻辑读 score。
- **验收**：低分假设不会被直接 verified（需额外证据）。

#### T3 falsification 代码化（Supports/Refutes/Orthogonal 三角验证）
- **目标**：verify 对每个假设产出结构化计数 `{supports, refutes, orthogonal}`（基于检索证据），代码判定：`refutes >= 2 且 supports == 0` → falsified；`supports >= 1 且 refutes == 0` → verified；否则 → prediction。
- **改动**：`events.py` 加验证结果结构（如 `Verification` dataclass）；`workflow.py` verify 产出计数；`director.py` `_absorb` 用代码判定替代「只看 verified/falsified 列表」。
- **验收**：falsified 是代码计数判定，不是 LLM 自由裁量。

#### T4 实算门（逼 verify 真算 DFT）
- **目标**：verify 节点在环境有 `run_python`/`check_job`/`materials-env` 工具时，**强制要求产出一份实际计算结果**（能量/能带数值，经 `run_python` 跑 GPAW/ASE 或等价计算）作为 verified 的证据；没有实算的假设一律降级 prediction（不可 verified）。
- **改动**：`workflow.py` verify 节点 prompt（要求先读 `materials-env` skill + 用 `run_python` 实算）+ 判定逻辑（检查 verify 返回里是否含实际数值）。
- **验收**：无实际计算结果的假设不可能 verified；实算结果（能量/能带数值）出现在 verify 返回。

### 阶段二（proposal 评审 + 查重）

#### T5 proposal 评审门
- **目标**：identify_gaps 提出的假设，先经 critique 独立评审（给分 + 反驳），分数过低的假设直接标记 DISCARD 或降级，不进 verify（对应 Discussion-Before-Queuing 简化）。
- **改动**：`workflow.py` identify_gaps→critique 流 + critique 对低分假设的处理。
- **验收**：低分假设在 critique 阶段就被过滤，不进 verify。

#### T6 查重门
- **目标**：identify_gaps 提案前强制对照 prior_context（champion/dead_ends），不重复提出已证伪/已确认的假设。
- **改动**：`workflow.py` identify_gaps prompt 强制读 prior_context + 逻辑检查重复。
- **验收**：已证伪假设不再被重复提出。

### 阶段三（per-agent 记忆，重，后续）

#### T7 per-agent 记忆（可选，T1-T6 完成后单独做）
- **目标**：critique/verify 跨 cycle 记住自己的评审历史（写 per-role memory 文件），不每次从零。
- **改动**：`director.py` 加 per-role memory 文件（`knowledge/role-{critic|verifier}.md`）。
- **验收**：跨 cycle 评审历史可追溯、被注入。

---

## 三、执行顺序与依赖

```
T1（角色分化，基础）
 ├─→ T2（verify 消费分数）──→ T3（三角验证代码化）──→ T4（实算门）
 └─→ T5（proposal 评审门）──→ T6（查重门）
                                    └─→ T7（记忆，后续）
```

T1-T6 改同一批文件（workflow.py / events.py / director.py），有依赖，**单 subagent 串行执行**，避免并发冲突。T7 单独一轮。

## 四、验证（每阶段必跑）

```bash
cd /home/jiangyuan/drbrain && uv run ruff check src/drbrain/loop/ src/drbrain/extractor/
uv run pytest tests/test_loop_director.py tests/test_loop_agent.py tests/test_loop_workflow.py tests/test_llm_client.py tests/test_rag_llm.py -m "not integration" -q
```

外加一条端到端冒烟：跑一轮 loop，检查 traces/cycle-NNN.json 里 verify 是否含 Supports/Refutes 计数 + 实算数值（T3/T4 生效）。
