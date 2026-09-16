# 统一 Tree RAG 算法附录：冻结的可执行协议

2026-09-15。状态：**协议已冻结**（计划 T04/T05/T06 缺省值；参数在 T57 用开发集冻结后不再改）。本文件把
[总体设计](unified-tree-rag-design.md) 中的公式落成无歧义、可测试的定义；参考实现分别在
`src/drbrain/tree/contracts.py`、`posteriors.py`、`affinity.py`、`cost.py`，fixtures 在
`tests/tree/fixtures/`。

本附录只定义协议与参考实现，不宣称检索效果；效果结论属于 T57/T58/T62 的实测。

## 1. 身份与坐标（T03，normative）

* 正文以 `content_blocks` 为唯一可检索正文：半开区间 `[char_start, char_end)`、按 `ordinal`
  连续、拼接后与原规范化正文逐字符一致；分隔符/空白/未归档残余必须落在某个 block 内。
* PDF 页码 1-based inclusive；MD/TeX 行号 1-based inclusive；两种 locator 不互相冒充。
* `content_hash` 只覆盖文本（用于跨出处复用嵌入计算）；`fingerprint` 覆盖身份+内容+成员
  （用于变更检测）。相同文字在不同文献中 node_id 不同。
* 叶身份：`nl-<sha256(schema|leaf|local_id|revision|block_id|char_start|char_end)[:24]>`。
* 区域身份：`nr-<sha256(schema|region|members_key|contract_digest)[:24]>`，其中
  `members_key = join(sorted(child_id@child_revision))`。成员集合或契约不同即不同节点，禁止
  静默覆盖。
* 区域节点**没有** page/line 字段；跨篇摘要不得伪造连续页范围。父子页范围重叠是定位重叠，
  不代表正文重复。

## 2. 两级后验协议（T04，normative）

上游 RAPTOR 的真实顺序是：全局 GMM →（严格 `>0.1` 阈值）→ 每个全局子集内独立局部 UMAP/GMM →
（各自严格 `>0.1` 阈值）。本协议保留该两阶段顺序与阈值语义：

1. `PosteriorStage(stage="global", ...)`：保存**原始** `predict_proba` 结果（行=节点，列=分量），
   绝不把最终离散 label 当概率。
2. 局部阶段 `PosteriorStage(stage="local", subset_of=<global 分量>)`：只在对应全局子集的行上拟合，
   行保持该子集坐标；局部分量命名 `"<global>.<local_index>"`（两个局部 UMAP 空间不可比）。
3. 每一阶段各自做结构重权：

   $$ \widetilde p(k\mid i) = \frac{p(k\mid i)\exp(\lambda A(i,k))}{\sum_j p(j\mid i)\exp(\lambda A(i,j))} $$

   然后各自用严格 `>` 阈值化。**禁止**把 global 概率与 local 概率相乘后再统一阈值化，并声称
   与上游等价（0.2×0.2=0.04 会被错误过滤，而上游两阶段都能通过）。
4. `lambda = 0` 时 `reweighted()` 是恒等操作（返回同一对象），阶段成员关系与未修改后验一致；
   这是消融条件。它**不等于**整个新管线与原始 RAPTOR 等价：覆盖补全、成本门、确定性修复、
   入口策略都是独立新增量，必须单独关闭或单独报告。
5. 空分配（所有分量都不高于阈值）必须显式保留该行（`membership()` 返回空 tuple），节点不
   因此失去可达性。数据/参数相同的阶段结果可核验：BIC 选簇、行序、阈值、模型拟合都在签名中。

参数：`lambda ∈ [0, 12]`，默认 2.0（**临时值**，T57 在开发集选定后冻结）。阈值默认 0.1（上游
值），每一阶段独立配置。

### 2.1 结构相容性 A(i,k)（T04/T29）

* 暂定簇画像：只用**未修改**后验构造；对簇 k，成员权重 = `p(k|i) × max(tokens(i), 1)`，按唯一
  原文范围计算，软多父路径不重复计权。评估节点 i 自身在构造画像时排除（`exclude_row`）。
* `doc_share(i,k)` = 节点 i 所属文献在簇 k 画像中的 token 质量占比，`∈[0,1]`；跨文献无质量时
  为 0。
* `path_similarity`：同文献内标题路径公共前缀长度 / `max(1, min(len(a), len(b)))`；任一侧缺失
  标题时为 0（中性，不虚构信号）。
* `A(i,k) = doc_share × 同文献画像的 token 加权平均 path_similarity`，取值 `[0,1]`。
  空画像 → 0；跨文献 → 0（中性，不是硬屏障）。
* **偏置声明（normative）**：重权后归一化会压低跨文献分量的相对概率，可能把它压到阈值以下；
  这是明确的同篇偏好，不得声称对跨篇召回中性。λ 的选择必须包含跨篇多证据问题（T57）。

## 3. 父节点接受与成本协议（T05，normative）

成本单位 = pipeline tokenizer 的 token；路由开销 = 每步工具调用的 token 常数
（`tool_overhead_tokens`，默认 60）。

* 唯一来源覆盖 `Coverage`：对每个 `(local_id, revision, block_id)` 合并区间取并集；相同的
  范围只算一次。读取成本 `read_cost = union_len × tokens/char`（按覆盖内平均值，四舍五入）。
* 单目标代理（沿用 PageIndex 的路由思想，公式重定义）：路由成本
  `route_cost(estimated) = summary_output_budget + tool_overhead`；实际
  `route_cost = summary_tokens + tool_overhead`。
* 两段判断：
  1. **预筛（调用模型前）**：成员数 ≥ 2；覆盖非空；成员集合未处理过（`duplicate_group`）；
     成员 token 量 ≤ 摘要输入预算（超预算 → 由构建器重切/重聚类，不许静默截断）；预估有压缩
     收益（`read_cost > route_cost(estimated)`）。
  2. **复核（摘要生成后）**：摘要非空；`finish_reason` 正常（否则 `summary_truncated`）；实际
     token ≤ 输出预算；成员来源引用覆盖全部唯一范围（`coverage_incomplete` 拒绝）；实际
     `read_cost > route_cost`。
* 被拒绝的候选组不产生父节点；**原成员留在 frontier**（由构建器保证并测试）。所有拒绝原因
  取自 `REJECTION_REASONS`。
* 单目标假设与多目标成本分开记录：`estimate_multi_target_cost` 输出 `read_all_unique`、
  `route_then_read`、`route_only_lower_bound` 三项；多证据问题的真实成本在 T57/T58 实测，
  不得当作最坏情况保证。
* 摘要变短不等于事实正确：接受门只证明覆盖/预算/收益；事实正确性属于 T26/T33 的返回校验与
  T62 的评测。

## 4. 阶段状态与对外返回协议（T06，normative）

* `StageState ∈ {absent, running, partial, ready, degraded, failed, stale}`；聚合规则 worst-wins
  （failed > stale > running > partial > degraded > absent > ready），错误不得计为 ready。
  可查询态只有 `ready / degraded`。
* ingest 职责：登记正文、锚点、叶与来源；每篇**不**生成 MD/tree/pages 文件；不把语义层标 ready。
* prepare 职责：增量补 FTS、共享向量、统一层次；只重建失效阶段；无 KG build/closure 依赖。
* ask 职责：只读索引 + chat 角色；禁止隐式建树/提交文档；证据必须带实际读取凭证（`ReadReceipt`），
  模型生成 ID/页码不算证据。
* tree 结果只含一份排名；同一叶经多条路径命中只记一次，保留路径来源；预算耗尽输出可审计
  `partial` 状态。
* CLI JSON：`StageReport.to_json()`（stage/state/counts/reused/created/failed/duration_ms/
  messages/details）为各阶段统一账目格式。

## 5. 参数登记表

| 参数 | 默认 | 范围 | 来源 | 冻结点 |
| --- | --- | --- | --- | --- |
| `lambda` | 2.0 | [0, 12] | 本附录 §2 | T57 开发集 |
| 软阈值 `threshold` | 0.1 | (0, 1) | 上游 RAPTOR | 已冻结 |
| `summary_input_budget` | 3500 | ≥ 512 | 上游 RAPTOR 起点，按实际 endpoint 上下文重算（T18） | T18/T57 |
| `summary_output_budget` | 512 | ≥ 64 | 本附录 §3 | T57 |
| `tool_overhead_tokens` | 60 | ≥ 0 | 本附录 §3 | T57 |
| `min_members` | 2 | ≥ 2 | 审查约束（单成员禁止上抬） | 已冻结 |
| UMAP dim / neighbors / BIC 上限 | 10 / sqrt(N-1) / 50 | 上游值 | RAPTOR 固定版本 | 已冻结 |

每次阶段验收在 `data/integration/unified-tree/acceptance.jsonl` 记录实际使用值（任务ID、代码版本、
命令/测试、退出码、关键指标、结论、日志位置）。
