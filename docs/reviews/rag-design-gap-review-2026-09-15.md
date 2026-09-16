# RAG 设计与实现差距审阅

审阅基准：`phy@e5b7a095c3f5ffefe8cf100f36f133bf6bd93c89`，2026-09-15。已纳入 [CLI/RAG handoff](../handoff-cli-rag-usage-2026-09-15.md)，不再用已经移除的 `rag-usage` worktree 代表当前实现。

收尾时主分支新增 `2c66a88`，仅同步文档，生产源码和测试未变；本文已考虑其中对真实模型验收与计划状态的更正。下述运行证据仍适用于当前生产代码。

**结论：CLI 重构和统一树算法的主体已落地，但还不能判定设计全部完成。** 最新提交修好了新库三路接线和重排，原有“新库只能走 tree”的发现不再适用于本提交。剩余问题主要在版本一致性、证据读取边界、重建及故障恢复；迁移、基线和规模验收也未完成。

本文审阅代码与契约，不修改生产实现、不重跑原始 10k ingest。所有新增复现使用合成材料和隔离的 SQLite/Zvec；没有移动、删除原始语料，也没有重启 8010。

## 1. 需要修复的差距

以下位置均指上述提交的主检出源码。P1 表示本阶段验收前应修复；P2 表示仍需补齐的生产行为契约。复现结果见第 4 节。

### P1 — 发布版本标签与实际读取内容不一致

[sql_retrie.py:973](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:973) 把调用方的活动主库传给 BM25；[最终物化](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:863) 同样查询活动 `tree_nodes/content_blocks`。但向量和 tree 读取已发布 ANN/SQLite，最终 evidence 又统一标上 `active_tree_generation`，能力声明 `snapshot=true`。

触发方式：先发布 G1，再登记一篇含独有关键词的新材料，不运行 `index build`，随后用 BM25 查询该词。新材料不在 G1 快照内，却能以 G1 的 generation 身份返回。这会使“此答案使用了哪个版本的语料”失真；三路也不再面对同一语料集合。

另外，vector helper 和 tree leg 分别重新解析 active 指针，没有接收请求开始时捕获的同一个 generation；查询期间发布切换的风险也应一起封闭。

应让整个请求固定同一发布上下文，或明确采用经过修订校验的 live 契约；不能把活动主库命中包装成旧快照证据。设计依据：[统一存储的修订约束](../unified-tree-rag-design.md)。

### P1 — tree 融合丢弃实际读取范围，重新返回整块正文

[sql_retrie.py:987](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:987) 把 `TreeLegHit` 缩成 key/score；其 `text`、实际 `char_start/char_end` 和读取凭据没有参与物化。[物化函数](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:847) 随后用叶节点范围与整块 `b.text` 重建结果。

触发方式：导航器对一个真实叶只执行 `read(char_start=7, char_end=27)`，随后结束。实际复现中融合结果变为 `[0,241)` 的 241 字符整块，而不是实际读取的 20 个字符。

这违反设计“最终原文候选必须来自实际读取记录”的要求，也使工具 token 预算和最终引用范围脱节。应保留 tree 的读取凭据与准确子范围；若需要扩大上下文，应执行并记录一次明确的读取。

### P1 — 论文作用域没有进入导航工具的读取权限

[run_tree_leg](/home/jiangyuan/drbrain-phy/src/drbrain/tree/leg.py:225) 只把 `local_ids` 传给 ANN 入口和再次搜索，没有传入只读存储/工具层。[TreeTools.visible](/home/jiangyuan/drbrain-phy/src/drbrain/tree/tools.py:73) 只检查节点存在且 `ready`。

触发方式：请求限定 `local_ids=['p1']`，规划器读取一个确实存在的 p2 叶节点，工具仍返回 p2 原文，tree leg 仍产出 p2 命中。这里使用合法节点 ID，没有伪造不存在的证据。

最终结果过滤不能替代工具读取限制：越界正文可能已经进入规划器上下文。应把允许的论文/范围落实到 `read/expand/parents/read_scope`，包括跨论文 region 的展开。

### P1 — 修改聚类参数或使用 `--force` 没有真正重建已有层次

[prepare.py:539](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:539) 检测签名后，实际 frontier 仍来自 `leaves_missing_parent()`。已有父节点的叶不会因 lambda/clustering 参数改变重新进入计算；[无 frontier 分支](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:549) 甚至直接更新水位。`force=True` 只跳过部分早返回，空 frontier 随后仍被跳过。

复现：16 叶的合成语料完成建树后，把 lambda 从 4 改为 0，第二次没有任何构建 round，旧层次被新签名认证；`force=True` 也没有层次 round。模型/摘要契约改变导致的旧 region 退役已修好，但不等于所有算法参数变化都能重建。

这直接影响 T57 消融是否真实，也使 `--force` 的“每阶段完整重建”帮助说明不成立。需要独立的算法版本失效规则和完整重建 frontier。

### P1 — 发布失败后，正常重跑跳过发布

[prepare.py:193](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:193) 只用本次新增嵌入数/region 数决定 `changed`；[第 200 行](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:200) 在无计算变化时跳过发布。

复现：向量和层次成功后，让首次 publication 抛出一次 I/O 异常；恢复 publisher 后原命令重跑。因为前面计算已完成，第二次返回 `publication=skipped, reason=unchanged`，没有 generation，却能得到 `ok=true`。`tree.prepare.last` 记录上次失败的改动是有效进展，但后续成功空跑会覆盖记录，不能替代发布恢复。

应单独追踪“工作产物尚未发布/上次发布失败”，恢复发布而不强迫重新计算模型产物。

### P2 — BM25 先截断全库候选，再应用论文作用域

[sql_retrie.py:785](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:785) 先 `search_content(limit=cap)`，再按论文过滤。限定论文里的正确命中只要排在全库 cap 之后就消失；默认 cap 较大只降低发生频率，并不修复契约。

复现用两篇都包含同一词的文档，限定全库排序第二篇、cap=1：限定范围内明明有命中，却返回空。应把范围加入 FTS 查询条件后再排名/截断。

### P2 — 导航预算耗尽仍被对外标记为成功

[tree/leg.py:284](/home/jiangyuan/drbrain-phy/src/drbrain/tree/leg.py:284) 在存在任意叶命中时直接设 `status='ok'`；[融合层](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:989) 也用是否有 entries 判断成功。

复现：`max_calls=1`，读取一个叶后预算耗尽，内部 `navigation_status='partial'`、仍有 unresolved，但外层 status 为 ok。诊断字段保留 partial 是进展，仍不满足冻结协议的“预算耗尽输出 partial”。应把这个状态传播到腿及整体检索结果。

### P2 — CPU/离线 ingest 仍被全局 LLM 列表硬性阻断

[db_ingest.py:291](/home/jiangyuan/drbrain-phy/src/drbrain/cli/_helpers/db_ingest.py:291) 在判断 `DRBRAIN_OFFLINE` 之前强制要求 `llm.models` 非空。此时规范正文已经提交。

复现：普通 Markdown、`DRBRAIN_OFFLINE=1`、`llm.models=[]`，CLI 返回 `failed=1, errors=['Exit: 1']`。即便分类分支会采用启发式，也走不到那里。在线分类仍读全局 models 列表，尚未完全收敛到设计的具名角色。

应让不需要模型的入库路径独立完成；需要模型的阶段按相应角色解析，避免先提交正文再因不相关配置报告整篇失败。

### P2 — 关闭 tree 后，就绪检查不认识已经可用的统一 BM25/vector

[index_commands.py:466](/home/jiangyuan/drbrain-phy/src/drbrain/cli/index_commands.py:466) 仍保留“没有 SQL 快照且没有 tree 路就不可检索”的旧判断。它没有随统一三路修复更新。

复现：统一索引已发布，启用 `['bm25','vector']`，`search` 两路可读；`index status` 却报告 `no_published_index`。这阻碍“任意启用检索路线”的配置契约。就绪状态应根据所选路实际读取的统一 generation 判断。

## 2. 最新 handoff 真正收口了什么

| 项目 | 当前评价 |
| --- | --- |
| CLI 命名与兼容 | Phase 3 已合并；`graph build/embed/closure`、隐藏别名和 pipeline 转调有契约覆盖。不能再按旧 worktree 状态评价。 |
| 新库三路接线 | `6f7324e` 实现规范 FTS、共享叶 ANN、统一 tree；修复 Zvec 守卫顺序。旧“三路缺两路”发现关闭。 |
| 融合与重排 | 当前统一路径已经执行 RRF、去重、CrossEncoder 重排和文档多样性处理。旧“rerank=true 不调用”发现关闭。 |
| 统一树本体 | 结构先验进入 global/local soft posterior；结构/语义候选共用摘要服务和成员关系；导航使用真实叶读取。不是给旧 SQL LIKE 换名。 |
| 新材料存储 | canonical blocks/leaf 成为事实源，新 ingest/ingest-link 不再生成每篇 raw.md/tree.json；原始材料保护仍在。 |
| 失败状态 | 上次构建结果进入 status/verify，模型失败不再只按节点数量判 ready；发布恢复仍有第 1 节所列问题。 |
| 真模型导航 | handoff 记录修复多 tool_calls 应答不完整和推理 token 预算不足；本轮回归覆盖相关代码。 |
| RAPTOR 对照命名 | 已把统一树平铺消融与真实 RAPTOR 分开；缺独立产物时 fail-closed。它纠正了比较口径，但不代表真实对照已完成。 |

## 3. 仍未完成的设计/迁移范围

| 设计要求 | 实际边界 |
| --- | --- |
| 一条通用 `ingest → index build → search/ask` 主线 | 当前统一检索主要走 SQL engine 的新分支。`rag_engine=llamaindex` 仍需隐藏的 `rag index`；`index build` 明确不准备其独立 generation，见 [实现说明](/home/jiangyuan/drbrain-phy/src/drbrain/cli/index_commands.py:235)。不能把先前 `7be77b9` 的行为当成当前行为。 |
| 旧库也统一使用最小存储 | 已发布 SQL 快照仍优先，新统一索引不会自动替换它。旧分片 ingest 尚不写 canonical；merge 支持统一表不等于分片生产者已迁移。handoff §5 也明确承认这两点。 |
| 最小事实存储 | 文献级重复已减少；发布仍复制整库 SQLite 与 ANN，兼容引擎另有索引。需区分“一个事实源”和“只有一个物理副本”，后续明确保留/回收策略。此审阅不建议自动删除历史快照或原素材。 |
| LlamaIndex 自带 BM25/vector | 新默认统一 SQL 路的 BM25 是主库 FTS，vector 是共享 Zvec 适配；不是直接使用 LlamaIndex 原生 BM25/VectorStoreIndex 两个对象。可选 LlamaIndex engine 的存储尚未统一。 |
| 有界语料级构建 | 当前 frontier 整体进入聚类，摘要候选按组串行处理；embedding batch_size 不等于 T59 的跨批 frontier 算法。handoff 记载 1049 叶构建约 40+ 分钟，也不能据 2 篇成功推断 10k 成本。 |
| 新算法效果验收 | 真实 RAPTOR/PageIndex 对照、消融和参数冻结尚未完成；不能宣称新 tree 的质量优于原方案。 |
| 10k 迁移与三路验收 | 旧 10k ingest 完成不等于新 schema/ANN/tree 已迁移并验收。原计划仍保留相应门。 |

原子计划有 **64 项，52 项勾选、12 项未勾选**。未勾选为 T47、T48、T54、T56–T64：分别涉及验收环境/CLI、混合迁移、真实基线与消融、100 篇、跨批构建、10k 迁移与检索、默认切换、回归交接。

这里需要区分验收账本缺失和功能未完成：handoff §4 已记录 Spark/DeepSeek 的 **2 篇、12 叶**真实模型冒烟，不能继续声称“4B 始终跑不起来”。`2c66a88` 已同步原子计划，明确本机可启动模型、T47 待按清单复验；但 `acceptance.jsonl` 仍只有 70 条旧记录，T48 保留 reopened，新运行尚未追加到规定的验收账本。本次没有复跑活模型，真实运行结果引用 handoff 的自述，不冒充独立实测。它也不能覆盖混合素材迁移、基线和 10k 验收。

## 4. 验证记录

本轮对主分支完成了全量非集成测试和两组独立契约复现，测试期间生产源码保持 `e5b7a09`，没有应用修复。

| 验证 | 结果 |
| --- | --- |
| 项目 `tests/`，排除 integration | **4025 passed、15 skipped、34 deselected**；没有项目测试失败。 |
| 第一轮审阅的 11 个探针，在当前提交复跑 | 3 passed：SQLite/Zvec 下三路可读、重排实际调用。6 个有效失败：重建参数/force 两例、发布恢复、越界读取、partial、离线 ingest。另 2 例因“缺少 BM25/vector”前提已被修复而失效，不计产品缺陷。 |
| 合并后新增 4 个探针 | **4 个契约断言失败**：版本混用、读取范围丢失、scope 后截断、关闭 tree 的 status 误报。没有夹具初始化错误。 |

因此，**10 个有效失败用例对应本文 9 类差距**，其中 lambda/force 两例归于同一个重建问题。项目现有回归通过和设计仍有缺口可以同时成立：现有覆盖主要缺少这些边界情形。

完整回归与第一组探针同进程运行，原始汇总为 `8 failed, 4028 passed, 15 skipped, 34 deselected`，耗时 **630.46 秒**；8 个失败全部来自审阅文件，其中 2 个是上表说明的旧前提。第二组运行耗时 **87.54 秒**。不会把这个总结果简写成“所有测试全绿”，也不会把两个过时探针当作当前故障。

复现工件仅放在审阅目录，采用 `repro_` 文件名，不加入默认测试发现；执行时原名为 `test_rag_48a3c6f_review.py`、`test_rag_e5b7a09_review.py`，运行后只调整文件名和说明，断言没有改变：

- [第一轮独立探针](repro_rag_48a3c6f.py)：故障恢复、重建、范围、状态及 CLI 契约；在当前提交应排除两个旧前提用例。
- [合并后新增探针](repro_rag_e5b7a09.py)：版本一致性、实际读取范围、BM25 scope、关闭 tree 的就绪检查。

重现本轮仍有效的探针：

```bash
cd /home/jiangyuan/drbrain-phy
env -u DRBRAIN_ROOT \
  DRBRAIN_REVIEW_PROJECT="$PWD" PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  docs/reviews/repro_rag_48a3c6f.py docs/reviews/repro_rag_e5b7a09.py \
  -k 'not readiness_cannot_certify_missing and not verify_cannot_pass_when_requested' \
  -q --tb=short --show-capture=no
```

测试使用主检出的源码、现有虚拟环境；清除 ambient `DRBRAIN_ROOT`，保留 pytest logging 插件。模型计算通过测试替身隔离，SQLite/Zvec 为真实实现。测试通过不能替代真实模型质量、arXiv 网络检索和 10k 验收。

## 5. 建议收口顺序

1. 先固定发布版本与实际读取证据的契约，封闭范围越界；补上对应负向测试。
2. 修复算法参数/force 重建和 publication 的可恢复状态，再验收原命令失败后重跑。
3. 统一 scope、partial、角色配置及 status 口径；保持三路可任意开关。
4. 用生产 CLI 跑完 PDF/MD/TeX 小样本的 ingest、index build、status/verify、search/ask，把命令、结果与日志写回统一验收记录。
5. 再处理旧 SQL/分片迁移、真实对照、100 篇和有界 10k。此阶段不重新盲跑全量 ingest。
6. 按用户于 2026-09-15 确认的规则：10k 全链路及故障恢复验收完成后，将本轮阶段计划、handoff、历史审阅报告移入 `docs/archive/rag-10k/`；保留提交和验收证据关联，修复链接，并把最终算法/契约归并到长期文档。`docs/` 主层保留当前设计、配置、CLI、运维与验收入口。此动作已记入 T64，当前不提前归档。
