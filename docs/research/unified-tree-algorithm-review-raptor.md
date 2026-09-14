# 统一 tree 算法审查：RAPTOR 依据与需要补齐的定义

审阅日期：2026-09-14。审查对象：一个持久化 nodes/children DAG，文档结构作为特征参与每层聚类，统一摘要模型、检索器及正文存储。本文是静态设计审查，不是效果验证。RAPTOR 依据固定为 [7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767](https://github.com/parthsarthi03/raptor/tree/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767)。

1. **方向可以成立，但必须明确吸收的是结构信息，不再保留原 PageIndex 树语义。** 原文叶节点可附 document_id、heading path、页面及字符范围；摘要父节点只用统一 children 表达成员关系。这样只有一个层次 DAG。查询中的 `parents` 应是聚类成员关系的逆向查询，`neighbors` 应明确为 canonical 原文次序相邻块，不要把二者解释成 PageIndex 目录父子关系。原文次序和heading路径是来源字段，按需派生即可，不必另存一棵目录树。

2. **上游没有能直接拿来相乘的“最终簇 posterior”接口。** [GMM_cluster](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L60) 内部计算 predict_proba 后只返回阈值化 label。需显式扩展为全局后验 `p_g(g|i)`、每个全局簇子集内的局部后验 `p_l(l|i,g)`，并保留原始 node_id→子集行号映射。局部簇标识必须含 `(g,l)`；两个局部 UMAP 的坐标和高斯编号不能直接比较。替换上游[按浮点向量相等重找原索引](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L107)的做法。

3. **`lambda=0` 复现条件必须限得更严。** 上游在全局后验阈值后决定局部拟合子集，再对局部后验独立阈值化；小子集直接作为一个局部簇。[真实两阶段顺序](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L69)。若把 `p_g*p_l` 当最终后验再阈值化，即使 lambda=0 也不等价，例如两个0.2在原流程均过0.1阈值，乘积0.04却不过。要验证等价，应在两个阶段各自应用同一形式的重权，保持分支、严格大于阈值、输入排序与模型拟合结果一致；对照比较成员集合而非随机簇编号。coverage补全、成本门、确定性修复必须单独关闭或单独报告，不能把“lambda=0”宣称为整个新管线等同原始RAPTOR。

4. **结构 affinity 目前未定义，且容易循环论证。** 若用“最终簇成员是否同章节”决定成员所属，该定义依赖它正在求的结果。可先用未施加结构先验的后验构造固定簇画像，再做一次重权；计算画像时排除被评估节点自身，避免自我加权。高层摘要可能涵盖多个文献和heading，需要沿children汇总来源分布，而非选一个主标题冒充所有成员；按唯一叶来源加权，不能因软多父路径重复计数。完整源码的Node本来不含这些字段，因此这是新增算法数据契约。[Node结构](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_structures.py#L4)。

5. **“跨文献 affinity=0 中性”只在未归一化乘子上成立。** 同文献簇乘以大于1的因子后，归一化会降低其他跨文献簇的相对后验，并可能把它们压到阈值以下。因此该先验会实质偏向同篇归组，不能同时声称跨篇召回不受影响。需限制affinity取值与lambda范围，并用lambda=0、仅同章/邻章/同篇三个层级、跨篇问题做消融；同名“Methods”不是语义相近证据。UMAP空间中GMM的membership不是经过校准的事实概率，重权公式只能称为结构偏好修正，不能称为已证明正确的贝叶斯后验。

6. **读/路由成本门必须写成可计算、可标定的目标。** RAPTOR [父节点构造](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L66)对每簇直接摘要，没有成本收益门。只比较成员总token与摘要token能证明压缩，不能证明导航正确或查询总成本更低。建议把生成前的输入预算/最小有效成员/重复成员集检查，与生成后的压缩率/来源覆盖/忠实性检查分开。离线构建成本与在线每次读取成本要分别记录；若做摊销需显式查询次数假设。门、权重和阈值均属新方案，不能直接归因于RAPTOR或笼统归因于PageIndex。

7. **门拒绝增加父节点时必须保持可达，并给出终止证明。** 未归组、未通过成本门的节点继续作为frontier/root；不能从下一轮输入消失。每个叶必须能从某个当前根沿children到达，所有节点仍可作为全层ANN入口。只有新父引用旧一轮节点，保证rank严格增长且无环；每轮检查是否新增有效摘要、frontier是否收缩、是否重复成员集合，并设层数/递归/token硬上限。单成员长文本不能靠重复生成等长摘要永远向上堆层。原RAPTOR[递归超预算簇](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L162)没有这些保障；新设计必须补齐。

8. **一个IndexModel很合适，但摘要必须是可追溯的有类型派生产物。** [BaseSummarizationModel](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/SummarizationModels.py#L11)足以接同一endpoint。摘要输入要带成员ID和来源边界，缓存签名包括成员内容版本、prompt、模型、tokenizer与预算；不能只按成员文本哈希忽略角色模板。返回异常、空摘要或截断时不发布该父节点，而保留原成员并明确构建状态。最终事实证据仍回到原文block；children集合只证明输入来源，不证明摘要每句话正确。

9. **统一walker要保留collapsed优势，而不是强制从根出发。** 全层ANN与原RAPTOR默认[collapsed retrieval](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L158)一致；`expand/read/neighbors/parents`的agent策略则是新增检索算法。给walker定义统一frontier、已读集合、最大模型调用/节点读取/上下文token预算，并对相同canonical证据只计一次。tree返回一个排名与读取trace；不能把每个跳步、摘要节点或根到叶路径各算一票。独立vector路可限定原文叶，而tree全层候选进入同一个walker，二者共用同一Zvec设施即可。

10. **单SQLite正文契约可行，但还需要版本与一致性设计。** 原文block正文保存一次，节点引用block/span；摘要节点保存新增摘要文本，正文哈希、次序、页码等源位置字段留在主库。FTS external-content索引同一主表；Zvec只存node_id、向量及索引版本，避免再存正文。不要为实现上游[整树pickle](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L301)而恢复另一份树库。SQLite提交与Zvec更新不是同一事务，要有可恢复watermark和新旧版本过滤；拒绝把旧向量命中已变更正文当成有效证据。

11. **全局聚类更新不是局部树编辑。** 新文献可能改变UMAP、BIC簇数、全局与局部GMM、结构画像乃至成本门结果。上游[add_documents](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L204)并没有可用的增量方法。确定性分区、按批重建、簇原型在线分配或周期全量重聚类都可以提出，但均有不同算法语义；不能承诺“只修受影响祖先且与全量重建等价”。10k规模前需定出更新协议和漂移指标。

12. **名称上的统一不等于算法收益已成立，验收要拆开因果因素。** 至少比较：原RAPTOR可复现基线、统一存储但lambda=0且无成本门、仅结构先验、仅成本门、完整统一tree；再比较三路融合。保持相同原文单元、embedding、IndexModel和查询预算，分别测跨篇与单篇证据召回、精确来源、摘要支持性、P95查询时延、构建token和存储。重权公式、成本门、统一walker、增量协议都应明确标成原创设计假设；“原创组合”不等于已经证明学术新颖性，也不等于当前已经优于两个上游。

以上设计收敛为一个层次DAG是实质变化，支持继续细化；在第2—7项定义补齐前，不建议直接把公式写成实现规范或据此开始全量建树。
