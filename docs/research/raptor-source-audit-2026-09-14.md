# RAPTOR 源码审阅：联合 tree 路的可复用能力与边界

审阅日期：2026-09-14。用途：为 DrBrain 的 `bm25 / vector / tree` 三路设计提供源码依据；本文只审阅 RAPTOR，不代替 PageIndex 的独立审阅，也不宣称组合方案已经实现或效果已经验证。

固定版本：[parthsarthi03/raptor@7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767](https://github.com/parthsarthi03/raptor/tree/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767)，提交时间 2024-09-03，审阅 checkout 工作区干净。覆盖 `raptor/` 全部 13 个 Python 模块（约 2.1k 行）、README、requirements、LICENSE 和 demo notebook 全部代码单元。未运行上游代码、未安装其依赖、未反序列化 `demo/cinderella`，未重跑论文实验。

## 1. 结论

RAPTOR 的核心是**对语义相关的多个节点做聚类摘要，再对摘要重新嵌入，递归建立多粒度表示**。它按内容相近程度组织证据，不要求成员在原文中相邻。软聚类允许一个节点进入多个父簇，所以实际数据结构是分层 DAG，可能有多个根，并非严格单父树。源码的默认查询是 collapsed retrieval：直接在所有层的原文节点和摘要节点中按向量距离检索；逐层下钻只是另一种可选查询策略。[聚类实现](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L60)、[默认检索入口](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L252)。

可复用的核心包括全局/局部 UMAP + GMM 软聚类、成员关系、递归摘要、摘要向量、多层候选检索。应复用 DrBrain 已解析的证据单元及已算向量，并由同一 `index_model` 角色承担 PageIndex 建树和 RAPTOR 摘要任务。上游的切块器、pickle 整库保存、默认 OpenAI 包装、原样错误处理和检索预算实现不宜直接作为生产适配层。

## 2. 模块覆盖

以下链接都固定到同一 commit；“已审阅”指静态阅读实际代码，不代表运行验证。

| 模块 | 已审阅实现及作用 |
|---|---|
| [tree_structures.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_structures.py#L4) | `Node`、`Tree`；文本/整数 ID/children 集合/多模型向量及层映射 |
| [cluster_utils.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L23) | 全局与局部降维、BIC 选簇数、GMM 概率阈值、簇超长递归、ClusteringAlgorithm 接口 |
| [cluster_tree_builder.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L17) | 聚类配置、逐层建父节点、摘要及新向量、终止条件、可选线程池 |
| [tree_builder.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_builder.py#L24) | 默认参数、模型验证、叶节点创建、向量与摘要适配、build_from_text、基础近邻方法 |
| [tree_retriever.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L19) | 检索配置、collapsed 全层搜索、逐层搜索、返回层信息、预算与阈值行为 |
| [RetrievalAugmentation.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L18) | 对外配置、建树/检索/QA 编排、对象或 pickle 加载、整树保存 |
| [EmbeddingModels.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/EmbeddingModels.py#L11) | BaseEmbeddingModel、OpenAI ada-002、SBERT 包装 |
| [SummarizationModels.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/SummarizationModels.py#L11) | BaseSummarizationModel、两种 OpenAI 包装、摘要 prompt、异常行为 |
| [QAModels.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/QAModels.py#L15) | BaseQAModel、GPT3/GPT3Turbo/GPT4、UnifiedQA T5；与建树模型分离 |
| [FaissRetriever.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/FaissRetriever.py#L14) | 独立平坦向量基线，既可重新切块也可复用已有 leaf_nodes |
| [Retrievers.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/Retrievers.py#L5) | 最小 `retrieve(query)` 抽象 |
| [utils.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/utils.py#L22) | 切块、节点排序、距离、文本拼接、层反向映射 |
| [__init__.py](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/__init__.py#L2) | 急切导入全部 builder/retriever/model，包括重量级可选后端 |
| [README](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/README.md#L30) / [demo notebook](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/demo.ipynb) | 默认 OpenAI 演示、Cinderella 文本、整树 pickle、自定义 GEMMA 与 SBERT 示例 |
| [requirements](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/requirements.txt#L1) / [LICENSE](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/LICENSE.txt#L1) | 旧版本研究环境依赖、MIT 授权及保留声明要求 |

该固定 checkout 没有 tests 目录、CI workflow、独立 benchmark 脚本、pyproject.toml 或 setup.py；demo 和论文不是自动化回归测试。`demo/cinderella` 是 pickle 格式示例，本文只检查其文件签名，没有加载。

## 3. 实际建树调用链和默认值

```text
RetrievalAugmentation.add_documents(docs: 实际为一个字符串)
  └─ ClusterTreeBuilder.build_from_text(text)
       ├─ split_text(text, tokenizer, max_tokens=100)
       ├─ create_node(index, chunk, children=set())
       │    └─ 每个配置的 embedding model 计算一次向量
       └─ construct_tree(current_level_nodes, all_tree_nodes, layer_to_nodes)
            ├─ 全局 UMAP → BIC 选 GMM 簇数 → 软分配
            ├─ 各全局簇内局部 UMAP/GMM → 软分配
            ├─ 超过簇输入预算时再次聚类
            ├─ 每簇拼接成员文本 → summarize → create_node(摘要, children)
            └─ 对新摘要节点重复上述步骤
```

上面不是论文伪代码，而是 [build_from_text](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_builder.py#L260) 与 [construct_tree](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L55) 的实际调用关系。

| 参数/行为 | 固定源码默认值与含义 |
|---|---|
| 叶块 token 目标 | 100，tokenizer 为 cl100k_base；切块器并不保证严格上限 |
| 最大聚类层数 | 5；不是必须构造 5 层 |
| 摘要输出 token | builder 配置 100；不要与模型方法直接调用时的默认值混淆 |
| 簇输入预算 | 3500 token，RAPTOR_Clustering 自己默认的 tokenizer；未自动继承 builder tokenizer |
| UMAP 维数 | 10；全局实际取 `min(dim, N-2)` |
| 全局 UMAP neighbors | `int(sqrt(N-1))` |
| 局部 UMAP neighbors | 10；全局簇大小 ≤ dim+1 时不做局部 UMAP/GMM，整体成为一个局部簇 |
| GMM 簇数选择 | BIC 最小者，候选 `1 .. min(50,N)-1`，上界不含 50/N |
| 软聚类阈值 | 严格 `probability > 0.1`；不是最近一个簇 |
| 停止 | 当前节点数 ≤ `reduction_dimension+1` 即停止，默认 ≤11；可保留多个根，小文本可只有叶层 |
| 摘要建树并发 | 正常 `build_from_text` 仅叶向量默认并发；调用 construct_tree 未透传该开关，而子类默认 False，因此簇摘要默认串行 |

来源：[builder 默认参数](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_builder.py#L38)、[聚类默认参数与递归](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L132)、[停止与并发](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L87)。

两组“threshold”不可混为一谈：`TreeBuilderConfig.threshold=0.5` 属于基础近邻方法；ClusterTreeBuilder 真正调用 RAPTOR_Clustering 的概率阈值来自 `clustering_params`，默认为 0.1。树构造的 top_k/selection_mode 也不是 GMM 的簇成员参数。

## 4. 检索的真实行为

### Collapsed：默认查询不是从根逐层下钻

`TreeRetriever.retrieve(... top_k=10, max_tokens=3500, collapse_tree=True)` 对 `tree.all_nodes` 全层节点计算 query 向量与节点向量的余弦距离，升序选择。只看前 top_k，累计文本 token；遇到第一个放不进预算的节点就 `break`，不会跳过它继续找后面较短的节点。结果可同时包含原文叶子、局部摘要和高层摘要；没有祖先/后代冗余消除。[实现](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L158)。

这是全量 Python/SciPy 距离计算加排序，不是 ANN。仓库另有 FAISS 基线类，但默认 TreeRetriever 没有调用它。将全层向量放入 DrBrain 的 Zvec 是可行的检索加速适配，不能声称上游已经提供这项集成。

### Traversal：每层选候选并展开成员

从 start_layer（默认最高层）出发，每层计算相似度，取配置的 top_k（默认5）或 threshold 匹配节点，把选中节点加入上下文，再合并其 children 并去重，继续下一层。它返回各层选中节点的合集，不只返回最终叶子。[实现](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L197)。

源码行为有两个容易遗漏的差异：此路径没有执行 `retrieve(max_tokens=...)` 的预算；此路径使用 `self.top_k`，所以调用 `retrieve(top_k=100, collapse_tree=False)` 不会把每层宽度改成100。threshold 分支取的是 **cosine distance > threshold**，会偏向更远而非更近的节点。适配必须修正这些行为并保留可核验的策略名称。

`return_layer_information=True` 只返回局部 node_index 和 layer_number，不返回原文位置、检索分数、成员权重、引用证据或实际读取轨迹。[返回值](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L313)。

### FaissRetriever 只是独立平坦基线

它支持 `build_from_leaf_nodes(leaf_nodes)` 复用叶向量，使用 `IndexFlatIP`，自身没有 normalize 操作，所以不能对任意 embedding provider 假定它等价于余弦。非 top_k 分支先加入文本再检查预算，可能越界；没有过滤 FAISS 返回的无效 -1 索引，候选数量大于库大小时有重复末项的风险。[构建](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/FaissRetriever.py#L128)、[查询](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/FaissRetriever.py#L166)。

## 5. 直接复用解析产物、向量及 index_model

不必调用会重新切块的 `add_documents` / `build_from_text`。已有的源码接口如下：

```python
Node(text: str, index: int, children: Set[int], embeddings)
ClusterTreeBuilder.construct_tree(
    current_level_nodes: Dict[int, Node],
    all_tree_nodes: Dict[int, Node],
    layer_to_nodes: Dict[int, List[Node]],
    use_multithreading: bool = False,
) -> Dict[int, Node]
BaseSummarizationModel.summarize(context, max_tokens=150)
BaseEmbeddingModel.create_embedding(text)
```

最小适配流程：

1. 从 canonical 证据单元读取原文视图及现有向量，用稠密整数 ID 构造临时 Node 映射；原有 `paper_id/node_id/source span/content_hash` 保留在 DrBrain，不交给上游整数 ID 代管。
2. 把同一叶集合传入 `current_level_nodes` 与初始 `all_tree_nodes`，建立 `layer_to_nodes[0]`，直接调用 construct_tree。注意下一 ID 来自 `len(all_tree_nodes)`，所以不应直接使用稀疏整数或哈希整数作为上游 ID。
3. 用自定义 summarizer 将 `summarize` 转发到现有 OpenAI-compatible 客户端及 `index_model` 角色；用 embedding adapter 对接 BGE。父摘要是新文本，只计算这些新增文本的向量。
4. 返回后持久化新增摘要和成员关系，并把临时整数映射回 canonical ID；不要把上游 Tree 对象另存成第二份正文库。

这条路线利用现有 `construct_tree` 的参数边界，但生产实现应增加 checkpoint、输入校验、事务与失败处理，并修复下节问题。源码本身没有公开 `build_from_nodes` 的完整稳定外观接口；可在 DrBrain 中补一个受测适配器，不能把上面的内部调用假称为上游现成生产 API。

**同一 index_model 完全可行，但角色共享不等于输入预算、prompt 和缓存键相同。** PageIndex 结构判定、章节摘要、RAPTOR 多成员摘要应分别指定任务模板和输出契约。簇可输入量要从实际服务 context window 减去 system/prompt/schema/输出预留后计算，不能照抄3500；模型 tokenizer 与预算也必须一致。模型名称、权重/服务版本、prompt 版本、成员内容哈希和摘要参数必须进入缓存签名。来源边界来自 [summarization ABC](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/SummarizationModels.py#L11) 与 [embedding ABC](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/EmbeddingModels.py#L11)，上述工程要求是 DrBrain 适配建议。

建树摘要和 QA 模型在上游本来就是分开的。RAPTOR 纯检索只调用 embedding model；最终回答由独立 qa_model 执行。因此本项目让 Spark 4B 承担 index_model，让 DeepSeek 承担在线推理/回答，不要求4B承担聊天，也不要求 DeepSeek 参与离线摘要。

## 6. 必须处理的实现风险

以下均为源码可定位的静态发现；未跑测试复现，不能当作该环境发生过的故障记录。

| 发现 | 源码依据 | 联合 tree 的工程约束 |
|---|---|---|
| 软阈值可能使某节点没有任何 label | [GMM_cluster L60](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L60) 仅保留概率大于阈值者；后续只遍历出现的 label | 校验所有输入节点至少有一个父成员关系；空分配要显式处理。例如11个近乎均匀分量均可能低于0.1。原叶仍在 all_nodes，但会失去根到叶的路径，不能称为原文被删除 |
| 超预算簇递归无终止保障 | [L162](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L162) singleton 直接保留；[L172](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L172) 重聚类无深度上限、重复成员集或严格缩小检查 | 设置可证明的缩小/退出条件、记录退化原因；单个长原文使用边界保真的子片段，不可静默截断 |
| 递归丢配置 | 同上 L178 只传 nodes/model/max_length | tokenizer、dim、threshold、随机种子必须全程传递 |
| 并不完全可复现 | [L23](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L23) UMAP 无 random_state；Python random.seed224 不控制它；BIC拟合seed224与最终GMM默认seed0不同 | 固定所有随机源、排序、算法版本，记录版本签名；不要只记录一个 random.seed |
| 聚类重映射按浮点向量相等匹配 | [L107](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_utils.py#L107) 广播比较局部簇向量与全部输入向量 | 用输入索引传播成员关系，避免重复向量身份歧义和大规模广播内存开销 |
| GMM 概率没有保留下来 | GMM 返回概率阈值后的 label，最终只返回 Node 列表 | 若新算法使用成员权重，必须明确这是新增产物，不能假称上游已持久化该权重 |
| 摘要异常被当作返回值 | [SummarizationModels L22](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/SummarizationModels.py#L22) except 打印并 return 异常对象，外层 tenacity 不会因返回值重试 | adapter 必须返回经验证的非空字符串或结构化失败；检查截断、预算、finish_reason，拒绝把异常写成摘要 |
| 可选多线程摘要吞任务异常 | [cluster_tree_builder L114](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L114) submit 后未保存 future、未 result；正常串行会在 tokenizer.encode(exception) 等位置次生失败 | 收集每个任务结果，验证期望父簇数，成功后才提交层；不能发布部分层假装完整 |
| builder 最大层数是可变实例状态 | [L95](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/cluster_tree_builder.py#L95) 早停改写 self.num_layers | 分离 configured_max_layers 和 actual_layers，避免同一 builder 后续建树继承前次的小层数 |
| 切块破坏原文与顺序边界 | [utils L38](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/utils.py#L38) 丢弃标点分隔符；长句子分支未先冲刷此前 current_chunk，可能先输出后来的子句；更长无分隔子句仍可超过 max_tokens | 不复用这个切块器处理物理公式/表格/页码；直接复用已解析的 exact-span 片段 |
| 原文格式与来源丢失 | [get_text L181](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/utils.py#L181) 展平换行；Node 仅 text/index/children/embeddings | 正文保留 canonical 文本；用于摘要的视图可格式化但需持有来源映射；摘要只能是派生产物 |
| 阈值方向和预算语义不一致 | [tree_retriever L226](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L226) 与 [L302](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/tree_retriever.py#L302) | 新统一检索契约需绑定 score 类型、候选上限和 token 上限，不照搬错误比较 |
| 没有原生增量维护 | [add_documents L204](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L204) 已有树时交互提示；提到的 add_to_existing 只有注释，没有实现 | 增删更新、成员哈希、依赖失效和世代发布属于新设计；不能承诺只改一个叶子就保持全局GMM结果等价 |
| 没有生产持久化协议 | [加载 L174](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L174)、[保存 L301](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/RetrievalAugmentation.py#L301)：整树pickle，无schema/version/checkpoint/内容哈希/原子替换 | 只复用算法，在现有主库保存派生节点与关系；不新增 pickle 数据源 |

默认摘要 prompt 是通用“保留尽量多关键细节”的摘要请求，没有逐陈述引用、事实核验、公式保护或摘要质量门。RAPTOR 的 children 集合支持追到输入成员，但**不等于已经验证每一句摘要由哪段原文支持**。联合检索最终输出原文证据时，必须再定位和核验片段，摘要不能冒充原文引用。

## 7. 模型、依赖和许可证

源码已使用 OpenAI Python SDK 的 `OpenAI()` 客户端；requirements 固定的是 `openai==1.3.3`，不是旧的0.28接口。其他依赖包含 numpy1.26.3、sentence-transformers2.2.2、transformers4.38.1、tiktoken0.5.1、umap-learn0.5.5、FAISS、torch等；这些是研究仓库环境版本，不是上述最小抽象接口的必要依赖。[requirements](https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/requirements.txt#L1)。

不要直接把它的 requirements 覆盖到 DrBrain。顶层 `import raptor` 急切导入 FAISS、torch/T5、sentence-transformers 等，甚至只想使用数据结构也会拖入这些依赖。若作为 submodule 保留原源码，应由窄适配层加载和隔离需要的算法，配合依赖分层；若移植部分代码，则保留上游 MIT 声明和固定版本来源。如何加载模块需在实现时测试，本文没有验证某种动态导入技巧。

默认 OpenAI 摘要包装在每次调用时自行创建客户端，不能直接注入独立角色 endpoint/api_key；因此应实现自己的 BaseSummarizationModel adapter，而不是通过修改全局 OPENAI_API_KEY/BASE_URL 让不同角色互相覆盖。QA 的 GPT3Turbo/GPT4 类在构造时读取 OPENAI_API_KEY，最终响应未给出证据结构；DrBrain 无需采用这层回答包装。

## 8. 论文、示例和代码的区别

以下论文核对只依据 [RAPTOR v1，方法第3节](https://arxiv.org/html/2401.18059v1#S3)，不把论文实验结论推成当前4B/BGE/物理语料效果：

- 论文实验用 SBERT 作为嵌入，代码默认是 OpenAI ada-002；SBERT 在代码中是可选实现。
- 论文描述 collapsed 在其对照实验中优于固定宽度 traversal，选2000-token条件；代码默认3500 token、top10。不能把“先根再叶”宣称为 RAPTOR 唯一或默认策略。
- 论文强调按句保持语义的100-token片段；代码实际用正则删除标点、对子句继续切分，不保证 exact span 或严格上限。
- 论文的摘要可靠性观察来自其模型和样本；代码没有相应在线事实验证器，不能据此免做本项目的4B摘要验收。

demo notebook 展示了自定义 GEMMA summarizer/QA 与 SBERT 的接口用法，但不是生产适配模板：其摘要示例直接返回 text-generation 的 generated_text，未像QA示例那样截去 prompt，可能把输入提示也当作摘要返回。是否触发取决于对应 pipeline 返回行为；未运行验证。

## 9. 给 PageIndex + RAPTOR 联合 tree 设计的边界建议

这些是依据上述源码提出的适配约束，属于新方案，未作效果验证：

1. **共用原文证据、向量和 index_model，保留不同关系语义。** PageIndex 的文档包含关系与 RAPTOR 的语义成员关系不能混成一个不分类型的 parent_id。语义层允许多父，根节点允许多个。
2. **章节是来源定位锚点，不必等于聚类最小单位。** 先保留原文层的边界，再按模型预算使用章节内 exact-span 片段；整章过长时，不应为了复用章节而让 embedding/summarizer 静默截断。建立片段到章节的映射即可复用结构。
3. **摘要复用有条件。** 只有输入成员、成员版本、顺序约定、任务语义、prompt/model/token预算相同，才可复用一份摘要；章节概括和跨章节聚类概括不能因“都是摘要”而互换。已有摘要可作候选输入或导航描述，需记录额外压缩造成的信息损失。
4. **语义簇候选必须可直接由整库产生。** 如果依旧先用BM25筛出文献再检索摘要，tree就不是独立召回。可以在tree内部使用全层摘要向量候选，再利用PageIndex结构/推理定位原文；这是组合适配，不是原仓库已有调用链。
5. **不要将默认collapsed能力隐藏。** 联合tree应保留跨层候选入口，必要时再下钻；强制只从根出发会丢掉RAPTOR的可变粒度优势。与单独vector路共享embedding计算和ANN设施，不代表两路检索语义相同。
6. **跨篇聚类是合理扩展，但范围需显式。** 原接口只是一个文本字符串，没有文献隔离或文献ID。算法可以对传入节点集合聚类，却没有现成的多文献权限、更新、删除和分片系统。跨篇成员关系必须带文献与访问范围；不能用原接口的 `add_documents` 名字推断已经解决语料库管理。
7. **不要承诺任意增量等价。** UMAP、BIC和GMM依赖输入集合，增删节点可能改变全局分布。可引入确定性分区与定期重建，但分区策略、召回损失和簇漂移属于需要消融的新增算法决定。
8. **tree应输出证据候选，不输出一次最终答案再混分。** 原摘要和来源成员可用于定位，最终返回共享 node_id/span/hash，之后进入三路RRF和BGE重排。在线推理读取的结构/摘要/原文都应记入trace，分别计入耗时和模型token。

## 10. 联合实现前后的最小验收项

- 同一原文ID在两种关系中共享，向量缓存命中时不重复计算；新增聚类摘要才调用index_model和embedding。
- 软多父、多个根、≤11叶的小输入、空label、全重复向量、超预算单成员、重聚类不缩小、摘要空响应/异常/截断，都有明确且可测试的行为。
- 所有可见叶证据在结构层可达；语义层coverage不满足时不得发布“完整”状态；部分任务失败不可漏节点后继续报成功。
- 模型、prompt、tokenizer、成员哈希改变可触发适当重建；未改变的原文和结构无需再次解析。
- 检索预算在collapsed、traversal和组合模式中真实执行；摘要与叶子重复计数受控，来源可回到canonical原文位置。
- 在同一证据库、同一候选/重排/模型预算上比较：vector-only、RAPTOR collapsed、PageIndex-only、联合tree及完整三路。测证据召回与来源准确性、答案可支持性、P50/P95延时、离线模型token和额外存储；不能仅比较答案看起来是否流畅。

本次完成的是固定源码的静态审阅。未核实项包括4B摘要质量、聚类实际耗时/峰值内存、指定版本依赖在当前环境的运行兼容、全部风险触发条件、10k跨文献聚类召回及组合算法的收益。这些须通过后续CLI验收和对照实验给出证据。
