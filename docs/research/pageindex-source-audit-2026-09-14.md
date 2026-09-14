# PageIndex 源码审阅：作为共享 tree 检索路的接入边界

审阅日期：2026-09-14。审阅对象固定为 VectifyAI/PageIndex 提交 [`bfbd4b305cd3f0f39a5094779627a2c93634ad79`](https://github.com/VectifyAI/PageIndex/tree/bfbd4b305cd3f0f39a5094779627a2c93634ad79)，包声明版本 0.2.10。这是源码设计审阅，未运行上游代码、安装依赖或测量生产效果；不能把测试文件存在等同于测试通过。结论不依赖本机已安装 SDK 的版本。

## 结论

PageIndex 适合承担联合 tree 路的**文献结构生成、结构导航和按原文范围阅读**。其公开本地实现确实包含多文献工具及完整问答 agent，不只有旧版建树脚本；但**语料级 File System/语义发现仍是云端能力**。本地 `browse_documents` 不支持 query/relevance，按时间枚举文件；默认 agent 指令甚至要求宣布未找到前翻完全部文库。这是 RAPTOR 语义层可以补充的具体缺口，不应把云端功能想象成已经公开的本地算法。[README](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/README.md#L178)、[本地 browse](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L723)、[发现策略](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L1587)。

建议复用其公开 builder 和 tools adapter，在 DrBrain 的 canonical 证据存储上提供读取接口；联合检索的工具、范围控制、预算和 evidence 输出由 DrBrain 编排。**不能把 RAPTOR 摘要塞入 PageIndex 的连续页码树，然后声称原生兼容。** 章节父子边表达文献包含关系；聚类成员边表达语义集合关系，必须分别标记。

## 审阅覆盖

| 源码族 | 深度与用途 |
|---|---|
| `client.py`, `local_api.py`, `local_store.py` | 深读构造、model/backend 路由、submit、存储、读取、scope、citation、agent 配置接口 |
| `page_index_classic.py` | 函数全表和主调用链；深读 TOC 检测/校验/修复/递归分节、`page_list` 接口 |
| `page_index_md.py` | 深读 Markdown heading、line provenance、thinning、tree、摘要和公开入口 |
| `flash/api.py`, `flash/main.py` | 深读公开参数、拒绝条件、分阶段布局流水线、文本丢弃/复用点 |
| `tree_optimize.py`, `utils.py` | 深读 merge/expand 成本、验证、ID 重编、两种摘要策略、LLM retry/backend ContextVar |
| `agent_tools.py`, `local_chat.py` | 深读本地工具、文库发现、scope、分页、工具轨迹、状态/异常和可组合边界 |
| `integrations/*`, `mcp_bridge.py`, `cloud_api.py` | 审阅 adapter 与传输调用链、cloud/local 区分；没有推断云端服务内部实现 |
| `types.py`, `errors.py`, `chat_stream.py`, `__init__.py`, config、依赖、LICENSE | 包装契约、配置、stream 单次消费、MIT 声明 |
| Flash 私有布局模块和 data | 全仓文件清单及入口依赖追踪；深入其 pipeline 而非把字符/字体几何启发式宣称逐条运行验证 |
| 8 个 `tests/test_*.py`、README、agent demo | 静态检查测试函数与关键断言/调用；没有执行网络测试或复现 benchmark |

上游 Git 树共 168 个 blob（包含 PDF、图片和示例产物）。核心库的 Flash 子模块较大，模块族包括：字符/PDF 对象解析与 Unicode/CMap 修复，span/line/block 模型，聚行与聚块，分栏及 gutters，统计与 token/style 哈希，页眉/页脚/水印/TOC 分类，caption 区域，标题和 heading 候选，多候选 outline assembly、过滤/序列化，以及 embedded bookmarks 合并。它们服务于**页面几何恢复及目录检测**，不提供跨文献语义聚类。[完整固定文件树](https://api.github.com/repos/VectifyAI/PageIndex/git/trees/bfbd4b305cd3f0f39a5094779627a2c93634ad79?recursive=1)、[布局主入口](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/flash/main.py#L124)。

## 构建：三套路径的能力不能混为一谈

### Classic

主链为 `page_index_main → tree_parser → check_toc/meta_processor → verify_toc → fix_incorrect_toc_with_retries → post_processing → process_large_node_recursively → merge_tree → IDs/文本/摘要`。有目录页码时利用目录并校准物理页；校验不佳时依次尝试无页码目录、直接从页文本推导目录，最后明确报错。校验正确率大于 0.6 的错误项最多修复三轮。递归拆分要求页跨度与 token 数同时超过阈值（默认配置 10 页、20,000 tokens），不是所有节点强制均匀切片。[主链](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_classic.py#L1126)、[配置](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/config.yaml#L9)。

可直接复用的接口：`page_index_main(doc, opt, logger, page_list=[(page_text, token_count), ...])`。传入 `page_list` 会绕过 `get_page_tokens`，可以使用已经解析的 PDF 页文本。入口仍要求真实 PDF 路径或 `BytesIO`；`tree_parser` 的当前实现只消费传入页面，`doc` 未用于重读正文。应传原始 PDF 和 DrBrain 已有的真实 page map；不应为 Markdown 人为造 PDF。[接口](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_classic.py#L1233)。

Classic 的摘要使用 `generate_summaries_for_structure`，将每个节点的页文本交给 LLM；父子页范围可能重叠，重复输入由调用方负担。它不是 Flash 的 bottom-up 摘要实现。[Classic 摘要调用](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_classic.py#L1250)、[摘要辅助函数](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/utils.py#L731)。

### Markdown

`md_to_tree` 是真实公开入口，直接处理 Markdown，不需要 LLM 建立已有标题的层级。识别 H1–H6 和独占行的粗体标题，跳过三反引号 code fence 内标题；每个节点记录 1-based `line_num`。正文先按相邻 heading 切分，再用层级 stack 生成树。没有标题时返回空结构，没有通用的无标题材料建树保证。原生输出不含完整 char end/line end；这些需要由 canonical 文本和下一边界补足并校验。[提取](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_md.py#L32)、[树与入口](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_md.py#L192)。

MD 的节点文本是自身 heading 到下一个 heading 的前缀内容。摘要代码对叶写 `summary`、对有子节点者写 `prefix_summary`，不等同于整个父节点子树概括；thinning 会改变节点集合及文本范围。不能把现有任意 `summary` 当作相同语义、相同 coverage 的缓存。[摘要语义](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/page_index_md.py#L10)。

### Flash

`page_index_flash` 从 PDF 布局统计建目录：PDFium 字符与页面 viewport → 行/分栏/块和阅读顺序 → 文档统计 → 页眉页脚/水印/TOC 标注 → 标题、caption 和 heading 候选 → outline assembly → bookmarks 校验合并。核心布局提取不需要 LLM；默认的 optimize/summary **需要** LLM。`summary=False, optimize=False` 才是完全无 LLM 路径。[公开参数](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/flash/api.py#L140)、[阶段顺序](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/flash/main.py#L124)。

公开接口仅收 PDF path/`BytesIO`，**没有接受任意 parser blocks 的接口**。内部 `extract_toc` 保留 `page_texts`，而公开 `page_index_flash` 在摘要后将其 pop 掉。若复用 Flash 解析结果，最小 upstream adaptation 是把解析阶段的规范化产物显式暴露，再复用 `optimize`/`summarize_tree`；不能假定 pdf-inspector 的块结构直接符合其字体/span/geometry 内部模型。

没有目录时以每页一个节点表示；超过 10 个这样的平坦节点，本地 CLI/SDK 拒绝并提示 standard。没有文字层时报告 unreadable，要求 OCR；公开本地实现不承诺 OCR、公式图片理解或 block-level citation。Flash 内部解析器可以多进程，但默认 worker 与 LLM concurrency 不能直接照搬到 DrBrain 多篇并发作业，以免并发相乘。[拒绝策略](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/flash/api.py#L120)、[worker 接口](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/flash/main.py#L124)。

## 已有优化与摘要算法

PageIndex 当前源码已经包含有明确定义的结构优化。令 `S(v)` 为折叠后线性扫描的页数，`R(v)=1` 为以页为单位的 routing cost，`S_residual(v)` 为未被任何子节点覆盖的页数：

- 已展开节点代价：`R + max(S_residual, max(tree_cost(child)))`。
- 自底向上，当 `S(v) <= tree_cost(v)` 时 merge；被合并的标题保存在 `key_items`。
- 叶页跨度大于 5 时才尝试 expand；临时子节点按折叠状态评估，只有代价严格降低才接受。候选标题必须实际存在于所标页，范围和顺序必须有效。
- 同页 sibling 会合并，因为原生读取单位是页；默认 expand concurrency 32、最多 3 轮，已作决定节点冻结以避免反复 merge/expand。
- 最后顺序重编 `node_id`，返回 old→new mapping；这不是内容稳定 ID。

这些成本和正确性条件依赖**连续页范围**，不能直接拿去优化跨篇且可重叠的语义集合。新 tree 可借鉴预算优化思想，但对语义分支重新定义证据读取/模型 token 成本，并作为新增算法验证。[成本定义](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/tree_optimize.py#L1)、[代价计算](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/tree_optimize.py#L231)、[候选验证](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/tree_optimize.py#L625)。

Flash 所用 `utils.summarize_tree` 已提供 bottom-up 汇总：叶由自己的原文页摘要；父由子摘要和开头原文生成整体摘要；小于 200 tokens 的叶直接复用原文；有非空 summary 时跳过；默认 concurrency 64。这是一处很好的**共享 index-model 摘要工作器**接入点，但该实现仍基于页范围，未携带成员 hash/prompt/model revision；调用方必须保证跳过的缓存没有过期。父开头文本默认最多 3 页，也不是所有 residual 原文的无损聚合。[bottom-up 实现](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/utils.py#L760)。

## 存储与 provenance

`LocalAPI.submit_document` 每次先用 PyPDF2 提取所有页；standard 随后复用提取的页，flash 则再走 PDFium 布局解析。提交结束分配新 `pi-<uuid>`，并以 basename 加序号消解重名；没有基于内容哈希的幂等判断。存储为：`docs/<id>/tree.json`、`pages.json`、`doc.json` 和根 `manifest.json`。**原生 DocStore 不复制 PDF**，但保存的完整 pages 与 DrBrain 的规范正文重复；另存树和元数据也重复。[submit](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_api.py#L93)、[store](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_store.py#L73)。

写文件是临时文件→fsync→rename，Linux 文件锁保护名称检查与保存，manifest 是可恢复缓存。它不是跨 DB/多文件事务，也没有 artifact revision、增量 invalidation、稳定证据 hash、跨文件 lineage。tree 移除 inline text，读取时按 pages 补全，但可能把父子覆盖页反复组合。我们应复用 builder/tool protocol，而不要求 DrBrain 镜像这个 DocStore。

PDF `start_index/end_index` 为 1-based inclusive 物理页。章节可以共享边界页，合并/展开可能重编 ID；MD 的 `line_num` 是另一种 locator。共享 evidence 应以 `(document_revision, canonical_start, canonical_end)` 和文本 hash 标识，同时保留可用的原始页/行映射；跨文献摘要只带成员关系，不能伪造连续页跨度。PageIndex 自身 node_id 只能作为一版结构内的别名。

## 真正的检索和问答循环

公开本地只读工具为 `browse_documents`、`get_document`、`get_document_structure`、`get_page_content`；默认不暴露删除。agent 通过 OpenAI Agents Runner 反复发起工具调用、读取结果，再选下一步，不是一次对目录做静态关键词匹配。默认指令对大于 20 页的文档先读结构再读页；小文档可以直接读页。`submit_query/get_retrieval` 是旧 cloud 接口，不能当本地 retrieval API。[工具和指令](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L1587)、[Runner](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_chat.py#L939)。

现成工具边界：

- 文库发现最多每页 50 条；local query/relevance 与非 root folder 被明确拒绝。
- 原生工具按 **doc_name** 定位，每次 `_all_documents` 最多 100 条一页枚举，重复名字取最新。DrBrain 必须生成可唯一映射稳定 ID 的显示名，不能把全部文件都叫 `source.pdf`。
- `get_document_structure` 去掉正文、按约 95,000 字符预算分段；这不是严格的模型 token 预算。
- `get_page_content` 内部先 `get_ocr(format='page')` 读取完整页面列表，再选请求页，返回 `{doc_name,total_pages,requested_pages,returned_pages,content:[{page,text}]}`。它没有原生 read-range provider；大篇文献需 canonical 索引/缓存或新增严格范围读取工具。
- local `doc_ids` 在每个工具查找处强制 scope；cloud own-model scope 仅用于 prompt targeting，不能把两者当同样 ACL。

[名称查找](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L337)、[页工具](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L1009)、[scope 差别](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/client.py#L1830)。

### 可注入边界与轨迹

`PageIndexClient` 没有公开 `store=` 参数；`LocalAPI` 构造里硬编码 `DocStore`。`local_chat._openai_agent` 固定工具集，且 `extra_body.tools` 明确拒绝。因此不要 monkeypatch 私有 `_store` 或向 chat 的 extra_body 塞新工具。

适配入口是公开 `agent_tools()/as_openai_tools()` 以及 `integrations.openai_agents.build_openai_tools(client, doc_ids=...)`。后者没有 `PageIndexClient` 类型检查，以 duck typing 读取 client。`api_key` 为空时走本地工具，其只读最小方法如下：

| 方法 | 返回契约 |
|---|---|
| `list_documents(limit, offset)` | `{documents:[{id,name,status,createdAt,pageNum,description,metadata}], total}` |
| `get_document(id)` | 上述单篇 metadata |
| `get_tree(id,node_summary=True,include_text=False)` | `{result: structure}`；`_api.raw_tree` 只是可选优化 |
| `get_ocr(id,format='page')` | `{result:[{page_index:int,markdown:str}]}` |

关闭管理工具不需要 `delete_document`。可以提供 DrBrain `CanonicalDocumentClient` 读取已有数据，再使用上游工具转换函数；契约测试必须证实 duck typing 与固定 commit 的工具实现兼容。这是利用源码的明确组合点，不是宣称上游提供稳定的第三方存储插件 API。[工具工厂](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/integrations/openai_agents.py#L33)、[local 工具绑定](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/agent_tools.py#L1522)。

普通 `chat_completions` 返回最终回答和 usage，没有工具轨迹，history 也拒绝 tool 角色。`chat(protocol='responses')` 返回完整 `items` 可供 continuation；不能假定 DeepSeek 提供 Responses endpoint。`chat(stream=True).events()` 则暴露 `tool_call`/`tool_result`，可以观察原生循环，但没有 retrieval-only typed result 约束。联合 tree 应由我们持有 Runner/状态和有界 frontier，返回经工具读取日志校验的 evidence 列表，再参加 BM25/vector/tree 三路融合，最后只进行一次答案生成。[history 限制](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_chat.py#L50)、[events](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_chat.py#L719)、[Responses items](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/local_chat.py#L1091)。

新版 local 支持通过 `chat(citations=True)` 加页级引用提示；`enable_citations=True` 仍是 managed chat 参数。`resolve_citations` 只解析标签和映射名称，未知文献仍可返回 `doc_id: None`，不是“该原文确实被读取且支持答案”的验证器；没有本地 block bbox。联合检索必须从实际工具结果确认文本 hash/locator，不应只解析模型写出的引用。[引用契约](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/client.py#L2145)。

## 模型角色与资源

上游已有 `index={model,summary_model,backend,storage_path}` 和 `chat={model,backend}` 两侧概念；`index_model` 包括结构修正和摘要，用户用同一个 Spark 4B 作为 PageIndex 和 RAPTOR 摘要模型符合该边界。聚类向量仍由 embed endpoint 生成，query-time 推理和最终答案归 DeepSeek，rerank 独立 BGE。相同 index model 并不意味着不同 coverage 的摘要内容可互换。[角色定义](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/client.py#L259)。

构建辅助函数通过 `utils._llm_backend` ContextVar 把 endpoint 参数传给 LiteLLM；每次 completion 默认最多 10 次尝试，对指定不可重试 HTTP 状态立即失败。部分 prompt 错误可留下空 summary，所有模型调用失败必须报错。上层要有共享并发/timeout/token 限制和每项摘要完成状态，不能仅检查树非空。[LLM helpers](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/utils.py#L145)。

尤其注意：`openai_agent_config()` **不会把 `chat_backend` 放进返回的配置**，文档明确要求调用者自己提供运行环境的模型认证。组装联合 Runner 时应显式构造绑定 DeepSeek base_url/api_key 的模型实例；不能只传一个模型名，依赖其他角色的全局变量。这是此前 credential 错误最容易复发的边界。[源码说明](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/pageindex/client.py#L1840)。

## 与 RAPTOR 共享 tree 的设计约束

以下是从源码接入边界推导的 DrBrain 新设计要求，不是声称 PageIndex 已经实现：

1. 原文证据先形成稳定、可校验范围；PageIndex 结构和 RAPTOR 聚类都引用它。父子共享页不能重复当作多个独立文本训练/聚类输入，章节开头 residual 不能遗漏。
2. 结构节点保留文献顺序、父子关系与真实页/行 locator；语义节点保留成员集、层级和来源。允许语义节点有重叠成员，禁止伪造它覆盖一段连续页面。
3. 共享 index model、工作队列、摘要缓存与 embedding；摘要 identity 包含成员/范围 hash、用途、prompt/model revision，只有这些相同时复用。
4. 联合 tree 内的语义入口可以在当前授权 corpus 的多层摘要上检索；PageIndex 推理继续读取结构和原文。不要把 BM25 候选文献设为 tree 的强制入口，否则它失去独立召回价值。
5. 用新工具显式提供 `search_semantic_nodes`、两类边的 `expand_node`、以及 `read_evidence`。PDF 范围读取可复用原生 page tools；无页码 MD/TeX 要用 line/char locator，不能套假 page。
6. 用我们自己的 retrieval-only 编排替换上游全库时间枚举/用户选文献问答策略；保留官方结构→原文阅读语义和工具失败契约。最终 evidence 才参加三路 RRF/rerank，摘要可以用于路由和概括，精确事实须下钻原文。
7. 不继承上游逐次提交新 UUID 的存储模式；canonical store 负责幂等、版本、权限、依赖失效，PageIndex provider 只负责读取/计算视图。

## 测试证据与未核实事项

静态识别 8 个测试模块共 556 个 `test_` 函数：classic 3、MD 3、Flash 38、client 146、agent tools 158、local chat 183、package surface 11、issue_163 14。参数化产生的 case 数不在此统计。覆盖主题包括配置双写拒绝、local/cloud scope、模型 endpoint、空/坏 PDF、过大 flat tree、部分摘要失败、文件损坏、页读取预算、tool history/stream、取消、协议 round-trip 与引用格式；其中很多是 fake model/monkeypatch 契约测试，不能据此宣称 Spark 或 10k corpus 验证通过。[tests](https://github.com/VectifyAI/PageIndex/tree/bfbd4b305cd3f0f39a5094779627a2c93634ad79/tests)。

待联合方案实现时验证：真实 canonical page-map 的偏移精度；MD/TeX heading 缺失和前言保留；Flash 与现有 parser 双重解析成本；共享摘要对数学公式、否定、数值的保真度；4B 的实际上下文与并发；多层语义选择/结构下钻的 evidence recall 与 tool/token 预算；增量聚类/版本失效；ACL 在摘要成员集和所有读取工具上的一致性；新 retrieval-only agent 是否有停止条件且不会把未读摘要当事实证据。

未读取云端 File System 服务内部代码，不能对其自动文件夹算法作复现声明；未复现 README 的准确率/延迟 benchmark；没有完成对 Flash 每条字体/Unicode/几何启发式的数值验证。

上游 MIT license 要求保留版权和许可文本；固定 submodule 比复制后丢失来源更易追溯。本审阅只讨论技术接入，不把更换 SDK 包名本身当作算法改进。[LICENSE](https://github.com/VectifyAI/PageIndex/blob/bfbd4b305cd3f0f39a5094779627a2c93634ad79/LICENSE)。
