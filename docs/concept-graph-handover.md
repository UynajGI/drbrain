# 概念共现图增强 — 开发交接文档

> 面向在**另一台机器**上继续开发（全量挖掘 / GNN 验证 / 扩展）的协作者。
> 对应计划：`.qoder/specs/Concept_Graph_Replication_task-798.md`
> 复现目标：Marwitz et al. (2026, NMI)《Predicting new research directions in materials science using LLMs and concept graphs》

---

## 1. 这是什么

在 DrBrain 现有「类型化语义知识图谱」之上，新增了一层**概念共现图**（`drbrain cg` 命令组），复现论文方法论：

```
多源语料摄入 → 概念抽取/归一化 → 共现图(clique+年时间戳) → 语义嵌入 → 时序链接预测 → 研究方向推荐
```

**核心边界（务必先读）**：
- **无全文也可建图**：仅凭元数据（标题/摘要/作者/年份/期刊/DOI）+ 引用关系 + 概念抽取 + 外部学术 API 即可建图。
- **不依赖模型微调**：论文微调 LLaMa-2-13B，本项目改用三条无微调路径（见 §6）。
- **与已有解析/RAG 共存不冲突**：全文解析（MinerU/PageIndex）、段落级嵌入、RAG 检索（RAPTOR/ask）是 DrBrain **已有能力**，本轮**未改动**；概念图层为纯增量。

---

## 2. 当前完成状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Phase 0 | 多源语料接入（Sciverse / OpenAlex 适配器、限速重试、unique_id 去重、引用网络、`cg ingest`） | ✅ |
| Phase 1 | 共现图构建（归一化、clique、频率过滤、schema v9、`cg build`） | ✅ |
| Phase 2 | 语义概念嵌入 + UMAP 科学地图（`cg embed` / `neighbors` / `map`） | ✅ |
| Phase 3 | 时序链接预测（拓扑+语义特征、MLP Baseline/Embeddings/混合、ROC/PR@k/d_prev 评估、`cg predict`） | ✅ |
| Phase 3 GNN | 可选 GraphSAGE（纯 PyTorch，懒加载，`--model gnn`） | ✅ 代码就绪，**待在装了 torch 的机器验证** |
| Phase 4 | 研究方向推荐（研究者画像、组合过滤、LLM curation、`cg recommend`） | ✅ |
| Phase 5 | 集成质量（ruff / mypy / pytest 全绿） | ✅ |

**质量门禁（最近一次）**：ruff 全绿 · mypy 16 文件无问题 · `pytest -m "not integration"` **2199 passed, 0 failed** · 9 个 `test_cg_*.py` 文件。

---

## 3. 在新机器上搭建环境

```bash
# 1) 克隆 + 进入分支
git clone git@github.com:UynajGI/DrBrain.git
cd DrBrain
git checkout feat/knowledge-graph-enhancement   # 或合并后的 main

# 2) 安装（uv 管理依赖）
uv sync                      # 基础依赖
uv sync --extra gnn          # 需要 GNN 时（安装 torch）；或 uv pip install torch

# 3) 配置 API token（config.local.yaml 已被 gitignore，勿提交）
cp config.example.yaml config.local.yaml
# 编辑 config.local.yaml 的 api 段：
#   sciverse_token: "<你的 Sciverse token>"   # https://sciverse.opendatalab.com/tokens
#   openalex_token: ""                         # OpenAlex 可无 token 试用
#   sciverse_rate_limit: 30

# 4) 初始化 + 自检
uv run drbrain setup
uv run drbrain check
```

> Sciverse / 点石 DianShi / SeqStudio 共用同一套账号与 API Key。Token 长期有效，**严禁提交 Git**。

---

## 4. 完整使用流程（端到端）

```bash
# ① 摄入语料（OpenAlex 无需 token；--limit 控制规模，增量去重）
uv run drbrain cg ingest --source openalex "graphene battery" --year-from 2018 --year-to 2023 --limit 2000
uv run drbrain cg ingest --source sciverse "perovskite solar cell" --year-from 2018 --limit 2000 --with-citations

# ② 构建共现图（概念归一化 + clique + 频率过滤）
uv run drbrain cg build --source terms --min-freq 3 --min-words 2

# ③ 计算概念语义嵌入（复用已配置 embedding provider，默认 Qwen3-Embedding）
uv run drbrain cg embed            # 加 --context 可用含该概念的论文标题做平均增强

# ④ 导出 UMAP 科学地图（自包含交互式 HTML）
uv run drbrain cg map -o concept_map.html
uv run drbrain cg neighbors "graphene" --top 10

# ⑤ 时序链接预测（特征快照 feat-cutoff，训练/测试窗口分离，无泄漏）
uv run drbrain cg predict --feat-cutoff 2020 --train-end 2021 --test-end 2022 --model mixture --json
uv run drbrain cg predict --feat-cutoff 2020 --train-end 2021 --test-end 2022 --model gnn  # 需 torch

# ⑥ 研究方向推荐（按作者画像）
uv run drbrain cg recommend --author "Brabec" --top 25 --curate -o report.md
```

**概念来源（`cg build --source`）**：
- `terms`（默认）：源端 keywords/topics（Sciverse/OpenAlex 提供，**零 LLM**）。
- `concepts`：复用已有全文解析产出的 `concepts.label`。
- `abstract`：调已配置 LLM API 从标题+摘要抽取（无需微调）。

---

## 5. 大规模挖掘建议（资源相关）

原开发主机为小主机（4 核 / 15G 内存 / 磁盘剩余约 53G），**不适合全量挖掘**（论文 221K 篇 → 137K 节点 / 13M 边）。

- 摄入已做成**增量分批**（`--limit`、内部 `commit_every`），可重复运行、自动去重。
- 全量语料获取与共现图构建请在**服务器/集群**运行；本机仅做小规模验证。
- 大规模时建议：分主题/分年份多次 `cg ingest`；`cg build` 前确保 `concept_cooccurrence` 已覆盖目标语料。
- GNN 在大规模图上应引入**邻居采样**（当前为全邻居均值聚合，适合中小图）——见 §8 待办。

---

## 6. 关键设计决策

1. **不复现微调**：三条无微调概念抽取路径——⓪ 源端 keywords/topics（零 LLM）；① `extract_concepts()` 调 LLM API；③ 复用 `concepts.label`。代价：精度不及微调模型，但可行。
2. **嵌入模型可配置**：默认 `Qwen/Qwen3-Embedding-0.6B`（BERT 族，1024 维）。想要材料领域特化，把 `embed.model` 换成 MatSciBERT 即可，**无需改代码**。论文显示 MatSciBERT 仅比通用 BERT 高 ~3% AUC。
3. **GNN 为可选**：论文最佳为 GNN+Embeddings 混合（AUC 0.9433），但 GNN 相比 MLP 混合仅 +1~2%。已实现为可选纯 PyTorch GraphSAGE（`drbrain[gnn]`），默认不强制安装。
4. **去重主键**：`unique_id` 优先、DOI 二级（Sciverse `unique_id` 任何记录都有，DOI 可能缺失）。
5. **时序无泄漏**：特征只用 `G_{feat-cutoff}`（≤ 快照年的边），正样本为快照后新增边。

---

## 7. 模块与文件清单

```
src/drbrain/concept_graph/
├── sources/
│   ├── base.py        # PaperRecord / PaperRelations / CorpusSource 协议 / 响应信封
│   ├── _http.py       # Sciverse HTTP 客户端（Bearer + 令牌桶限速 + 重试退避）
│   ├── sciverse.py    # SciverseSource（meta-search / meta-paper-relations / meta-catalog）
│   ├── openalex.py    # OpenAlexSource（/works 主题发现 + 分页）
│   └── registry.py    # get_source 工厂（sciverse / openalex，预留 crossref/s2/arxiv）
├── ingest.py          # ingest_corpus（unique_id 去重）/ ingest_citations（引用网络）
├── builder.py         # normalize_concept / concepts_for_paper / build_cliques / apply_filter
├── embeddings.py      # compute_concept_embeddings / aggregate_vectors / nearest_neighbors
├── map.py             # umap_project / export_html（交互式科学地图）
├── features.py        # yearly_subgraph / topo_features(20维) / semantic_features
├── dataset.py         # temporal_pairs（无泄漏切分）/ oversample_indices（30% 过采样）
├── link_predict.py    # MLPLinkClassifier / MixtureEnsemble（3:2 加权）
├── gnn.py             # GNNLinkClassifier（可选 GraphSAGE，torch 懒加载）
├── eval.py            # roc_auc / precision_recall_at_k / stratify_by_dprev
└── recommend.py       # own_concepts / recommend_combinations / llm_curation

CLI: src/drbrain/cli/concept_graph_commands.py  →  drbrain cg {ingest,build,embed,neighbors,map,predict,recommend}
存储: schema v9（concept_nodes / concept_cooccurrence / concept_embeddings / paper_terms / paper_citations / corpus_sources）
测试: tests/test_cg_{http,sources,ingest,cli,builder,embeddings,linkpredict,gnn,recommend}.py
```

---

## 8. 待办 / 可扩展项（交给下一台机器）

- [ ] **GNN 验证**：在装了 torch 的机器跑 `pytest tests/test_cg_gnn.py`，并用真实语料对比 `--model gnn` vs `mixture` 的 AUC。
- [ ] **GNN+Embeddings 混合**：论文最佳组合（1:1），当前 GNN 与嵌入混合尚未接线（可作为 `--model gnn-mixture`）。
- [ ] **邻居采样**：大规模图下把全邻居均值聚合改为 GraphSAGE 邻居采样，降内存/提速。
- [ ] **全量挖掘**：在服务器跑材料科学全量语料（OpenAlex/Sciverse 按期刊列表批量摄入）。
- [ ] **更多数据源**：registry 已预留 crossref / semantic-scholar / arxiv 适配器位。
- [ ] **MatSciBERT 对比实验**：`embed.model` 换 MatSciBERT，评估领域特化增益。
- [ ] **推荐系统评估**：复现论文的专家访谈式人工评估（A1/A2/B/C/D 分类）。

---

## 9. 测试与质量门禁

```bash
uv run ruff check . && uv run ruff format .       # 风格
uv run mypy src/drbrain/concept_graph             # 类型
uv run pytest -m "not integration" -q             # 快速全量（不含慢速集成）
uv run pytest tests/test_cg_gnn.py -q             # GNN（无 torch 自动跳过）
```

---

## 10. 提交历史（feat/knowledge-graph-enhancement）

```
0a2b4c0 feat(concept-graph): 可选 GNN(GraphSAGE) 链接预测 (torch 懒加载, drbrain[gnn])
df639e5 fix(concept-graph): schema v9 兼容 (迁移测试断言 >=8) + ruff E741
6e042d9 feat(concept-graph): 研究方向推荐 (研究者画像/组合过滤/LLM curation + cg recommend CLI)
5445a91 feat(concept-graph): 时序链接预测 (拓扑+语义特征/MLP混合/评估 + cg predict CLI)
020f382 feat(concept-graph): 语义概念嵌入 + UMAP 科学地图 (cg embed/neighbors/map)
0ca38f9 feat(concept-graph): 共现图构建 (归一化/clique/频率过滤 + cg build CLI)
6ccd6c6 feat(concept-graph): schema v9 + 语料摄入/引用网络 + cg ingest CLI
a1ea94f feat(concept-graph): 多源语料适配器基础 (Sciverse/OpenAlex + 限速重试)
```
