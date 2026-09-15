# Handoff — CLI/RAG 使用方式重构（round 2：四项遗留收口）

日期：2026-09-15 ｜ 分支：`phy` = **`e5b7a09`**（本地，未 push，无 upstream）
状态：**本轮功能代码已合并**；接手审阅已补跑项目非集成回归，但完整设计验收仍有缺口（见 §7）。真实模型冒烟由本轮跑过（见 §4）。

---

## 1. 本轮做了什么（相对 `c512df4`，4 个提交）

| commit | 内容 |
|---|---|
| `6f7324e` | **统一三路检索**：无 SQL 语料库时，`search`/`ask` 的 bm25/vector/tree 三腿全部从主库统一 store 出——bm25 读规范 FTS（`db.search_content`，块命中解析为已发布叶节点）、vector 读共享叶 ANN（`view=leaf`）、tree 走同一 generation；同一 RRF 融合 + 重排 + 主库物化（`tree_nodes` 叶行 + `content_blocks` 原文）。三腿共用 `paper_id:node_id` 键、修订与内容哈希（同一证据身份），无 `drbrain_rag.db` 参与。修复了 zvec 后端下统一路径不可达的守卫顺序问题（先捕获快照、再查语料库目录）。 |
| `acbf223` | **skills 全量迁移**：11 个 `skills/*/SKILL.md` 改用新主线命令（`graph build/embed/closure`、`index build`、`search`）；剩余 3 处旧命令名是**有意保留**的兼容说明（fsearch/index/paper-query 里标注 hidden alias）。CHANGELOG 同步。 |
| `ba3cd13` | **分片合并支持统一表**：`scripts/pipeline/merge_shards.py` 除旧表外合并 `document_revisions`/`content_blocks`/`tree_nodes(kind='leaf')`；分片 region 明确留在原库不合并；FTS 由 `content_blocks` 触发器随合并自动同步；ANN/hierarchy 由主库一次 `index build` 重建（分片阶段不再需要 `embed --tree`）。新增 `tests/test_merge_shards.py`。 |
| `e5b7a09` | **模型导航两处真实缺陷修复**（真实模型验收发现）：① 一次响应含多个 `tool_calls` 时只记第一个 id → 下一条请求缺 tool 应答，DeepSeek 返回 400（"insufficient tool messages following tool_calls"）；现在记录全部 id，仅执行第一个动作，其余回 "one action per round" 拒绝。② `ChatActionPlanner(max_tokens=512)` 在推理端点上被隐藏推理吃光 → `finish_reason=length` 无动作；默认提到 2048。新增回归用例。 |

背景（上一轮已合并，`c512df4` 及之前）：CLI 主线收敛为 `ingest → index build → search / ask`，旧命令（query/hybrid/fsearch/build/embed/closure/rag prepare|index|health、裸 index）为 hidden 兼容别名 + stderr 迁移提示；`index build/status/verify`；`search` 证据检索（`--paper`、`--source arxiv|all`）；`library search`；`prepare_unified_index` 记录 `tree.prepare.last` 并据此把失败构建挡在 ready 之外。

## 2. 测试环境（主检出即可，无需 worktree）

```bash
cd /home/jiangyuan/drbrain-phy
env -u DRBRAIN_ROOT uv run pytest -m "not integration"        # 全量非集成（≈11 分钟）
env -u DRBRAIN_ROOT uv run pytest tests/tree tests/test_sql_retrie.py tests/test_cli_index.py tests/test_cli_search.py tests/test_cli_compat.py tests/test_merge_shards.py tests/tree/test_tree_leg.py tests/tree/test_navigator_planner.py -q
```

要点：
- **必须** `env -u DRBRAIN_ROOT`（本机 ambient 的 `DRBRAIN_ROOT=/tmp/drbrain-integration` 会污染无关测试）。
- 不要加 `-p no:logging`（会禁用 `caplog` 导致 7 个假 ERROR）。
- 已知环境性失败（与代码无关）：5 个需要活模型的 `*_real_*`（`test_rag_llm::test_integration_real_call_hits_cache`、`test_rag_engine::test_integration_ask_llamaindex_real`、`test_rag_agent::test_integration_reason_llamaindex_real_llm`、`test_rag_eval::test_integration_eval_real_corpus`、`test_rag_retrievers::test_tree_retriever_real_llm_smoke`）——`-m "not integration"` 下被 deselect。
- **本轮全量套件未跑完**：按用户要求中途停止（当时 ~46%，此前基线 4004 passed / 15 skipped / 34 deselected；本轮的单元级验证见 §3）。请补跑一次完整 `-m "not integration"`。

## 3. 本轮已完成的验证（接手人可直接引用）

- 变更级单测：`tests/test_cli_index.py` 26、`tests/test_cli_compat.py` 10、`tests/test_cli_search.py` 13、`tests/tree/test_navigator_planner.py`（含新增多 tool_calls 用例）、`tests/test_merge_shards.py`、`tests/tree/test_tree_leg.py::TestUnifiedCorpusThreeLegs`（3 例：同一发布三腿 / 规范 FTS bm25 / 论文作用域三腿）——全绿。
- 影响面回归：`tests/tree` + `test_sql_retrie` + `test_rag_layer_contracts` + `test_rag_engine` + `test_rag_retrievers` + `test_cli_*`（-m "not integration"）= **727 passed**（改 navigator 前；改后 navigator 专项 44 passed）。
- CLI 冒烟（staged 夹具，无 SQL 语料）：`search "solar"` 三腿全命中（bm25 22 / vector 100 / tree 1），行 `legs=['bm25','vector','tree']`；`--paper` 作用域正确；`library search`、裸 `index --json`、`rag health --json` 旧契约不变。

## 4. 真实模型验收（本轮已跑，可复现）

环境准备：
```bash
# 起 index model（Spark-X2.5-4B fp16，GPU0；模型在 snapshots/master 子目录里）
cd /home/jiangyuan/drbrain-phy
DRBRAIN_SERVE_MODEL_PATH=/home/jiangyuan/.cache/modelscope/models/XHToken--Spark-X2.5-4B/snapshots/master \
DRBRAIN_SERVE_MODEL_ID=spark-x25-4b DRBRAIN_NO_QUANT=1 DRBRAIN_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=0 \
/home/jiangyuan/mineru-venv/bin/python scripts/serve_transformers_llm.py 8010
# 就绪判定：GET http://127.0.0.1:8010/v1/models 返回 200 且 id=spark-x25-4b
```
验收结果（小语料，2 篇/6 节/12 叶；配置为本仓库 `data/integration/unified-tree/config.yaml` 的副本）：
- `index build --json`：exit 0、`ok=true`、发布 `gen-…`、层级 `built`（1 region 接受 / 7 拒 no_compression_gain / 0 失败），全程 34s（层级 23s）；模型单次调用 ≈2–3s。
- `index status --json`：`status: ready`、tree ready（1 region / 12 leaves）、`last_build.ok=true`。
- `search "…" --json`：三腿 ok、同一 generation、行内含 block/char 定位。
- `ask --json`（DeepSeek 合成）：exit 0、10 条 evidence_id、答案正常；模型导航 `planner=chat_model`、无回退（修复后）。
- 规模参考：1049 叶的夹具语料单卡串行构建 ≈40+ 分钟（~284 个候选组 × 2–3s）——语料级构建成本需优化（并行/缓存）。
- ⚠️ 验收后**已停掉** 8010 服务（GPU 已释放）。

## 5. 已知限制 / 后续（不要当成本轮已完成）

1. **分片 ingest 尚未写规范正文**：`merge_shards` 已能携带统一表，但 `scripts/pipeline/ingest_scibase.py` 还是旧式写入；分片管线改 `index build` 的前提是这个 ingest 迁移（当前两个 shard 脚本仍走 compat 的 `embed --tree`，行为未变）。
2. **统一 store 优先级**：存在已发布 SQL 快照时仍走旧快照腿（bm25/vector 读 `drbrain_rag.db`）；"统一优先"的切换未做（设计验收顺序 3 的残留）。
3. **`search --source arxiv|all`**：单元测试用 mock 覆盖；未做真实网络验收（arXiv 外呼）。
4. **构建成本**：叶规模大时层级构建按候选组串行调用 index model，无并行/增量缓存复用（仅 summary cache 级）。
5. **skills**：主命令语义变化的三类已改；其余以兼容别名继续可用。
6. 未提交的参考文档仍在磁盘（按既定口径不入库）：`docs/cli-pipeline-redesign.md`、`docs/post-rag-layers.md`、`docs/reviews/`。

## 6. 接触面速查

- 检索链：`src/drbrain/rag/sql_retrie.py`（`_unified_corpus_retrieval` + `_unified_bm25_entries`/`_unified_vector_entries`/`_unified_materialize`/`_unified_embedder`）、`src/drbrain/tree/leg.py`（`tree_storage_root` 公开）、`src/drbrain/tree/navigator.py`（ChatActionPlanner）。
- CLI：`src/drbrain/cli/index_commands.py`、`search_commands.py`、`library_commands.py`、`_compat.py`、`main.py`。
- 管线：`scripts/pipeline/merge_shards.py`。
- 测试：`tests/test_cli_index.py`、`tests/test_cli_search.py`、`tests/test_cli_compat.py`、`tests/test_merge_shards.py`、`tests/tree/test_tree_leg.py`、`tests/tree/test_navigator_planner.py`。

## 7. 接手复核与归档门槛（2026-09-15）

- 在 `phy@e5b7a09` 补跑项目 `tests/` 非集成回归：4025 passed、15 skipped、34 deselected。独立审阅另复现 9 类设计差距，涉及发布版本/读取范围、scope、重建/恢复、partial、离线 ingest 和就绪判定。代码位置、复现条件和测试统计见 [差距审阅报告](reviews/rag-design-gap-review-2026-09-15.md)；不能把项目回归通过视为完整设计验收通过。
- 用户确认的收尾规则：10k 的入库/迁移、索引、三路检索、引用一致性和故障恢复全部验收通过后，将本轮阶段计划、handoff、历史审阅报告移到 `docs/archive/rag-10k/`，保留提交及验收证据关联并修复链接；最终算法/契约归并到长期架构文档，`docs/` 主层保留当前设计、配置、CLI、运维与验收入口。归档动作纳入原子计划 T64，当前条件尚未满足。
