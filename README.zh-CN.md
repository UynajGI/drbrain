<div align="center">

# 🧠 DrBrain

**符号驱动的学术知识图谱 + 轻量向量检索。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0a3-orange.svg)](https://github.com/UynajGI/DrBrain/releases)
[![Agent Skills](https://img.shields.io/badge/Agent_Skills-DrBrain-purple.svg)](skills/)
[![CI](https://github.com/UynajGI/DrBrain/actions/workflows/ci.yml/badge.svg)](https://github.com/UynajGI/DrBrain/actions)

[English](README.md) · **[简体中文](README.zh-CN.md)**

</div>

---

你的 AI 编程代理已经会读写代码了。DrBrain 给它一张
**结构化的学术论文知识图谱**——让它能检索文献、追溯因果链、发现研究空白、
并通过规则推理推断新的关系。

- 📚 论文库变成**概念粒度**的可查询知识图谱。
- 🧩 推理是**符号驱动**的：闭包规则、置信度传播、反事实分析——不只是向量相似度。
- ⚡ 轻量向量检索：只对**语义完整的树节点**做向量，不做任意切片。
- 🤖 **为 AI 代理而生**：每个功能都通过 CLI 访问，你的代理直接就能用。

---

## ✨ 亮点

- **默认增量更新**——往 N 篇论文的库里加 1 篇，只重建这篇；闭包扫描它的邻域；
  嵌入微调即可。
- **7 条推理工作流**——review、gap-analysis、impact、compare、frontier、
  lineage、paradigm。
- **OKF 导出**——把整个知识图谱导出为
  [OKF](https://github.com/UynajGI/DrBrain) v0.1 markdown bundle，
  人类和代理用 `cat` 就能读。
- **27 个代理技能**，遵循开放的 [AgentSkills.io](https://agentskills.io) 标准。

---

## 🚀 快速开始

```bash
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync && uv pip install -e .
drbrain setup          # 交互式向导（中英双语）
```

这会在 `~/DrBrain/` 创建你的库根目录（Windows 下是
`%USERPROFILE%/DrBrain`）。

```bash
# Ingest → build → embed → closure（全增量）
drbrain fetch "10.1038/nature14539"     # 按 DOI 抓取论文
drbrain build                           # 5 阶段 LLM 抽取
drbrain embed                           # TransE 图嵌入
drbrain closure                         # 规则推理
drbrain ask "深度学习还有哪些未解决的问题？"

# 或者一次性串起来
drbrain pipeline --preset full
```

> `pipx install drbrain` 和 `uv tool install drbrain` 将在 beta 版提供。

---

## 📖 功能一览

| 分类 | 命令 | 说明 |
|------|------|------|
| **入库** | `ingest` `fetch` | PDF 经 MinerU 解析 → 5 源元数据交叉验证（arXiv、CrossRef、S2、OpenAlex、DeepXiv）→ LLM 树结构化 |
| **构建** | `build` | 5 阶段概念抽取（增量）：本体扩展 → 实体抽取（10 路并发）→ 关系抽取 → 共指消解 → 迭代精修 |
| **检索** | `query` `search` | BM25 关键词搜索 + PageRank 加权、有向图遍历、混合排序 |
| **知识图谱** | `closure` | 规则闭包（增量）：8+4 条推理规则、t-norm 传递接地、TransE 嵌入链接预测 |
| **嵌入** | `embed` | TransE 图嵌入（增量微调）或 PageIndex/RAPTOR 文本向量 |
| **推理** | `reason` | 符号驱动发现：因果链、置信度传播、反事实分析、跨域同构、假设生成 |
| **工作流** | `reason --workflow` | 7 条结构化推理管线：review、gap-analysis、impact、compare、frontier、lineage、paradigm |
| **会话** | `session` | 持久推理上下文：数据库多轮会话、构建上下文注入、跨调用连续性 |
| **分析** | `analyze` `frontier` | 知识前沿报告：研究种子、辩论区、技术断崖、LLM 执行摘要 |
| **谱系** | `evolve` `descendants` `paradigm` | 概念进化树、学术后代、范式转移检测、跨域迁移 |
| **引用** | `citations` | 多源引用扩展：前向/后向引用、共同引用分析、引用验证 |
| **导出** | `export` `export-okf` | BibTeX/RIS/Markdown（4 种样式）+ OKF v0.1 markdown bundle |
| **导入** | `import` | Zotero（Web API + 本地 SQLite）、BibTeX、Endnote XML/RIS |
| **翻译** | `translate` | LLM 论文翻译：占位符保护分块、语种检测、断点续译 |
| **联邦搜索** | `fsearch` | 本地库 + arXiv 跨源搜索，自动标注已入库状态 |
| **专利搜索** | `patent-search` | USPTO PPUBS（免费）或 ODP（API-key）专利检索 |
| **流水线** | `pipeline` | 步骤串联（增量）：预设（full/quick/embed）+ 自定义步骤；`--full` 强制全量重建 |
| **修复/富化** | `repair` `enrich` | CrossRef/arXiv/OpenAlex 元数据回填、字段补全、可疑记录检测 |
| **审计** | `audit` | 数据质量扫描：15 条分级规则、PDF 预校验、入库质量门 |
| **工作区** | `ws` | 工作区 CRUD：按主题分组论文子集 |
| **备份** | `backup` `restore` | tar.gz 本地归档 + rsync 远程同步 |
| **文档** | `document` | DOCX/PPTX/XLSX 结构化摘要（无需 GUI） |
| **指标** | `metrics` | 使用分析：热门搜索词、最多阅读论文、周趋势 |
| **其他** | `proceedings` `explore` `check` `clean` | 会议集管理、发现集合、环境检查、数据清理 |

<details>
<summary><b>全部命令一览</b></summary>

`setup` `ingest` `fetch` `build` `embed` `closure` `query` `search` `ask`
`reason` `graph` `analyze` `evolve` `landscape` `frontier` `paradigm`
`citations` `export` `export-okf` `import` `translate` `session` `ws`
`pipeline` `repair` `enrich` `audit` `backup` `restore` `metrics`
`document` `fsearch` `patent-search` `proceedings` `explore` `check` `clean`

运行 `drbrain --help` 查看完整列表，或参阅
[CLI 参考手册](docs/cli-reference.md)。

</details>

---

## 🤖 与你的 AI 代理协作

安装 DrBrain 技能，让你的编程代理直接使用：

```bash
npx skills add https://github.com/UynajGI/DrBrain/skills
```

技能遵循开放的 [AgentSkills.io](https://agentskills.io) 标准，兼容
Claude Code、Codex、Cline、Cursor、Windsurf、通义千问 Code、GitHub
Copilot 等 AI 编程工具。

---

## ⚙️ 配置

`drbrain setup` 交互式引导你完成基础配置（中英双语）：

- LLM API key（任意 litellm provider：OpenAI、Anthropic、Ollama、DeepSeek 等）
- MinerU token（可选；无则用 PyMuPDF 回退解析 PDF）
- Semantic Scholar / CrossRef / OpenAlex API keys（可选；提高速率限制）

`drbrain check` 端到端验证你的环境。每个设置详见
[配置文档](docs/configuration.md)。

---

## 📚 文档

| | |
|---|---|
| [入门指南](docs/getting-started.md) | 从安装到第一次查询 |
| [CLI 参考](docs/cli-reference.md) | 所有命令及示例 |
| [架构设计](docs/architecture.md) | 系统设计与推理模块 |
| [配置](docs/configuration.md) | 每个设置、默认值、provider 模板 |
| [工作流](docs/workflows.md) | 结构化推理工作流指南 |
| [会话](docs/sessions.md) | 持久化 Session Agent 深入 |
| [嵌入](docs/embedding.md) | local、openai-compat、none 三种 provider |
| [故障排除](docs/troubleshooting.md) | 常见问题与恢复 |
| [技能参考](docs/skills.md) | 27 个代理技能及其 CLI 命令 |
| [贡献指南](docs/contributing.md) | 如何添加命令、模块和技能 |

---

## 🙏 致谢

DrBrain 的代理优先设计受
[ScholarAIO](https://github.com/ZimoLiao/scholaraio)——"AI 代理的研究基础设施"
先驱——启发。DrBrain 走了不同的技术路线：符号驱动的知识图谱推理 +
语义完整节点的轻量向量检索（而非全文切片嵌入）。树结构检索受
[PageIndex](https://github.com/answerdotai/pageindex) 和
[RAPTOR](https://arxiv.org/abs/2401.18059) 启发。

---

## 🤝 贡献

欢迎贡献——bug 报告、功能建议、文档、代码都有帮助。完整指南见
**[CONTRIBUTING.md](CONTRIBUTING.md)**（英文）。

贡献者快速上手：

```bash
git clone https://github.com/UynajGI/DrBrain.git && cd DrBrain
uv sync && uv pip install -e .
pre-commit install                      # 可选：提交时自动 lint
uv run pytest -m "not integration"      # 快速测试
```

提交 PR 前请确保以下检查通过：

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/drbrain
uv run pytest -m "not integration"
```

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范
（`feat:`、`fix:`、`docs:`、`refactor:`、`test:`、`chore:`）。

有问题或想法？开一个
[讨论](https://github.com/UynajGI/DrBrain/discussions)。发现 bug？
提一个 [issue](https://github.com/UynajGI/DrBrain/issues)。

---

## 📄 许可证

MIT——详见 [LICENSE](LICENSE)。
