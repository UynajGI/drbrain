# DrBrain 文档索引

本文档目录以当前代码和 CLI 为准。历史调研、阶段计划和已被实现替代的设计稿位于 [`archive/`](archive/)。知识图谱与 WebUI 文档保留原位置，暂不纳入本次整理。

## 快速开始

- [安装与首次运行](getting-started.md)
- [CLI 参考](cli-reference.md)
- [配置](configuration.md)
- [故障排查](troubleshooting.md)

## 当前架构与运行时

- [系统架构](architecture.md)
- [RAG 层当前契约](rag-layer-completion.md)
- [插件系统](plugins.md)
- [研究循环当前状态](loop-current-state.md)
- [自动研究运维](autoresearch-operations.md)
- [结构化工作流](workflows.md)
- [持久化会话](sessions.md)

## 数据与模型能力

- [Embedding](embedding.md)
- [技能目录](skills.md)
- [术语表](glossary.md)
- [API 参考](api-reference.md)

## 开发与维护

- [贡献指南](contributing.md)
- [备份技能](../skills/backup/SKILL.md)
- [安全策略](../SECURITY.md)
- [变更记录](../CHANGELOG.md)

文档中的命令应以 `uv run drbrain --help` 和对应模块的实际实现为准；如果说明与代码不一致，以代码和 CLI 输出为准，并应在同一变更中修正文档。
