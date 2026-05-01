# DrBrain 论文库管理设计

> 日期：2026-05-01 | 来源：ScholarAIO 图书馆功能学习 | 分支：dev-scholaraio

## 总览

5 个模块：Inbox 分类 → Workspace → Export → Delete 增强 → Backup

## 1. Inbox 分类系统

### 目录结构

```
data/spool/
├── inbox/       # 所有 PDF 入口，自动分类
└── pending/     # 解析失败 PDF + failure.log
```

### 行为

- `drbrain ingest` 扫描 `data/spool/inbox/` 下所有 PDF
- LLM 自动识别 `paper_type`（基于标题+摘要+结构特征），写入 papers 表新增列
- 入库成功 → PDF 留在 papers 目录（不重复存储）
- 入库失败 → PDF 移入 `data/spool/pending/`，写 `pending.jsonl` 记录原因（每行 JSON）

### paper_type 枚举

`paper` / `review` / `thesis` / `preprint` / `book` / `document`

### DB 变更

```sql
ALTER TABLE papers ADD COLUMN paper_type TEXT DEFAULT 'paper'
  CHECK(paper_type IN ('paper','review','thesis','preprint','book','document'));
```

### Config 变更

```yaml
dirs:
  inbox: "data/spool/inbox"
  pending: "data/spool/pending"
```

旧配置中的 `inbox_thesis` 移除，统一用 `inbox`。

## 2. Workspace（引用式论文子集）

### 目录结构

```
workspace/<name>/
├── workspace.yaml     # name, description, created_at
└── refs/
    └── papers.json    # [{"local_id": "p1a2b3c4", "added_at": "2026-05-01T00:00:00"}, ...]
```

### CLI 命令

```
drbrain ws create <name> [-d "description"]
drbrain ws add <name> <local_id...>
drbrain ws remove <name> <local_id...>
drbrain ws list
drbrain ws show <name>
drbrain ws delete <name>
```

### 分析命令扩展

以下命令新增 `--workspace <name>` 参数，从 `papers.json` 读取 local_id 过滤：

```
drbrain query "..." --workspace <name>
drbrain closure --workspace <name>
drbrain seed --workspace <name>
drbrain stats --workspace <name>
drbrain export --workspace <name> --format bib
```

### 实现要点

- Graph load_from_db 时只加载 workspace 内的 paper 的 nodes/edges
- Query 时只搜索 workspace 内的 papers

## 3. Export（论文导出）

### CLI

```
drbrain export <local_id>                    # 默认 BibTeX
drbrain export <local_id> --format ris       # RIS（EndNote/Zotero）
drbrain export <local_id> --format md        # Markdown 引用
drbrain export --all --format bib            # 批量全部
drbrain export --workspace <name> --format bib  # 工作区批量
```

### 格式

| 格式 | 用途 | 文件扩展名 |
|------|------|-----------|
| BibTeX | LaTeX 引用 | .bib |
| RIS | EndNote/Zotero 导入 | .ris |
| Markdown | 人类可读引用列表 | .md |

## 4. Delete 增强

### CLI 变更

```
drbrain delete <local_id>                    # 仅删 DB（现状）
drbrain delete <local_id> --rm-files         # 删除 DB + 论文目录
drbrain clean --papers                       # 清空全部论文目录（保留 inbox）
```

### 行为

- `--rm-files`：删除 `data/papers/<local_id>/` 整个目录
- `--rm-files` 结果写入 JSON 输出

## 5. Backup（本地 tar）

### CLI

```
drbrain backup                               # 创建备份
drbrain backup --output <path>               # 指定输出路径
drbrain backup --list                        # 列出已有备份
```

### 行为

- 默认输出：`data/backups/drbrain-YYYYMMDD-HHMMSS.tar.gz`
- 打包内容：`data/papers/` + `data/db/drbrain.db` + `workspace/` + `data/reports/`
- 排除：`data/cache/`、`data/logs/`、`data/spool/`、`data/backups/`

## 实现顺序

1. DB schema: `paper_type` 列 + Inbox `paper_type` 分类 → Backup（无依赖）
2. Export（读 papers 表）→ Delete --rm-files（读 papers 表）
3. Workspace（需要 DB + Export 整合）

### 可以并行的模块

- Delete --rm-files 和 Backup 不互斥
- Export 独立于 Inbox 分类
