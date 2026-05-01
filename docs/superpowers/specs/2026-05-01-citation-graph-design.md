# DrBrain 引用图增强设计

> 日期：2026-05-01 | 来源：ScholarAIO citation_check.py + 引用图 | 分支：dev-scholaraio

## 总览

3 个模块：共享引用分析 → 引用校验 → 引用图深度查询

## B1. 共享引用分析（shared-refs）

### 逻辑

给定 paper A：
1. 从 edges 表获取 A 的 `references` 关系 → 得到引用文献集合 R
2. 查找 edges 表中引用 R 中任意文献的其他论文 B（排除 A 自身）
3. 检查 A 和 B 之间是否存在直接引用关系
4. 不存在 → 标记为 `unlinked`（潜在知识盲区/合作机会）

### CLI

```
drbrain citations <local_id> --type shared-refs
```

### 输出（JSON/rich）

```json
{
  "paper": "p1a2b3c4",
  "title": "...",
  "shared_refs": [
    {
      "shared_with": "p5d6e7f8",
      "shared_with_title": "...",
      "shared_count": 3,
      "status": "unlinked",
      "shared_papers": [
        {"title": "Ref A", "year": 2023},
        {"title": "Ref B", "year": 2022}
      ]
    }
  ]
}
```

## B2. 引用校验（citation check）

### 逻辑

1. 从文本中提取 (Author, Year) 引用模式
2. 在本地库中 fuzzy-match 作者名 + 年份
3. 报告匹配/缺失/存在歧义

### 提取模式

- `Author (Year)` — 叙述性引用
- `(Author, Year)` — 括号引用
- `Author et al. (Year)` — et al. 形式

### CLI

```
drbrain check-citations "<text>"
drbrain check-citations --file <path>
```

### 输出

```
Found 5 citations:
  ✓ Smith et al. (2023) → p1a2b3c4 "Deep Learning for Graphs"
  ✗ Jones (2022) → no match (closest: Jones et al. (2021))
  ✓ Lee & Park (2024) → p8f7e6d5 "Attention Mechanisms"
```

## B3. 引用图深度查询（citations 命令）

### 统一命令（替代 expand_cmd）

```
drbrain citations <local_id>                     # 默认：refs + citing
drbrain citations <local_id> --type refs          # 参考文献
drbrain citations <local_id> --type citing         # 被引用
drbrain citations <local_id> --type shared-refs    # 共享引用
drbrain citations <local_id> --type all            # 全部
drbrain citations <local_id> --workspace <name>    # 工作区限定
drbrain citations <local_id> --json                # JSON 输出
```

### 新增：citation_cache 表

缓存外部 API 拉回的引用数据，避免重复请求：

```sql
CREATE TABLE IF NOT EXISTS citation_cache (
    source_paper TEXT NOT NULL,
    target_title TEXT NOT NULL,
    target_year INTEGER,
    relation TEXT NOT NULL CHECK(relation IN ('references','citing')),
    target_doi TEXT,
    target_s2_id TEXT,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_paper, target_title)
);
```

## 文件

| File | Action | Purpose |
|------|--------|---------|
| `src/drbrain/storage/citation_graph.py` | Create | 共享引用逻辑 + 引用图查询 |
| `src/drbrain/extractor/citation.py` | Modify | +citation_cache 读写 |
| `src/drbrain/extractor/citation_check.py` | Create | 文本引用提取 + 库内匹配 |
| `src/drbrain/cli/commands.py` | Modify | citations_cmd + check_citations_cmd |
| `src/drbrain/cli/main.py` | Modify | 注册新命令 |
| `src/drbrain/storage/database.py` | Modify | citation_cache 表 |
| `tests/test_citation_graph.py` | Create | |
| `tests/test_citation_check.py` | Create | |

## 实现顺序

B1（shared-refs） → B3（citations 命令） → B2（引用校验，可并行）
