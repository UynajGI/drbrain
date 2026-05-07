# Layer 5: Graph Engine Tree Integration

## Goal

图引擎感知树节点。closure/descendants/transfers 输出带 section 面包屑，landscape 按文档结构分组。

## Requirements

### GraphEngine 新增方法
- `get_concepts_by_node(db, node_id)` → 查询某个 tree node 下的概念
- `get_section_context(db, concept_label)` → 给概念查找其 section/node 上下文
- `traverse_with_sections(...)` → 遍历结果带 section 面包屑

### CLI 增强（部分）
- graph_commands.py 已有命令输出中注入 section 信息
- 在 neighbors/path/descendants 的输出中加入 `section` 字段

## Out of Scope
- ReasonerAgent 树工具（Layer 6）
- CLI --sections flags（Layer 7）
