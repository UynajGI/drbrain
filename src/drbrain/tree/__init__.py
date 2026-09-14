"""Unified tree RAG: one content store, one node DAG, one navigator.

Design: ``docs/unified-tree-rag-design.md``.  Frozen protocols:
``docs/unified-tree-algorithms.md``.  Task plan:
``docs/unified-tree-atomic-plan.md``.

This package deliberately depends on nothing in ``drbrain.rag`` at import
time: the contracts must be testable without models, network, or the
retrieval runtime.
"""

from drbrain.tree.contracts import (
    CHILD_ORIGINS,
    CONTRACT_SCHEMA,
    DOCUMENT_STATES,
    NODE_KINDS,
    NODE_STATES,
    REJECTION_REASONS,
    STAGE_STATES,
    ChildRef,
    ContentBlock,
    DocumentRevision,
    LeafRef,
    NodeRecord,
    ReadReceipt,
    StageReport,
    StageState,
    child_ref_from_json,
    child_ref_to_json,
    combine_stage_states,
    content_block_id,
    contract_digest,
    is_queryable,
    leaf_node_id,
    node_fingerprint,
    region_node_id,
)

__all__ = [
    "CONTRACT_SCHEMA",
    "NODE_KINDS",
    "NODE_STATES",
    "DOCUMENT_STATES",
    "STAGE_STATES",
    "CHILD_ORIGINS",
    "REJECTION_REASONS",
    "ContentBlock",
    "ChildRef",
    "DocumentRevision",
    "LeafRef",
    "NodeRecord",
    "ReadReceipt",
    "StageReport",
    "StageState",
    "content_block_id",
    "leaf_node_id",
    "region_node_id",
    "child_ref_from_json",
    "child_ref_to_json",
    "contract_digest",
    "combine_stage_states",
    "is_queryable",
    "node_fingerprint",
]
