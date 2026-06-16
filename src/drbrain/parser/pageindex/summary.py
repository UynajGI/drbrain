"""LLM-based summarization for document tree nodes."""

from __future__ import annotations

import asyncio

import litellm

from drbrain.extractor.llm_client import acall_text_with_fallback


async def _generate_node_summary(node: dict, models: list[dict]) -> str:
    """Generate summary for a single node via LLM."""
    prompt = (
        "You are given a part of a document. Generate a concise description "
        "of what the main points are covered.\n\n"
        f"Partial Document Text: {node.get('text', '')}\n\n"
        "Directly return the description, do not include any other text."
    )
    response = await acall_text_with_fallback(prompt, models, max_tokens=256)
    return response or ""


async def _generate_doc_description(structure: dict, models: list[dict]) -> str:
    """Generate one-sentence document description via LLM."""
    prompt = (
        "You are an expert in generating descriptions for documents.\n"
        "Given a document structure, generate a one-sentence description "
        "that distinguishes this document from others.\n\n"
        f"Document Structure: {structure}\n\n"
        "Directly return the description, do not include any other text."
    )
    response = await acall_text_with_fallback(prompt, models, max_tokens=128)
    return response or ""


async def _generate_summaries_for_structure_md(
    structure: list[dict],
    summary_token_threshold: int,
    model: str,
    models: list[dict],
) -> list[dict]:
    """Generate summaries for all nodes in the structure."""
    from drbrain.parser.pageindex.builder import _structure_to_list

    nodes = _structure_to_list(structure)

    async def _get_summary(node: dict) -> str:
        node_text = node.get("text", "")
        num_tokens = litellm.token_counter(model=model, text=node_text)
        if num_tokens < summary_token_threshold:
            return node_text
        return await _generate_node_summary(node, models)

    tasks = [_get_summary(node) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        if not node.get("nodes"):
            node["summary"] = summary
        else:
            node["prefix_summary"] = summary

    return structure
