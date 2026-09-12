"""Unit contracts for the loop's per-agent tool-space resolver."""

from __future__ import annotations

from drbrain.capabilities import CapabilityCatalog, CapabilityDescriptor
from drbrain.capabilities.adapters import APIAdapter, ModelAdapter
from drbrain.loop.director import ResearchDirector
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.tool_space import LoopToolSpace
from drbrain.loop.workflow import ResearchLoopWorkflow


def test_role_tool_space_enforces_prompt_boundaries():
    policy = ToolPolicy()
    graph = ToolDefinition(
        name="search_concepts",
        source="graph",
        input_schema={"type": "object"},
        side_effect="read",
        required_capabilities=("graph:read",),
    )
    assert not LoopToolSpace(step_name="identify_gaps", role="analyst", policy=policy).is_visible(
        graph
    )
    assert not LoopToolSpace(step_name="critique", role="critic", policy=policy).is_visible(graph)
    assert LoopToolSpace(step_name="verify", role="verifier", policy=policy).is_visible(graph)
    future_tool = ToolDefinition(
        name="future_tool",
        source="future_protocol",
        input_schema={"type": "object"},
        side_effect="read",
        required_capabilities=("future:read",),
    )
    assert not LoopToolSpace(step_name="identify_gaps", role="analyst").is_visible(future_tool)


def test_catalog_recommendation_is_filtered_by_node_policy():
    catalog = CapabilityCatalog()
    catalog.register_adapter(
        APIAdapter(
            provider="papers",
            name="search",
            description="Search papers",
            method="GET",
            url="https://example.test/search",
            request=lambda **_kwargs: (200, {"papers": []}),
        )
    )
    catalog.register_adapter(
        ModelAdapter(
            name="calculator",
            description="Run a calculator model",
            predict=lambda _args: {"value": 1},
        )
    )
    retrieve = LoopToolSpace(
        step_name="retrieve",
        policy=ToolPolicy(step_capabilities={"retrieve": {"api:papers:search"}}),
        catalog=catalog,
    )
    assert [item.id for item in retrieve.recommend("search papers")] == ["api:papers:search"]


def test_skill_context_is_instruction_material_and_is_bounded(tmp_path):
    skill = tmp_path / "paper-search"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: paper-search\ndescription: Search papers\n---\nUse the paper search workflow.",
        encoding="utf-8",
    )
    catalog = CapabilityCatalog()
    catalog.register_skills(str(tmp_path))
    space = LoopToolSpace(step_name="retrieve", catalog=catalog)
    context = space.skill_context(max_chars=200)
    assert "Skill: paper-search" in context
    assert "Use the paper search workflow." in context


def test_role_matrix_is_enforced_across_external_capability_kinds():
    catalog = CapabilityCatalog()
    catalog.register_adapter(
        APIAdapter(
            provider="papers",
            name="lookup",
            description="Read paper metadata",
            method="GET",
            url="https://example.test/lookup",
        )
    )
    catalog.register_adapter(
        ModelAdapter(name="score", description="Score a candidate", predict=lambda _args: 1)
    )
    catalog.register(
        CapabilityDescriptor(
            id="mcp:papers:search",
            name="search",
            description="Search via MCP",
            kind="mcp_tool",
            input_schema={"type": "object"},
            permissions=("mcp:papers:search",),
            metadata={"trusted": True, "allowed_tools": ["search"], "side_effect": "read"},
        ),
        lambda _args: {"papers": []},
    )
    policy = ToolPolicy(
        step_capabilities={
            "retrieve": {"api:papers:lookup", "model:score", "mcp:papers:search"},
            "identify_gaps": {"api:papers:lookup", "model:score", "mcp:papers:search"},
            "compute": {"api:papers:lookup", "model:score", "mcp:papers:search"},
            "verify": {"api:papers:lookup", "model:score", "mcp:papers:search"},
        }
    )
    analyst = LoopToolSpace(
        step_name="identify_gaps", role="analyst", policy=policy, catalog=catalog
    )
    assert not analyst.recommend("lookup score search")

    compute = LoopToolSpace(step_name="compute", role="compute", policy=policy, catalog=catalog)
    assert {item.id for item in compute.recommend("lookup score search")} == {
        "api:papers:lookup",
        "model:score",
        "mcp:papers:search",
    }

    verifier = LoopToolSpace(step_name="verify", role="verifier", policy=policy, catalog=catalog)
    assert {item.id for item in verifier.recommend("lookup score search")} == {
        "api:papers:lookup",
        "model:score",
        "mcp:papers:search",
    }


def test_director_checkpoint_records_supplied_capability_contract(tmp_path):
    adapter = ModelAdapter(name="local-model", description="A local model", predict=lambda _args: 1)
    manifest = ResearchDirector(
        cfg=object(),
        capability_adapters=[adapter],
        skills_root=tmp_path / "skills",
        require_trusted_mcp=True,
    )._checkpoint_manifest()  # noqa: SLF001 - contract-level test
    assert manifest.tool_manifest["capability_descriptors"][0]["id"] == "model:local-model"
    assert manifest.tool_manifest["require_trusted_mcp"] is True


def test_workflow_augments_injected_catalog_once():
    catalog = CapabilityCatalog()
    catalog.register_adapter(
        ModelAdapter(name="seed-model", description="Seed model", predict=lambda _args: 1)
    )
    adapter = APIAdapter(
        provider="papers",
        name="lookup",
        description="Read paper metadata",
        method="GET",
        url="https://example.test/lookup",
    )
    workflow = ResearchLoopWorkflow(capability_catalog=catalog, capability_adapters=[adapter])
    first = workflow._tool_space(step_name="retrieve")  # noqa: SLF001 - resolver contract
    second = workflow._tool_space(step_name="verify")  # noqa: SLF001 - resolver contract
    assert {item.id for item in first.catalog.list()} == {
        "model:seed-model",
        "api:papers:lookup",
    }
    assert first.catalog is second.catalog
