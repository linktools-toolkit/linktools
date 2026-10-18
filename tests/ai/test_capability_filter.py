#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application capability filters narrow one captured candidate universe."""

from dataclasses import dataclass, replace

import pytest
import pytest_asyncio
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai import Runtime
from linktools.ai.agent import AgentBindingContract, AgentCompiler
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import CapabilityContribution, RuntimeFeatures, SkillCapability, SkillDefinition, SkillSourceRegistry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, SkillSpec, SkillSpecCodec
from linktools.ai.storage import StorageOverlay


@dataclass
class _ExtraCapability(AbstractCapability[object]):
    id: str


@pytest_asyncio.fixture
async def contributions():
    group = RuntimeFeatures("application")
    group.agent("root")
    group.agent("child", metadata={"business": "captured"}, description="Delegate work", revision=3)
    group.mcp(MCPServerSpec("server", "python"))
    group.capability(_ExtraCapability("extra"))

    def business(context: object) -> str:
        return "ready"

    group.tool(business, name="business")
    return (*((await group.capture()).contributions), CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("guide", "Instructions"))))


def _compiler(contributions, capability_filter=None):
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="test").capture(),
        candidates=contributions,
        agents={item.id: item.value for item in contributions if item.kind == "agent"},
        capability_filter=capability_filter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["agent", "tool", "skill", "mcp", "capability"])
async def test_filter_narrows_every_supported_kind_without_filtering_root(contributions, kind):
    seen = []

    def include(agent, item):
        seen.append((agent.id, item.kind, item.id))
        return item.kind != kind

    agent = AgentSpec("root")
    visible = AgentCompiler.filter_capabilities(agent, contributions, capability_filter=include)
    assert all(item.kind != kind for item in visible)
    assert not any(identity == "root" for _, _, identity in seen)
    compiled = _compiler(contributions, include).compile(agent)
    selected = (*compiled.selected_tools, *compiled.selected_skills, *compiled.selected_mcp, *compiled.selected_capabilities)
    assert all(item.kind != kind for item in selected)
    assert compiled.spec.id == "root"
    if kind == "agent":
        assert compiled.selected_subagents == ()
    if kind == "mcp":
        assert compiled.mcp_policy == ()
    # A child's own candidates use its own AgentSpec, not the parent's result.
    _compiler(contributions, include).compile(AgentSpec("child"))
    assert any(agent_id == "child" for agent_id, _, _ in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"allow_tools": ("business",)},
    {"allow_tools": ("mcp:server:*",)},
    {"allow_skills": ("guide",)},
    {"allow_subagents": ("child",)},
    {"allow_capabilities": ("extra",)},
    {"preload_skills": ("guide",)},
])
async def test_explicit_filtered_selection_or_preload_is_a_configuration_error(contributions, options):
    with pytest.raises(AIError) as error:
        _compiler(contributions, lambda agent, item: False).compile(AgentSpec("root", **options))
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_default_filter_preserves_selection_and_diagnostic_errors(contributions):
    agent = AgentSpec("root")
    plain = _compiler(contributions).compile(agent)
    included = _compiler(contributions, lambda agent, item: True).compile(agent)
    assert included == plain
    error = RuntimeError("application metadata is invalid")

    def broken(agent, item):
        raise error

    with pytest.raises(RuntimeError) as raised:
        _compiler(contributions, broken).compile(agent)
    assert raised.value is error
    with pytest.raises(TypeError, match="return bool"):
        _compiler(contributions, lambda agent, item: None).compile(agent)

    async def asynchronous(agent, item):
        return True

    with pytest.raises(TypeError, match="synchronously"):
        _compiler(contributions, asynchronous).compile(agent)


@pytest.mark.asyncio
async def test_filtered_skill_is_absent_from_instructions_list_and_load(contributions):
    compiled = _compiler(contributions, lambda agent, item: item.kind != "skill").compile(AgentSpec("root"))
    capability = SkillCapability(compiled.skill_definitions, SkillSourceRegistry())
    assert await capability.list_skills() == []
    assert capability.get_instructions() is None
    with pytest.raises(AIError) as error:
        await capability.load_skill("guide")
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_restore_keeps_pins_and_does_not_execute_current_filter(contributions):
    compiler = _compiler(contributions)
    binding = compiler.bind(compiler.compile(AgentSpec("root")))

    def unexpected(agent, item):
        raise AssertionError("restore must not invoke the application filter")

    restored = _compiler(contributions, unexpected).restore(AgentBindingContract.from_payload(binding.binding_contract.to_payload()))
    assert restored.binding_digest == binding.binding_digest
    assert restored.compiled_agent.selected_subagents == ("child",)
    assert restored.compiled_agent.skill_definitions == binding.compiled_agent.skill_definitions
    assert restored.compiled_agent.selected_tools == binding.compiled_agent.selected_tools
    assert restored.compiled_agent.selected_mcp == binding.compiled_agent.selected_mcp
    assert restored.compiled_agent.selected_capabilities == binding.compiled_agent.selected_capabilities


@pytest.mark.asyncio
async def test_runtime_wires_filter_into_roots_and_derived_agents(tmp_path):
    group = RuntimeFeatures("application", filter=lambda agent, item: item.kind != "skill")
    group.agent("root")
    capture = await group.capture()
    capture = replace(capture, contributions=(*capture.contributions, CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("guide", "Instructions")))))
    async with Runtime.open(
        "capability-filter-test", models=ModelRegistry.openai(model="test"),
        storage=RuntimeStorage.from_root(tmp_path), features=capture,
    ) as runtime:
        root = runtime.agents.get("root")
        assert root.derive(allow_skills=("*",)).compiled.skill_definitions == ()
        with pytest.raises(AIError) as error:
            root.derive(allow_skills=("guide",))
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_custom_asset_capture_filter_and_historical_resources():
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("skill", "guide/SKILL.md")
        await store.put(key, b"---\nname: guide\ndescription: Guide\nmetadata:\n  audience: [analyst]\n---\nOld instructions")
        await store.put(AssetKey("skill", "guide/reference.txt"), b"Old reference")
        capture = await RuntimeFeatures("custom", assets=store).capture()
        include = lambda agent, item: item.kind != "skill" or agent.metadata.get("role") in item.contract["metadata"]["audience"]
        compiler = _compiler(capture.contributions, include)
        agent = AgentSpec("custom-investigator", metadata={"role": "analyst"})
        binding = compiler.bind(compiler.compile(agent))
        assert binding.compiled_agent.skill_definitions[0].source_ref.resources
        await store.put(key, b"---\nname: guide\ndescription: Guide\nmetadata:\n  linktools-revision: 2\n  audience: [reporter]\n---\nNew instructions")
        newer = await RuntimeFeatures("custom", assets=store).capture()
        current = _compiler(newer.contributions, include)
        assert current.compile(agent).skill_definitions == ()
        assert current.restore(binding.binding_contract).compiled_agent.skill_definitions == binding.compiled_agent.skill_definitions
    finally:
        await store.close()


def test_application_metadata_remains_outside_upstream_semantic_projection():
    for key in ("agent_type", "tags", "anything"):
        agent = AgentSpec("agent", metadata={key: ["value"]})
        assert "metadata" not in AgentSpecCodec().to_contract_payload(agent)
        skill = SkillSpec("guide", "Instructions", metadata={key: ["value"]})
        assert "metadata" not in SkillSpecCodec().to_contract_payload(skill)


@pytest.mark.asyncio
async def test_delegation_filter_keeps_captured_agent_metadata_and_named_reference(contributions):
    captured = next(item for item in contributions if item.kind == "agent" and item.id == "child")
    observed = []

    def include(agent, item):
        if item.kind == "agent":
            observed.append(item.value)
        return True

    compiler = _compiler(contributions, include)
    binding = compiler.bind(compiler.compile(AgentSpec("root")))
    assert observed == [captured.value]
    assert observed[0].metadata == {"business": "captured"}
    assert binding.binding_contract.subagents[0].to_payload() == {
        "kind": "agent", "id": "child", "revision": 3, "description": "Delegate work",
    }


@pytest.mark.asyncio
async def test_runtime_scope_is_captured_but_never_agent_selected_or_filtered():
    group = RuntimeFeatures("application")
    spec = group.agent("root")
    group.capability(_ExtraCapability("selected"))
    group.capability(_ExtraCapability("host"), scope="runtime", revision=2)
    capture = await group.capture()
    seen = []

    def include(_agent, item):
        seen.append(item.id)
        return True

    scopes = {item.id: item.scope for item in capture.contributions}
    assert scopes == {"root": "agent", "selected": "agent", "host": "runtime"}
    visible = AgentCompiler.filter_capabilities(spec, capture.contributions, capability_filter=include)
    assert [item.id for item in visible] == ["selected"]
    assert seen == ["selected"]
    compiler = _compiler(capture.contributions, include)
    compiled = compiler.compile(spec)
    assert [item.id for item in compiled.selected_capabilities] == ["selected"]
    binding = compiler.bind(compiled)
    without_host = _compiler(tuple(item for item in capture.contributions if item.scope == "agent"))
    assert without_host.bind(without_host.compile(spec)).binding_digest == binding.binding_digest
    assert without_host.restore(binding.binding_contract).binding_digest == binding.binding_digest
    assert "host" not in seen


@pytest.mark.asyncio
async def test_filter_configuration_is_captured_and_replace_preserves_source():
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("skill", "guide/SKILL.md"), b"---\nname: guide\ndescription: Guide\nmetadata:\n  audience: [analyst]\n---\nInstructions")
        calls = []

        def policy(agent, item):
            calls.append((agent.id, item.kind, item.id))
            return True

        class CustomFeatures(RuntimeFeatures):
            pass

        capture = await CustomFeatures("filtered", assets=store, filter=policy).capture()
        assert capture.filter is policy
        assert "filter=" not in repr(capture)
        seen_metadata = []

        def selected(agent, item):
            seen_metadata.append(item.contract["metadata"]["audience"])
            return agent.metadata.get("role") in item.contract["metadata"]["audience"]

        filtered = replace(capture, filter=selected)
        assert filtered == capture
        assert filtered.contributions is capture.contributions
        assert filtered.asset_reader is capture.asset_reader
        assert filtered.source_revision == capture.source_revision
        await filtered.verify_source_revision()
        compiler = _compiler(filtered.contributions, filtered.filter)
        binding = compiler.bind(compiler.compile(AgentSpec("root", metadata={"role": "analyst"})))
        assert seen_metadata == [["analyst"]]
        assert calls == []
        compiler.restore(binding.binding_contract)
        assert seen_metadata == [["analyst"]]
        compiler.compile(AgentSpec("new", metadata={"role": "analyst"}))
        assert seen_metadata == [["analyst"], ["analyst"]]
        assert "filter" not in binding.binding_contract.to_payload()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_filter_configuration_rejects_invalid_values_and_old_runtime_parameter():
    for invalid in (False, 1, "policy"):
        with pytest.raises(TypeError, match="filter must be callable"):
            RuntimeFeatures("invalid", filter=invalid)
        capture = await RuntimeFeatures("valid").capture()
        with pytest.raises(TypeError, match="filter must be callable"):
            replace(capture, filter=invalid)
    with pytest.raises(TypeError, match="capability_filter"):
        Runtime.open("old", models=ModelRegistry.openai(model="test"),
                     storage=RuntimeStorage.in_memory(), capability_filter=lambda *_: True)


def test_runtime_rejects_removed_runtime_features_keyword():
    with pytest.raises(TypeError, match="runtime_features"):
        Runtime.open("old", models=ModelRegistry.openai(model="test"),
                     storage=RuntimeStorage.in_memory(), runtime_features=RuntimeFeatures("old"))
