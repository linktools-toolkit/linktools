#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill applicability is shared by discovery, compilation and loading."""

from dataclasses import replace

import pytest

from linktools.ai.agent import AgentBindingContract, AgentCompiler
from linktools.ai.capability import CapabilityContribution, SkillCapability, SkillDefinition, SkillSourceRegistry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.spec import AgentSpec, AgentSpecCodec, SkillSpec, SkillSpecAdapter, SkillSpecCodec


def _compiler(skills: tuple[SkillSpec, ...]) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="test").capture(),
        candidates=tuple(CapabilityContribution.from_declaration(SkillDefinition(skill)) for skill in skills),
        agents={},
    )


def test_type_selection_is_explicit_and_preserves_unscoped_skills() -> None:
    common = SkillSpec("common", "Common instructions")
    scoped = SkillSpec("scoped", "Scoped instructions", metadata={"agent_types": ["analyst"]})
    skills = (common, scoped)
    untyped = AgentSpec("analyst")
    assert untyped.select_skills(skills) == (common,)
    typed = replace(untyped, id="renamed", metadata={"agent_type": "analyst"})
    assert typed.select_skills(skills) == skills
    assert replace(typed, metadata={"agent_type": "Analyst"}).select_skills(skills) == (common,)
    assert replace(typed, allow_skills=()).select_skills(skills) == ()
    assert replace(typed, allow_skills=("scoped",)).select_skills(skills) == (scoped,)
    assert AgentSpec("legacy").select_skills((common,)) == (common,)


@pytest.mark.parametrize("options", [
    {"allow_skills": ("scoped",)},
    {"allow_skills": ("missing",)},
    {"preload_skills": ("scoped",)},
])
def test_unavailable_explicit_or_preloaded_selection_fails(options: dict[str, object]) -> None:
    skill = SkillSpec("scoped", "Instructions", metadata={"agent_types": ["analyst"]})
    with pytest.raises(AIError) as error:
        _compiler((skill,)).compile(AgentSpec("qa", **options))
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.parametrize("metadata", [{"agent_type": None}, {"agent_type": ""}, {"agent_type": "*"}, {"agent_type": []}])
def test_malformed_agent_type_is_rejected(metadata: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentSpec("agent", metadata=metadata)


@pytest.mark.parametrize("value", [None, [], "analyst", [None], [""], ["*"]])
def test_malformed_skill_types_are_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SkillSpec("skill", "Instructions", metadata={"agent_types": value})


@pytest.mark.asyncio
async def test_compiled_scope_controls_prompt_list_and_direct_load() -> None:
    common = SkillSpec("common", "Common instructions")
    scoped = SkillSpec("scoped", "Private instructions", metadata={"agent_types": ["analyst"]})
    compiled = _compiler((common, scoped)).compile(AgentSpec("qa"))
    capability = SkillCapability(compiled.skill_definitions, SkillSourceRegistry())
    assert [item["id"] for item in await capability.list_skills()] == ["common"]
    assert "scoped" not in capability.get_instructions()
    assert "Private instructions" not in capability.get_instructions()
    assert (await capability.load_skill("common"))["instructions"] == common.content
    with pytest.raises(AIError) as error:
        await capability.load_skill("scoped")
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_only_type_metadata_enters_semantic_projection() -> None:
    agent = AgentSpec("agent", metadata={"agent_type": "analyst", "author": "One"})
    codec = AgentSpecCodec()
    assert codec.to_contract_payload(agent)["metadata"] == {"agent_type": "analyst"}
    assert codec.to_contract_payload(agent) == codec.to_contract_payload(replace(agent, metadata={"agent_type": "analyst", "author": "Two"}))
    assert "metadata" not in codec.to_contract_payload(AgentSpec("agent", metadata={"author": "One"}))
    skill = SkillSpecAdapter().decode_markdown(b"---\nname: skill\ndescription: Instructions\nmetadata:\n  agent_types: [analyst]\n  author: One\n---\nBody\n")
    skill_codec = SkillSpecCodec()
    assert skill_codec.to_contract_payload(skill)["metadata"] == {"agent_types": ["analyst"]}
    assert "agent_types" not in skill_codec.model_content(skill)
    assert skill_codec.decode(skill_codec.encode(skill)) == skill


def test_restore_preserves_historical_selection_and_type() -> None:
    original = SkillSpec("scoped", "Old body", metadata={"agent_types": ["analyst"]})
    agent = AgentSpec("agent", metadata={"agent_type": "analyst"}, preload_skills=("scoped",))
    compiler = _compiler((original,))
    binding = compiler.bind(compiler.compile(agent))
    wire = binding.binding_contract.to_payload()
    current = replace(original, content="New body", metadata={"agent_types": ["reporter"]}, revision=2)
    current_compiler = _compiler((current,))
    assert current_compiler.compile(replace(agent, preload_skills=())).skill_definitions == ()
    restored = current_compiler.restore(AgentBindingContract.from_payload(wire))
    assert restored.compiled_agent.skill_definitions[0].spec == original
    assert restored.compiled_agent.spec.metadata == {"agent_type": "analyst"}
    assert restored.binding_digest == binding.binding_digest
    # An old untyped binding also keeps skills that have since become scoped.
    old_compiler = _compiler((SkillSpec("scoped", "Legacy body"),))
    old = old_compiler.bind(old_compiler.compile(AgentSpec("legacy")))
    assert current_compiler.restore(old.binding_contract).compiled_agent.skill_definitions[0].spec.content == "Legacy body"


@pytest.mark.asyncio
async def test_store_capture_keeps_scope_and_historical_asset_resources() -> None:
    from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.storage import StorageOverlay

    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("skill", "guide/SKILL.md")
        await store.put(key, b"---\nname: guide\ndescription: Guide\nmetadata:\n  agent_types: [analyst]\n---\nOld instructions")
        await store.put(AssetKey("skill", "guide/reference.txt"), b"Old reference")
        capture = await CapabilityGroup("custom", assets=store).capture()
        compiler = AgentCompiler(model_resolver=ModelRegistry.openai(model="test").capture(), candidates=capture.contributions, agents={})
        agent = AgentSpec("custom-investigator", metadata={"agent_type": "analyst"})
        binding = compiler.bind(compiler.compile(agent))
        assert binding.compiled_agent.skill_definitions[0].source_ref.resources
        await store.put(key, b"---\nname: guide\ndescription: Guide\nmetadata:\n  linktools-revision: 2\n  agent_types: [reporter]\n---\nNew instructions")
        newer = await CapabilityGroup("custom", assets=store).capture()
        current = AgentCompiler(model_resolver=ModelRegistry.openai(model="test").capture(), candidates=newer.contributions, agents={})
        assert current.compile(agent).skill_definitions == ()
        restored = current.restore(binding.binding_contract)
        assert restored.compiled_agent.skill_definitions == binding.compiled_agent.skill_definitions
    finally:
        await store.close()
