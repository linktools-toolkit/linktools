#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill discovery and subagent instruction contracts."""
from linktools.ai.model import ModelResolver, create_model

import pytest

from linktools.ai.agent import AgentCompiler
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AssetCapability,
    SkillCapability,
    SkillDefinition,
    SubagentCapability,
)
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.capability._skill_source import SkillSourceRegistry
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.spec import AgentSpec, AgentSpecCodec, SkillSpec, SubagentRef
from linktools.ai.storage import StorageOverlay


def _skill(identity: str, content: str) -> SkillDefinition:
    return SkillDefinition(SkillSpec(identity, content))


@pytest.mark.asyncio
@pytest.mark.parametrize("available", (False, True))
async def test_group_agent_skill_requirements_survive_capture(
    available: bool,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        if available:
            await store.put(
                AssetKey("skill", "guide/SKILL.md"),
                b"---\nname: guide\ndescription: Follow the guide.\n---\nguide instructions",
            )
        group: RuntimeFeatures[object] = RuntimeFeatures("application")
        group_assets = AssetCapability(store)
        group.capability(group_assets, scope="runtime")
        selections = ["guide", "guide"]
        declared = group.agent("agent", allow_skills=selections)
        selections.clear()
        capture = await group.capture()
        captured = next(item.value for item in capture.contributions if item.kind == "agent")
        assert isinstance(captured, AgentSpec)
        assert captured.allow_skills == declared.allow_skills == ("guide",)
        compiler = AgentCompiler(
            model_resolver=ModelResolver({"default": create_model(protocol='openai', model="unused-offline-model")}),
            candidates=capture.contributions,
            agents={captured.id: captured},
        )
        if available:
            compiled = compiler.compile(captured)
            assert compiled.spec.allow_skills == ("guide",)
            assert tuple(skill.id for skill in compiled.skill_definitions) == ("guide",)
        else:
            with pytest.raises(AIError) as raised:
                compiler.compile(captured)
            assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


def test_agent_spec_skill_selection_codec_round_trips() -> None:
    codec = AgentSpecCodec()
    spec = AgentSpec("agent", allow_skills=("z", "a", "z"))
    assert spec.allow_skills == ("a", "z")
    payload = codec.to_contract_payload(spec)
    assert payload["version"] == 1
    assert payload["allow_skills"] == ["a", "z"]
    assert codec.from_payload(payload) == spec


def test_agent_spec_codec_rejects_unknown_version() -> None:
    payload = AgentSpecCodec().to_contract_payload(AgentSpec("agent"))
    with pytest.raises(AIError) as error:
        AgentSpecCodec().from_payload({**payload, "version": 2})
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_skill_catalog_defers_instruction_bodies_until_requested() -> None:
    capability = SkillCapability(
        (_skill("z", "z-content"), _skill("a", "a-content")),
        SkillSourceRegistry(),
    )
    instructions = capability.get_instructions()
    assert instructions is not None
    assert "load_skill" in instructions
    assert instructions.index("- a:") < instructions.index("- z:")
    assert "a-content" not in instructions
    assert "z-content" not in instructions
    assert (await capability.load_skill("a"))["instructions"] == "a-content"
    assert (await capability.load_skill("z"))["instructions"] == "z-content"


def test_skill_capability_without_skills_has_no_instructions() -> None:
    assert SkillCapability((), SkillSourceRegistry()).get_instructions() is None


@pytest.mark.parametrize("available", (False, True))
def test_subagent_capability_hook_exposes_delegation_instructions(available: bool) -> None:
    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> dict[str, JsonValue]:
        raise AssertionError("Reading instructions must not delegate a task")

    capability = SubagentCapability(
        (SubagentRef("agent", "reviewer"),) if available else (),
        delegate,
        {"reviewer": "Review the proposed changes"} if available else {},
    )
    instructions = capability.get_instructions()
    if available:
        assert instructions is not None
        assert "delegate_task" in instructions
        assert "- reviewer: Review the proposed changes" in instructions
    else:
        assert instructions is None
