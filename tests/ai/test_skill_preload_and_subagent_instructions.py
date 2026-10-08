#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill preload and subagent instruction contracts."""

import pytest

from linktools.ai.agent import AgentCompiler
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    CapabilityGroup,
    SkillCapability,
    SkillDefinition,
    SubagentCapability,
)
from linktools.ai.capability._skill_source import SkillSourceRegistry
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.spec import AgentSpec, AgentSpecCodec, SkillSpec, SubagentRef
from linktools.ai.storage import StorageOverlay


def _skill(identity: str, content: str) -> SkillDefinition:
    return SkillDefinition(SkillSpec(identity, content))


@pytest.mark.asyncio
@pytest.mark.parametrize("available", (False, True))
async def test_group_agent_preloads_survive_capture_and_require_selected_skills(
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
        group: CapabilityGroup[object] = CapabilityGroup("application", assets=store)
        preloads = ["guide", "guide"]
        declared = group.agent("agent", preload_skills=preloads)
        preloads.clear()
        capture = await group.capture()
        captured = next(item.value for item in capture.contributions if item.kind == "agent")
        assert isinstance(captured, AgentSpec)
        assert captured.preload_skills == declared.preload_skills == ("guide",)
        compiler = AgentCompiler(
            model_resolver=ModelRegistry.openai(model="unused-offline-model").capture(),
            candidates=capture.contributions,
            agents={captured.id: captured},
        )
        if available:
            compiled = compiler.compile(captured)
            assert compiled.spec.preload_skills == ("guide",)
            assert tuple(skill.id for skill in compiled.skill_definitions) == ("guide",)
        else:
            with pytest.raises(AIError) as raised:
                compiler.compile(captured)
            assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.parametrize("preloads", (("*",), ("other",)))
def test_group_agent_validates_preload_selectors(preloads: tuple[str, ...]) -> None:
    with pytest.raises(AIError) as raised:
        CapabilityGroup("application").agent(
            "agent", allow_skills=("guide",), preload_skills=preloads,
        )
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_agent_spec_preload_codec_keeps_v1_and_omits_empty_field() -> None:
    codec = AgentSpecCodec()
    default = AgentSpec("agent")
    payload = codec.to_contract_payload(default)
    assert payload["version"] == 1
    assert "preload_skills" not in payload
    assert codec.from_payload(payload).preload_skills == ()

    explicit_empty = codec.from_payload({**payload, "preload_skills": []})
    assert explicit_empty.preload_skills == ()
    assert "preload_skills" not in codec.to_contract_payload(explicit_empty)


def test_agent_spec_preload_codec_canonicalizes_non_empty_ids() -> None:
    codec = AgentSpecCodec()
    spec = AgentSpec(
        "agent",
        allow_skills=("z", "a", "z"),
        preload_skills=("z", "a", "z"),
    )
    assert spec.preload_skills == ("a", "z")
    payload = codec.to_contract_payload(spec)
    assert payload["version"] == 1
    assert payload["preload_skills"] == ["a", "z"]
    assert codec.from_payload(payload) == spec


def test_agent_spec_preload_rejects_invalid_type_wildcard_and_not_allowed() -> None:
    codec = AgentSpecCodec()
    base = codec.to_contract_payload(AgentSpec("agent"))
    with pytest.raises(AIError) as invalid_type:
        codec.from_payload({**base, "preload_skills": "skill"})
    assert invalid_type.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    with pytest.raises(AIError) as wildcard:
        AgentSpec("agent", preload_skills=("*",))
    assert wildcard.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    with pytest.raises(AIError) as not_allowed:
        AgentSpec("agent", allow_skills=("allowed",), preload_skills=("other",))
    assert not_allowed.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_agent_spec_preload_rejects_unknown_version() -> None:
    payload = AgentSpecCodec().to_contract_payload(AgentSpec("agent"))
    with pytest.raises(AIError) as error:
        AgentSpecCodec().from_payload({**payload, "version": 2})
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_preloaded_skills_are_eager_instructions_on_the_skill_capability() -> None:
    capability = SkillCapability(
        (_skill("z", "z-content"), _skill("a", "a-content")),
        SkillSourceRegistry(),
        preloaded_skill_ids=("z", "a"),
        max_preloaded_bytes=1024,
    )
    instructions = capability.get_instructions()
    assert instructions is not None
    assert "<preloaded-skills>" not in instructions
    assert instructions.index("[skill: a]\na-content") < instructions.index(
        "[skill: z]\nz-content"
    )


def test_preloaded_skill_instructions_reject_unpaired_surrogates() -> None:
    definition = _skill("skill", "bad\ud800content")
    capability = SkillCapability(
        (definition,),
        SkillSourceRegistry(),
        preloaded_skill_ids=(definition.id,),
    )
    with pytest.raises(AIError) as error:
        capability.get_instructions()
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID

    with pytest.raises(ValueError):
        _skill("bad\ud800id", "content")


def test_preloaded_skill_instructions_enforce_total_byte_limit() -> None:
    definition = _skill("skill", "content")
    content_size = len("[skill: skill]\ncontent".encode("utf-8"))
    capability = SkillCapability(
        (definition,),
        SkillSourceRegistry(),
        preloaded_skill_ids=("skill",),
        max_preloaded_bytes=content_size - 1,
    )
    with pytest.raises(AIError) as error:
        capability.get_instructions()
    assert error.value.code is ErrorCode.PROMPT_TOO_LARGE


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
