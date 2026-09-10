#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill preload and subagent instruction pin regressions."""

import pytest

from linktools.ai.capability import LinkToolsSkills, SkillDefinition
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec, AgentSpecCodec, SkillSpec
from linktools.ai.capability._skill_source import SkillSourceRegistry


def _skill(identity: str, content: str) -> SkillDefinition:
    return SkillDefinition(SkillSpec(identity, content))


def test_agent_spec_preload_codec_keeps_v1_and_omits_empty_field() -> None:
    codec = AgentSpecCodec()
    default = AgentSpec("agent")
    payload = codec.to_payload(default)
    assert payload["version"] == 1
    assert "preload_skills" not in payload
    assert codec.from_payload(payload).preload_skills == ()

    explicit_empty = codec.from_payload({**payload, "preload_skills": []})
    assert explicit_empty.preload_skills == ()
    assert "preload_skills" not in codec.to_payload(explicit_empty)


def test_agent_spec_preload_codec_canonicalizes_non_empty_ids() -> None:
    codec = AgentSpecCodec()
    spec = AgentSpec(
        "agent",
        allow_skills=("z", "a", "z"),
        preload_skills=("z", "a", "z"),
    )
    assert spec.preload_skills == ("a", "z")
    payload = codec.to_payload(spec)
    assert payload["version"] == 1
    assert payload["preload_skills"] == ["a", "z"]
    assert codec.from_payload(payload) == spec


def test_agent_spec_preload_rejects_invalid_type_wildcard_and_not_allowed() -> None:
    codec = AgentSpecCodec()
    base = codec.to_payload(AgentSpec("agent"))
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
    payload = AgentSpecCodec().to_payload(AgentSpec("agent"))
    with pytest.raises(AIError) as error:
        AgentSpecCodec().from_payload({**payload, "version": 2})
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_preloaded_skills_are_eager_instructions_on_the_skill_capability() -> None:
    capability = LinkToolsSkills(
        (_skill("z", "z-content"), _skill("a", "a-content")),
        SkillSourceRegistry(),
        preloaded_skill_ids=("z", "a"),
        max_preloaded_bytes=1024,
    )
    instructions = capability.instructions()
    assert instructions is not None
    assert "<preloaded-skills>" not in instructions
    assert instructions.index("[skill: a]\na-content") < instructions.index(
        "[skill: z]\nz-content"
    )


def test_preloaded_skill_instructions_reject_unpaired_surrogates() -> None:
    for definition in (
        _skill("skill", "bad\ud800content"),
        _skill("bad\ud800id", "content"),
    ):
        capability = LinkToolsSkills(
            (definition,),
            SkillSourceRegistry(),
            preloaded_skill_ids=(definition.id,),
        )
        with pytest.raises(AIError) as error:
            capability.instructions()
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_preloaded_skill_instructions_enforce_total_byte_limit() -> None:
    definition = _skill("skill", "content")
    content_size = len("[skill: skill]\ncontent".encode("utf-8"))
    capability = LinkToolsSkills(
        (definition,),
        SkillSourceRegistry(),
        preloaded_skill_ids=("skill",),
        max_preloaded_bytes=content_size - 1,
    )
    with pytest.raises(AIError) as error:
        capability.instructions()
    assert error.value.code is ErrorCode.PROMPT_TOO_LARGE


def test_skill_capability_without_skills_has_no_instructions() -> None:
    assert LinkToolsSkills((), SkillSourceRegistry()).instructions() is None
