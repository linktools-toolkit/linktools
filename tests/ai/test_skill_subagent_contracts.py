#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Skill and Subagent declaration contracts."""

import json
from pathlib import Path

import pytest

from linktools.ai.agent import AgentBindingSnapshot, AgentCompiler, SemanticPin
from linktools.ai.capability import SkillDefinition
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    SkillMarkdownSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)

_FIXTURES = Path(__file__).with_name("fixtures")


def _load_json(name: str) -> dict[str, object]:
    value = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _compiler(agents: dict[str, AgentSpec]) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(),
        agents=agents,
    )


def test_v1_skill_wire_and_semantic_pin_round_trip() -> None:
    wire = (
        (_FIXTURES / "skill_spec_v1_golden.json")
        .read_text(encoding="utf-8")
        .strip()
        .encode("utf-8")
    )
    decoded = SkillSpecCodec().decode(wire)
    assert decoded == SkillSpec("legacy", "legacy instructions")
    assert SkillSpecCodec().encode(decoded) == wire

    fixture = _load_json("skill_semantic_pin_v1_golden.json")
    pin_payload = fixture["pin"]
    assert isinstance(pin_payload, dict)
    pin = SemanticPin.from_payload(pin_payload)
    assert pin.contract["version"] == 1
    assert pin.fingerprint == fixture["fingerprint"]
    assert SkillDefinition.from_semantic_contract(pin.contract).semantic_contract == {
        "version": 1,
        "id": "legacy",
        "content": "legacy instructions",
    }


def test_v1_binding_restores_from_current_semantic_snapshot() -> None:
    payload = _load_json("agent_binding_subagent_v1_golden.json")
    snapshot = AgentBindingSnapshot.from_payload(payload)
    compiler = _compiler(
        {
            "parent": AgentSpec("parent", allow_subagents=("child",)),
            "child": AgentSpec("child", allow_subagents=()),
        }
    )

    restored = compiler.restore(snapshot)

    assert restored.digest == snapshot.binding_digest
    assert restored.snapshot.subagent_ids == ("child",)
    assert restored.snapshot.subagents[0].to_payload() == {
        "kind": "agent",
        "id": "child",
    }
    assert restored.snapshot.to_payload() == payload


def test_skill_and_agent_use_v1_declaration_contracts() -> None:
    skill = SkillSpec("review", "instructions", "Review changes")
    assert SkillSpecCodec().to_payload(skill) == {
        "version": 1,
        "id": "review",
        "content": "instructions",
        "description": "Review changes",
    }
    assert SkillSpecCodec().decode(SkillSpecCodec().encode(skill)) == skill

    plain = AgentSpec("agent")
    described = AgentSpec("agent", description="Worker")
    assert AgentSpecCodec().to_payload(plain) == AgentSpecCodec().to_payload(described)
    assert AgentSpecCodec().to_wire_payload(described)["description"] == "Worker"
    assert _compiler({"agent": plain}).compile(plain).digest == _compiler(
        {"agent": described}
    ).compile(described).digest


def test_future_semantic_pin_version_is_typed_as_unsupported() -> None:
    with pytest.raises(AIError) as error:
        SemanticPin(
            "skill",
            "review",
            {
                "version": 2,
                "id": "review",
                "content": "instructions",
            },
        )
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_skill_markdown_preserves_description_and_rejects_mismatch() -> None:
    content = "---\nname: review\ndescription: Review changes\n---\n\nDo the review.\n"
    codec = SkillMarkdownSpecCodec()
    decoded = codec.decode(content.encode("utf-8"))
    assert decoded.description == "Review changes"
    assert codec.encode(decoded) == content.encode("utf-8")

    with pytest.raises(AIError) as error:
        codec.encode(SkillSpec("review", content, "Different description"))
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH
