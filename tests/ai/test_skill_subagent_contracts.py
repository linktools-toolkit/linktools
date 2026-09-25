#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Skill and Subagent declaration contracts."""

import json
from pathlib import Path

import pytest

from linktools.ai.agent import (
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    CapabilityPin,
)
from linktools.ai.capability import (
    SkillCapability,
    SkillDefinition,
    SkillSourceRegistry,
)
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    SkillMarkdownSpecAdapter,
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


def test_v1_skill_wire_and_capability_pin_round_trip() -> None:
    wire = (
        (_FIXTURES / "skill_spec_v1_golden.json")
        .read_text(encoding="utf-8")
        .strip()
        .encode("utf-8")
    )
    decoded = SkillSpecCodec().decode(wire)
    assert decoded == SkillSpec("legacy", "legacy instructions")
    assert SkillSpecCodec().encode(decoded) == wire

    fixture = _load_json("skill_capability_pin_v1_golden.json")
    pin_payload = fixture["pin"]
    assert isinstance(pin_payload, dict)
    pin = CapabilityPin.from_payload(pin_payload)
    assert pin.contract["version"] == 1
    assert pin.revision == fixture["revision"]
    assert SkillDefinition.from_contract(pin.contract).contract == {
        "version": 1,
        "id": "legacy",
        "revision": 1,
        "content": "legacy instructions",
    }


def test_v1_binding_restores_from_current_snapshot() -> None:
    payload = _load_json("agent_binding_subagent_v1_golden.json")
    snapshot = AgentBindingSnapshot.from_payload(payload)
    compiler = _compiler(
        {
            "parent": AgentSpec("parent", allow_subagents=("child",)),
            "child": AgentSpec("child", allow_subagents=()),
        }
    )

    restored = compiler.restore(snapshot)

    assert restored.binding_digest == snapshot.binding_digest
    assert restored.snapshot.subagent_ids == ("child",)
    assert restored.snapshot.subagents[0].to_payload() == {
        "kind": "agent",
        "id": "child",
        "revision": 1,
    }
    assert restored.snapshot.to_payload() == payload




def test_durable_subagent_binding_does_not_fall_back_to_current_catalog() -> None:
    agents = {
        "parent": AgentSpec("parent", allow_subagents=("child",)),
        "child": AgentSpec("child", allow_subagents=()),
    }
    compiler = _compiler(agents)
    catalog = AgentCatalog(
        {
            agent_id: compiler.compile(spec)
            for agent_id, spec in agents.items()
        }
    )
    snapshot = compiler.bind(catalog.root_agent("parent")).snapshot
    dispatcher = SubagentDispatcher(
        catalog,
        compiler,
        object(),  # type: ignore[arg-type]
    )

    with pytest.raises(AIError) as raised:
        dispatcher.delegate_for(
            parent_execution_id="parent-execution",
            root_execution_id="parent-execution",
            memory_scope=None,
            principal=Principal("principal", "tenant"),
            refs=snapshot.subagents,
            binding=snapshot,
            mode="run",
            require_frozen_bindings=True,
        )

    assert raised.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
    assert callable(
        dispatcher.delegate_for(
            parent_execution_id="parent-execution",
            root_execution_id="parent-execution",
            memory_scope=None,
            principal=Principal("principal", "tenant"),
            refs=snapshot.subagents,
            binding=snapshot,
            mode="run",
            require_frozen_bindings=False,
        )
    )


def test_future_binding_snapshot_version_is_rejected() -> None:
    payload = _load_json("agent_binding_subagent_v1_golden.json")
    payload["version"] = 2
    with pytest.raises(AIError) as raised:
        AgentBindingSnapshot.from_payload(payload)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

def test_skill_and_agent_use_v1_declaration_contracts() -> None:
    skill = SkillSpec("review", "instructions", "Review changes")
    assert SkillSpecCodec().to_payload(skill) == {
        "version": 1,
        "id": "review",
        "revision": 1,
        "content": "instructions",
        "description": "Review changes",
    }
    assert SkillSpecCodec().decode(SkillSpecCodec().encode(skill)) == skill

    plain = AgentSpec("agent")
    described = AgentSpec("agent", description="Worker")
    assert AgentSpecCodec().to_payload(plain) == AgentSpecCodec().to_payload(described)
    assert AgentSpecCodec().to_wire_payload(described)["description"] == "Worker"


def test_future_capability_pin_contract_version_is_unsupported() -> None:
    with pytest.raises(AIError) as error:
        CapabilityPin(
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


@pytest.mark.asyncio
async def test_skill_markdown_metadata_round_trips_without_changing_instructions_or_identity() -> None:
    content = (
        "---\nname: review\nmetadata:\n"
        "  author: Mei\n  version: 2\n  '': retained\n"
        "  flags: [true, null, 1.5]\n"
        "  options: {enabled: false, '': retained}\n"
        "description: Review changes\n---\n\nDo the review.\n"
    )
    codec = SkillMarkdownSpecCodec()
    local = codec.decode(content.encode("utf-8"))
    expected = {
        "author": "Mei",
        "version": 2,
        "": "retained",
        "flags": [True, None, 1.5],
        "options": {"enabled": False, "": "retained"},
    }
    assert dict(local.metadata) == expected
    assert codec.encode(local) == content.encode("utf-8")
    with pytest.raises(AIError) as mismatch:
        codec.encode(SkillSpec("review", content, "Review changes", {"author": "Other"}))
    assert mismatch.value.code is ErrorCode.ASSET_CONTENT_MISMATCH

    adapter = SkillMarkdownSpecAdapter()
    spec = adapter.to_logical("team/review", local)
    assert dict(spec.metadata) == expected
    assert adapter.to_storage("team/review", spec) == local
    wire_codec = SkillSpecCodec()
    assert wire_codec.decode(wire_codec.encode(spec)) == spec
    definition = SkillDefinition(spec)
    assert SkillDefinition.from_contract(definition.contract) == definition

    changed_metadata = SkillDefinition(
        adapter.to_logical(
            "team/review",
            codec.decode(content.replace("version: 2", "version: 3").encode()),
        )
    )
    changed_body = SkillDefinition(
        adapter.to_logical(
            "team/review",
            codec.decode(content.replace("Do the review.", "Review carefully.").encode()),
        )
    )
    without_metadata = SkillDefinition(
        adapter.to_logical(
            "team/review",
            codec.decode(
                b"---\nname: review\ndescription: Review changes\n---\n\nDo the review.\n"
            ),
        )
    )
    assert definition.model_content == changed_metadata.model_content
    assert definition.model_content == without_metadata.model_content
    assert "metadata:" not in definition.model_content
    assert "author: Mei" not in definition.model_content
    assert wire_codec.to_payload(spec)["content"] == definition.model_content
    assert (
        CapabilityPin("skill", definition.id, definition.contract).revision
        == CapabilityPin("skill", changed_metadata.id, changed_metadata.contract).revision
    )
    assert (
        CapabilityPin("skill", definition.id, definition.contract).revision
        == CapabilityPin("skill", without_metadata.id, without_metadata.contract).revision
    )
    assert (
        CapabilityPin("skill", definition.id, definition.contract).revision
        == CapabilityPin("skill", changed_body.id, changed_body.contract).revision
    )
    revised_body = SkillDefinition(
        SkillSpec(
            changed_body.spec.id,
            changed_body.spec.content,
            changed_body.spec.description,
            changed_body.spec.metadata,
            revision=2,
        )
    )
    assert (
        CapabilityPin("skill", definition.id, definition.contract).revision
        != CapabilityPin("skill", revised_body.id, revised_body.contract).revision
    )

    capability = SkillCapability(
        (definition,),
        SkillSourceRegistry(),
        preloaded_skill_ids=(definition.id,),
    )
    instructions = capability.instructions()
    assert instructions is not None
    assert "Do the review." in instructions
    assert "author: Mei" not in instructions
    root = await capability.load_skill(definition.id)
    assert root["instructions"] == definition.model_content


def test_skill_markdown_maps_reserved_revision_metadata() -> None:
    content = (
        "---\nname: review\nmetadata:\n"
        "  linktools-revision: 3\n"
        "  author: Mei\n"
        "description: Review changes\n---\n\nReview.\n"
    )
    codec = SkillMarkdownSpecCodec()
    skill = codec.decode(content.encode("utf-8"))

    assert skill.revision == 3
    assert dict(skill.metadata) == {"author": "Mei"}
    assert codec.encode(skill) == content.encode("utf-8")
    definition = SkillDefinition(skill)
    baseline = SkillDefinition(
        SkillSpec(
            skill.id,
            skill.content,
            skill.description,
            skill.metadata,
            revision=2,
        )
    )
    assert (
        CapabilityPin("skill", definition.id, definition.contract).revision
        != CapabilityPin("skill", baseline.id, baseline.contract).revision
    )


def test_skill_flow_frontmatter_metadata_does_not_change_identity() -> None:
    codec = SkillMarkdownSpecCodec()
    plain = SkillDefinition(
        codec.decode(
            b"---\n{name: review, description: Review changes}\n---\nReview.\n"
        )
    )
    annotated = SkillDefinition(
        codec.decode(
            b"---\n{name: review, description: Review changes, "
            b"metadata: {author: Mei}}\n---\nReview.\n"
        )
    )
    assert plain.model_content == annotated.model_content
    assert (
        CapabilityPin("skill", plain.id, plain.contract).revision
        == CapabilityPin("skill", annotated.id, annotated.contract).revision
    )


def test_markdown_metadata_rejects_non_json_values() -> None:
    with pytest.raises(AIError) as error:
        SkillMarkdownSpecCodec().decode(
            b"---\nname: review\ndescription: Review changes\n"
            b"metadata: {published: 2026-01-02}\n---\nReview.\n"
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
