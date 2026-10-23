#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.model import create_model

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Annotated

import pytest
from linktools.ai.agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    CapabilityPin,
)
from linktools.ai.agent._output import bind_output
from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import (
    CapabilityContribution,
    SkillDefinition,
    SkillResource,
    SkillSourceRef,
    tool_metadata,
)
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec, SkillSpec
from linktools.ai.storage import StorageEntryRevision, StoredPayload
from pydantic_ai import Tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)


class _PydanticOutput(BaseModel):
    value: int
    evidence: list[str] = Field(default_factory=list)


class _PythonValidatedOutput(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def reject_seven(cls, value: int) -> int:
        if value == 7:
            raise ValueError("python-only rule")
        return value


class _SchemaTwinA(BaseModel):
    model_config = ConfigDict(title="SharedOutput")
    value: Annotated[int, Field(ge=0)]

    @field_validator("value")
    @classmethod
    def require_even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("value must be even")
        return value


class _SchemaTwinB(BaseModel):
    model_config = ConfigDict(title="SharedOutput")
    value: Annotated[int, Field(ge=0)]

    @field_validator("value")
    @classmethod
    def require_odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("value must be odd")
        return value


def _binding_contract() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("agent", tool_retries=3, output_retries=3),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def test_binding_round_trip_preserves_nonsemantic_wire_extensions() -> None:
    original = _binding_contract()
    payload = original.to_payload()
    agent_spec = dict(payload["agent_spec"])
    agent_spec["future_display_note"] = {"source": "declaration"}
    payload["agent_spec"] = agent_spec
    payload["future_binding_note"] = {"source": "envelope"}

    restored = AgentBindingContract.from_payload(payload)
    written = restored.to_payload()

    assert written["agent_spec"]["future_display_note"] == {
        "source": "declaration"
    }
    assert written["future_binding_note"] == {"source": "envelope"}
    assert restored.binding_digest == original.binding_digest


def _execution(
    *,
    binding: AgentBindingContract | None = None,
    planning: bool = False,
    thinking: bool = False,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    binding_contract = _binding_contract() if binding is None else binding
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_seq=0,
        agent_run_seq=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=planning,
        thinking=thinking,
        binding=binding_contract,
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
    )


def _compiler() -> AgentCompiler:
    return AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")},
        candidates=(),
        agents={"agent": AgentSpec("agent")},
    )


def _versioned_skill(
    *,
    layer_id: str,
    revision: int,
    size: int,
    digest: str = "a" * 64,
) -> SkillDefinition:
    asset = AssetVersionRef(
        AssetKey("skill", "review/guide.md"),
        layer_id,
        StorageEntryRevision(revision),
        digest,
        size,
    )
    return SkillDefinition(
        SkillSpec("review", "review instructions"),
        SkillSourceRef(
            "application",
            "review",
            (SkillResource("guide.md", asset),),
        ),
    )


def test_skill_asset_version_locator_is_not_named_identity() -> None:
    first = _versioned_skill(layer_id="source-a", revision=1, size=1)
    second = _versioned_skill(layer_id="source-b", revision=9, size=99)

    assert first.contract != second.contract
    first_pin = CapabilityPin("skill", "review", first.contract)
    second_pin = CapabilityPin("skill", "review", second.contract)
    assert first_pin.revision == second_pin.revision

    restored = SkillDefinition.from_contract(first.contract)
    assert restored == first


def test_skill_asset_content_change_requires_revision_bump() -> None:
    first = _versioned_skill(layer_id="source", revision=1, size=1)
    assert first.source_ref is not None
    changed = SkillDefinition(
        first.spec,
        replace(
            first.source_ref,
            resources=(
                SkillResource(
                    "guide.md",
                    AssetVersionRef(
                        AssetKey("skill", "review/guide.md"),
                        "source",
                        StorageEntryRevision(2),
                        "c" * 64,
                        1,
                    ),
                ),
            ),
        ),
    )
    revised = SkillDefinition(
        SkillSpec(
            first.spec.id,
            first.spec.content,
            first.spec.description,
            first.spec.metadata,
            revision=2,
        ),
        changed.source_ref,
    )

    first_pin = CapabilityPin("skill", "review", first.contract)
    changed_pin = CapabilityPin("skill", "review", changed.contract)
    revised_pin = CapabilityPin("skill", "review", revised.contract)
    assert first.contract != changed.contract
    assert first_pin.revision == changed_pin.revision
    assert first_pin.revision != revised_pin.revision


@pytest.mark.parametrize(
    "asset",
    (
        {
            "version": 1,
            "kind": "skill",
            "id": "review/guide.md",
            "layer_id": "",
            "revision": 1,
            "digest": "a" * 64,
            "size": 1,
        },
        {
            "version": 1,
            "kind": "skill",
            "id": "review/guide.md",
            "layer_id": "source",
            "revision": "1",
            "digest": "a" * 64,
            "size": 1,
        },
    ),
)
def test_skill_asset_version_reference_rejects_malformed_fields(
    asset: dict[str, object],
) -> None:
    with pytest.raises(AIError) as raised:
        SkillDefinition.from_contract(
            {
                "version": 1,
                "id": "review",
                "content": "instructions",
                "source": {
                    "source_id": "application",
                    "root": "review",
                    "resources": [
                        {
                            "path": "guide.md",
                            "asset": asset,
                            "executable_bits": 0,
                        }
                    ],
                },
            }
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_binding_asset_versions_are_not_runtime_object_dependencies() -> None:
    pin = CapabilityPin(
        "skill",
        "review",
        _versioned_skill(
            layer_id="source",
            revision=1,
            size=1,
        ).contract,
    )
    binding_contract = replace(_binding_contract(), selected=(pin,))

    assert tuple(
        runtime_codec.iter_runtime_object_refs(
            runtime_codec._encode_persisted_domain(_execution(binding=binding_contract)),
            default_domain=RuntimeDomain.EXECUTION,
        )
    ) == ()


def test_agent_declaration_identity_uses_explicit_revision() -> None:
    first = CapabilityContribution.from_declaration(
        AgentSpec("agent", model="first", revision=1)
    )
    changed = CapabilityContribution.from_declaration(
        AgentSpec("agent", model="second", revision=1)
    )
    revised = CapabilityContribution.from_declaration(
        AgentSpec("agent", model="second", revision=2)
    )

    assert first.contract != changed.contract
    assert first.revision == changed.revision
    assert first.revision != revised.revision


def test_model_contract_ignores_openai_prefix_and_connection_config() -> None:
    plain = create_model(protocol='openai',
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
    )
    prefixed = create_model(protocol='openai',
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
    )

    assert dict(plain.contract) == {
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "input_mime_types": [],
        "settings": {},
    }
    assert dict(prefixed.contract) == dict(plain.contract)
    assert plain.model_identity == "openai:gpt-test"


@pytest.mark.asyncio
async def test_feature_capture_keeps_replaced_model_connection() -> None:
    features = RuntimeFeatures()
    features.model("default", create_model(
        protocol="openai", model="gpt-test",
        base_url="https://first.example/v1", api_key="first-key",
    ))
    first_capture = await features.capture()
    first = first_capture.models["default"]

    features.model("default", create_model(
        protocol="openai", model="gpt-test",
        base_url="https://second.example/v1", api_key="second-key",
    ))
    second = (await features.capture()).models["default"]

    assert second is not first
    assert dict(second.contract) == dict(first.contract)
    assert first_capture.models["default"] is first


def test_agent_identity_ignores_model_but_catalog_uses_current_binding() -> None:
    models = {
        "first": create_model(protocol="openai", model="gpt-test", base_url="https://first.example/v1"),
        "second": create_model(protocol="openai", model="gpt-test", base_url="https://second.example/v1"),
    }
    compiler = AgentCompiler(
        models=models,
        candidates=(),
        agents={"agent": AgentSpec("agent", model="first")},
    )
    first = compiler.bind(compiler.compile(AgentSpec("agent", model="first")))
    second = compiler.bind(compiler.compile(AgentSpec("agent", model="second")))

    assert first.compiled_agent.spec.id == second.compiled_agent.spec.id
    assert first.compiled_agent.spec.revision == second.compiled_agent.spec.revision
    assert first.binding_digest != second.binding_digest
    assert first.binding_contract != second.binding_contract
    assert first.compiled_agent.model is not second.compiled_agent.model

    catalog = AgentCatalog({"agent": first.compiled_agent})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is second
    assert catalog.binding(first.binding_digest) is first
    assert catalog.binding(second.binding_digest) is second


def test_current_binding_contract_has_minimal_wire_shape() -> None:
    binding_contract = _binding_contract()

    assert set(binding_contract.to_payload()) == {
        "version",
        "agent_spec",
        "model_contract",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    assert len(binding_contract.binding_digest) == 64


def test_custom_output_materializes_from_durable_json_schema() -> None:
    binding = bind_output(_PydanticOutput)
    assert binding.mode == "structured"
    assert binding.runtime_output_type is not _PydanticOutput

    parsed = TypeAdapter(binding.runtime_output_type).validate_python(
        {"value": 50, "evidence": []}
    )
    assert parsed == {"value": 50, "evidence": []}

    with pytest.raises(ValidationError):
        TypeAdapter(binding.runtime_output_type).validate_python(
            {"value": "50", "evidence": []}
        )


def test_python_only_output_validator_is_not_part_of_durable_contract() -> None:
    binding = bind_output(_PythonValidatedOutput)

    parsed = TypeAdapter(binding.runtime_output_type).validate_python({"value": 7})

    assert parsed == {"value": 7}


def test_restore_accepts_tool_configuration_changes_with_same_revision() -> None:
    def sample(value: str) -> str:
        return value

    spec = AgentSpec("agent", allow_tools=("sample",))
    semantic = tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="business",
    )
    first_candidate = CapabilityContribution.from_opaque(
        "tool",
        "sample",
        Tool(
            sample,
            name="sample",
            metadata={**semantic, "upstream.trace": "first"},
        ),
    )
    second_candidate = CapabilityContribution.from_opaque(
        "tool",
        "sample",
        Tool(
            sample,
            name="sample",
            metadata={**semantic, "upstream.trace": "second"},
        ),
    )
    assert first_candidate.contract != second_candidate.contract
    assert first_candidate.revision == second_candidate.revision

    first_compiler = AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")},
        candidates=(first_candidate,),
        agents={"agent": spec},
    )
    second_compiler = AgentCompiler(
        models={"default": create_model(protocol='openai', model="gpt-test")},
        candidates=(second_candidate,),
        agents={"agent": spec},
    )
    original = first_compiler.bind(first_compiler.compile(spec))

    restored = second_compiler.restore(original.binding_contract)
    assert restored.binding_contract is original.binding_contract
    assert restored.binding_digest == original.binding_digest
    assert restored.compiled_agent.selected_tools[0].contract == second_candidate.contract


def test_catalog_reuses_binding_for_unchanged_compiled_semantics() -> None:
    compiler = _compiler()
    first = compiler.bind(
        compiler.compile(AgentSpec("agent", description="first label"))
    )
    second = compiler.bind(
        compiler.compile(AgentSpec("agent", description="second label"))
    )

    assert first.binding_contract == second.binding_contract
    assert first.binding_digest == second.binding_digest

    catalog = AgentCatalog({"agent": first.compiled_agent})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is first


def test_same_json_schema_produces_same_binding_identity() -> None:
    compiler = _compiler()
    compiled_agent = compiler.compile(AgentSpec("agent"))
    first = compiler.bind(compiled_agent, output=_SchemaTwinA)
    second = compiler.bind(compiled_agent, output=_SchemaTwinB)

    assert first.binding_digest == second.binding_digest
    assert first.binding_contract == second.binding_contract
    assert first.output_binding.schema_definition == second.output_binding.schema_definition

    catalog = AgentCatalog({"agent": compiled_agent})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is first
    assert catalog.binding(first.binding_digest) is first


def test_restored_binding_uses_only_contract_semantics() -> None:
    compiler = _compiler()
    compiled_agent = compiler.compile(AgentSpec("agent"))
    current = compiler.bind(compiled_agent, output=_SchemaTwinA)

    restored = compiler.restore(current.binding_contract)

    assert restored.binding_digest == current.binding_digest
    assert restored.binding_contract == current.binding_contract
    assert restored.output_binding.schema_definition == current.output_binding.schema_definition
    assert restored.output_type is not _SchemaTwinA


def test_binding_rejects_selected_compiled_agent_contract_mismatch() -> None:
    compiler = _compiler()
    binding = compiler.bind(compiler.compile(AgentSpec("agent")))
    mismatched_compiled_agent = replace(
        binding.compiled_agent,
        selected_tools=(
            SimpleNamespace(
                kind="tool",
                id="unexpected-tool",
                contract={"version": 1},
            ),
        ),
    )

    with pytest.raises(AIError) as raised:
        AgentBinding(
            mismatched_compiled_agent,
            binding.output_binding,
            binding.binding_contract,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_binding_preserves_selected_pin_version_error() -> None:
    compiler = _compiler()
    binding = compiler.bind(compiler.compile(AgentSpec("agent")))
    invalid_compiled_agent = replace(
        binding.compiled_agent,
        selected_tools=(
            SimpleNamespace(
                kind="tool",
                id="future-tool",
                contract={"version": 2},
            ),
        ),
    )

    with pytest.raises(AIError) as raised:
        AgentBinding(
            invalid_compiled_agent,
            binding.output_binding,
            binding.binding_contract,
        )
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_execution_binding_digest_is_derived_from_binding_contract() -> None:
    value = _execution(planning=True, thinking=True)
    assert value.binding_digest == value.binding.binding_digest


def test_model_diagnostics_do_not_affect_contract_equality_or_other_guards() -> None:
    original = _binding_contract()
    current = replace(original, model_contract={"api": "anthropic", "model_identity": "other", "profile": {"supports_thinking": True}})
    assert original == current
    assert original.binding_digest == current.binding_digest
    assert original.to_payload() != current.to_payload()
    assert original != replace(current, agent_spec=replace(current.agent_spec, model="other-route"))
    assert original != replace(current, agent_spec=replace(current.agent_spec, instructions=("changed",)))
    assert original != replace(current, selected=(CapabilityPin("tool", "different", {"version": 1, "revision": 1}),))
    assert original != replace(current, output_mode="structured", output_schema={"type": "object"})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_contract", []),
        ("model_contract", {"diagnostic": object()}),
        ("output_schema", []),
        ("output_schema", {"type": object()}),
        ("output_mode", []),
        ("selected", {}),
        ("subagents", {}),
    ),
)
def test_binding_reader_rejects_malformed_known_fields(field: str, value: object) -> None:
    payload = _binding_contract().to_payload()
    payload[field] = value
    with pytest.raises(AIError) as raised:
        AgentBindingContract.from_payload(payload)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", []),
        ("id", 1),
        ("contract", []),
        ("contract", {"version": 1, "revision": True}),
        ("contract", {"version": 1, "revision": 0}),
        ("contract", {"version": 1, "extra": object()}),
    ),
)
def test_capability_pin_reader_rejects_malformed_known_fields(field: str, value: object) -> None:
    payload = {"kind": "tool", "id": "sample", "contract": {"version": 1, "revision": 2}}
    payload[field] = value
    with pytest.raises(AIError) as raised:
        CapabilityPin.from_payload(payload)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_binding_reader_detaches_mutable_contract_inputs() -> None:
    payload = _binding_contract().to_payload()
    payload["model_contract"] = {"diagnostic": ["original"]}
    payload["selected"] = [{"kind": "tool", "id": "sample", "contract": {"version": 1, "revision": 2, "metadata": ["original"]}}]
    restored = AgentBindingContract.from_payload(payload)
    payload["model_contract"]["diagnostic"].append("changed")
    payload["selected"][0]["contract"]["metadata"].append("changed")
    assert restored.model_contract["diagnostic"] == ["original"]
    assert restored.selected[0].contract["metadata"] == ["original"]
    assert restored.selected[0].revision == 2


def test_subagent_reader_preserves_description_validation() -> None:
    from linktools.ai.spec import SubagentRef

    with pytest.raises(AIError) as raised:
        SubagentRef.from_payload({"kind": "agent", "id": "child", "description": []})
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def _configured_contribution(kind: str, identity: str, revision: int, setting: str) -> CapabilityContribution[object]:
    from pydantic_ai.capabilities import AbstractCapability
    from linktools.ai.spec import MCPServerSpec

    if kind == "skill":
        return CapabilityContribution.from_declaration(
            SkillDefinition(SkillSpec(identity, setting, revision=revision))
        )
    if kind == "mcp":
        return CapabilityContribution.from_declaration(
            MCPServerSpec(identity, "python", (setting,), revision=revision)
        )
    if kind == "capability":
        class ConfiguredCapability(AbstractCapability[object]):
            id = identity

        return CapabilityContribution.from_opaque(
            "capability", identity, ConfiguredCapability(),
            revision=revision, config={"setting": setting},
        )

    def sample(value: str) -> str:
        return value

    return CapabilityContribution.from_opaque(
        "tool", identity,
        Tool(sample, name=identity, description=setting, metadata=tool_metadata(
            effect_policy="none", plan_safe=True, tool_class="business",
        )),
        revision=revision,
    )


def _contribution_compiler(candidate: CapabilityContribution[object]) -> AgentCompiler:
    return AgentCompiler(
        models={"default": create_model(protocol="openai", model="gpt-test")},
        candidates=(candidate,), agents={"agent": AgentSpec("agent")},
    )


@pytest.mark.parametrize("kind", ("tool", "skill", "mcp", "capability"))
def test_capability_binding_identity_ignores_configuration_but_preserves_named_reference(kind: str) -> None:
    original_candidate = _configured_contribution(kind, "sample", 1, "original")
    changed_candidate = _configured_contribution(kind, "sample", 1, "changed")
    original_compiler = _contribution_compiler(original_candidate)
    changed_compiler = _contribution_compiler(changed_candidate)
    spec = AgentSpec("agent")
    original = original_compiler.bind(original_compiler.compile(spec))
    changed = changed_compiler.bind(changed_compiler.compile(spec))
    assert original_candidate.contract != changed_candidate.contract
    assert original.binding_digest == changed.binding_digest
    accepted = AgentBinding(changed.compiled_agent, original.output_binding, original.binding_contract)
    assert accepted.binding_digest == original.binding_digest
    restored = changed_compiler.restore(original.binding_contract)
    assert restored.binding_contract is original.binding_contract
    assert restored.binding_digest == original.binding_digest
    if kind == "skill":
        assert restored.compiled_agent.selected_skills[0].contract == original_candidate.contract

    for identity, revision in (("different", 1), ("sample", 2)):
        other_compiler = _contribution_compiler(_configured_contribution(kind, identity, revision, "changed"))
        other = other_compiler.bind(other_compiler.compile(spec))
        assert other.binding_digest != original.binding_digest
        with pytest.raises(AIError) as raised:
            AgentBinding(other.compiled_agent, original.output_binding, original.binding_contract)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        if kind != "skill":
            with pytest.raises(AIError) as raised:
                other_compiler.restore(original.binding_contract)
            assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE


@pytest.mark.asyncio
async def test_mcp_resolution_keeps_historical_contract_when_current_arguments_change() -> None:
    from linktools.ai.runtime._agent_binding_resolver import _AgentBindingResolver

    original_compiler = _contribution_compiler(_configured_contribution("mcp", "sample", 1, "original"))
    changed_compiler = _contribution_compiler(_configured_contribution("mcp", "sample", 1, "changed"))
    spec = AgentSpec("agent")
    original_catalog = AgentCatalog({"agent": original_compiler.compile(spec)})
    changed_catalog = AgentCatalog({"agent": changed_compiler.compile(spec)})
    original = await _AgentBindingResolver(original_catalog, original_compiler).resolve_root("agent")
    resolved = await _AgentBindingResolver(changed_catalog, changed_compiler).resolve_contract(original)
    assert resolved == original
    assert resolved.selected[0].contract["args"] == ["original"]
    assert changed_compiler.restore(resolved).compiled_agent.mcp_servers[0].args == ("changed",)
