#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

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
    SkillResourceVersion,
    SkillSourceRef,
    tool_metadata,
)
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
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
        agent_spec=AgentSpec("agent"),
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
        event_sequence=0,
        agent_run_sequence=0,
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
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(),
        agents={"agent": AgentSpec("agent")},
    )


def _versioned_skill(
    *,
    layer_id: str,
    revision: int,
    size: int,
    etag: str = "a" * 64,
) -> SkillDefinition:
    asset = AssetVersionRef(
        AssetKey("skill", "review/guide.md"),
        layer_id,
        StorageEntryRevision(revision),
        etag,
        size,
    )
    return SkillDefinition(
        SkillSpec("review", "review instructions"),
        SkillSourceRef(
            "application",
            "review",
            (SkillResourceVersion("guide.md", asset),),
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
            resource_versions=(
                SkillResourceVersion(
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
            "etag": "a" * 64,
            "size": 1,
        },
        {
            "version": 1,
            "kind": "skill",
            "id": "review/guide.md",
            "layer_id": "source",
            "revision": "1",
            "etag": "a" * 64,
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
                    "asset_source_id": "application",
                    "root": "review",
                    "resource_versions": [
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
        AgentSpec("agent", model_route="first", revision=1)
    )
    changed = CapabilityContribution.from_declaration(
        AgentSpec("agent", model_route="second", revision=1)
    )
    revised = CapabilityContribution.from_declaration(
        AgentSpec("agent", model_route="second", revision=2)
    )

    assert first.contract != changed.contract
    assert first.revision == changed.revision
    assert first.revision != revised.revision


def test_model_contract_ignores_openai_prefix_and_connection_config() -> None:
    plain = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
    ).capture().resolve("default")
    prefixed = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
    ).capture().resolve("default")

    assert dict(plain.contract) == {
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "vision": False,
        "settings": {},
    }
    assert dict(prefixed.contract) == dict(plain.contract)
    assert plain.model_identity == "openai:gpt-test"


def test_model_registry_replaces_connection_with_same_model_contract() -> None:
    registry = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
    )
    first_snapshot = registry.capture()
    first = first_snapshot.resolve("default")

    registry.register_openai(
        "default",
        model="gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
    )
    second = registry.capture().resolve("default")

    assert second is not first
    assert dict(second.contract) == dict(first.contract)
    assert first_snapshot.resolve("default") is first


def test_agent_identity_ignores_model_route_but_catalog_uses_current_binding() -> None:
    registry = ModelRegistry()
    registry.register_openai(
        "first",
        model="gpt-test",
        base_url="https://first.example/v1",
    )
    registry.register_openai(
        "second",
        model="gpt-test",
        base_url="https://second.example/v1",
    )
    compiler = AgentCompiler(
        model_resolver=registry.capture(),
        candidates=(),
        agents={"agent": AgentSpec("agent", model_route="first")},
    )
    first = compiler.bind(compiler.compile(AgentSpec("agent", model_route="first")))
    second = compiler.bind(compiler.compile(AgentSpec("agent", model_route="second")))

    assert first.compiled_agent.spec.id == second.compiled_agent.spec.id
    assert first.compiled_agent.spec.revision == second.compiled_agent.spec.revision
    assert first.binding_digest == second.binding_digest
    assert first.binding_contract != second.binding_contract
    assert first.compiled_agent.model is not second.compiled_agent.model

    catalog = AgentCatalog({"agent": first.compiled_agent})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is second
    assert catalog.binding(first.binding_digest) is second


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


def test_restore_rejects_tool_contract_drift_without_revision_bump() -> None:
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
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(first_candidate,),
        agents={"agent": spec},
    )
    second_compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(second_candidate,),
        agents={"agent": spec},
    )
    original = first_compiler.bind(first_compiler.compile(spec))

    with pytest.raises(AIError) as raised:
        second_compiler.restore(original.binding_contract)

    assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE


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
