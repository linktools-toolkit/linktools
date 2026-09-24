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
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    SemanticPin,
)
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import (
    CapabilityContribution,
    SkillDefinition,
    SkillSourceRef,
    tool_semantic_metadata,
)
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec, SkillSpec
from linktools.ai.storage import ObjectRef, StoredPayload
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


def _snapshot() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("agent"),
        base_model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def test_binding_round_trip_preserves_nonsemantic_wire_extensions() -> None:
    original = _snapshot()
    payload = original.to_payload()
    agent_spec = dict(payload["agent_spec"])
    agent_spec["future_display_note"] = {"source": "declaration"}
    payload["agent_spec"] = agent_spec
    payload["future_binding_note"] = {"source": "envelope"}

    restored = AgentBindingSnapshot.from_payload(payload)
    written = restored.to_payload()

    assert written["agent_spec"]["future_display_note"] == {
        "source": "declaration"
    }
    assert written["future_binding_note"] == {"source": "envelope"}
    assert restored.binding_digest == original.binding_digest


def _execution(
    *,
    binding: AgentBindingSnapshot | None = None,
    planning: bool = False,
    thinking: bool = False,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    snapshot = _snapshot() if binding is None else binding
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
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
        binding=snapshot,
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
    )


def _compiler() -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(),
        agents={"agent": AgentSpec("agent")},
    )


def test_skill_snapshot_semantics_ignore_physical_store_id() -> None:
    specification = SkillSpec("review", "review instructions")
    first = SkillDefinition(
        specification,
        SkillSourceRef(
            "application",
            "review",
            ObjectRef("store-a", "skill/snapshot", "a" * 64, 1),
            "b" * 64,
        ),
    )
    second = SkillDefinition(
        specification,
        SkillSourceRef(
            "application",
            "review",
            ObjectRef("store-b", "skill/snapshot", "a" * 64, 1),
            "b" * 64,
        ),
    )

    assert first.semantic_contract == second.semantic_contract
    snapshot = first.semantic_contract["source"]["snapshot"]
    assert snapshot["store_id"] == "runtime"
    restored = SkillDefinition.from_semantic_contract(first.semantic_contract)
    assert restored.source_ref is not None
    assert restored.source_ref.snapshot is not None
    assert restored.source_ref.snapshot.store_id == "runtime"


def test_skill_snapshot_identity_ignores_integrity_size() -> None:
    specification = SkillSpec("review", "review instructions")
    first = SemanticPin(
        "skill",
        "review",
        SkillDefinition(
            specification,
            SkillSourceRef(
                "application",
                "review",
                ObjectRef("store", "skill/snapshot", "a" * 64, 1),
                "b" * 64,
            ),
        ).semantic_contract,
    )
    second = SemanticPin(
        "skill",
        "review",
        SkillDefinition(
            specification,
            SkillSourceRef(
                "application",
                "review",
                ObjectRef("store", "skill/snapshot", "a" * 64, 2),
                "b" * 64,
            ),
        ).semantic_contract,
    )

    assert first.contract != second.contract
    assert first.fingerprint == second.fingerprint


def test_skill_snapshot_reference_rejects_malformed_known_fields() -> None:
    with pytest.raises(AIError) as raised:
        SkillDefinition.from_semantic_contract(
            {
                "version": 1,
                "id": "review",
                "content": "instructions",
                "source": {
                    "source_id": "application",
                    "root": "review",
                    "resource_semantic_digest": "b" * 64,
                    "snapshot": {
                        "store_id": 1,
                        "key": "snapshot",
                        "digest": "a" * 64,
                        "size": "1",
                    },
                },
            }
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_skill_snapshot_reference_requires_store_id() -> None:
    with pytest.raises(AIError) as raised:
        SkillDefinition.from_semantic_contract(
            {
                "version": 1,
                "id": "review",
                "content": "instructions",
                "source": {
                    "source_id": "application",
                    "root": "review",
                    "resource_semantic_digest": "b" * 64,
                    "snapshot": {
                        "key": "snapshot",
                        "digest": "a" * 64,
                        "size": 1,
                    },
                },
            }
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_binding_object_dependency_scan_requires_runtime_store_id() -> None:
    pin = SemanticPin(
        "skill",
        "review",
        {
            "version": 1,
            "id": "review",
            "content": "instructions",
            "source": {
                "source_id": "application",
                "root": "review",
                "resource_semantic_digest": "b" * 64,
                "snapshot": {
                    "store_id": "runtime",
                    "key": "snapshot",
                    "digest": "a" * 64,
                    "size": 1,
                },
            },
        },
    )
    snapshot = replace(_snapshot(), selected=(pin,))

    refs = tuple(
        runtime_codec._iter_agent_binding_object_refs(
            snapshot,
            RuntimeDomain.EXECUTION,
        )
    )

    assert refs == (
        (
            RuntimeDomain.EXECUTION,
            ObjectRef("runtime", "snapshot", "a" * 64, 1),
        ),
    )


def test_agent_declaration_identity_keeps_model_selector() -> None:
    first = CapabilityContribution.from_declaration(
        AgentSpec("agent", model="first")
    )
    second = CapabilityContribution.from_declaration(
        AgentSpec("agent", model="second")
    )

    assert first.fingerprint != second.fingerprint


def test_model_semantic_identity_ignores_openai_prefix_and_connection_config() -> None:
    plain = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
    ).snapshot().resolve("default")
    prefixed = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
    ).snapshot().resolve("default")

    assert dict(plain.semantic_payload) == {
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "vision": False,
        "settings": {},
    }
    assert dict(prefixed.semantic_payload) == dict(plain.semantic_payload)
    assert plain.fingerprint == prefixed.fingerprint
    assert plain.model_identity == "openai:gpt-test"


def test_model_registry_replaces_connection_binding_with_same_semantic_identity() -> None:
    registry = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
    )
    first_snapshot = registry.snapshot()
    first = first_snapshot.resolve("default")

    registry.register_openai(
        "default",
        model="gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
    )
    second = registry.snapshot().resolve("default")

    assert second is not first
    assert second.fingerprint == first.fingerprint
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
        model_resolver=registry.snapshot(),
        candidates=(),
        agents={"agent": AgentSpec("agent", model="first")},
    )
    first = compiler.bind(compiler.compile(AgentSpec("agent", model="first")))
    second = compiler.bind(compiler.compile(AgentSpec("agent", model="second")))

    assert first.definition.digest == second.definition.digest
    assert first.digest == second.digest
    assert first.snapshot != second.snapshot
    assert first.definition.model is not second.definition.model

    catalog = AgentCatalog({"agent": first.definition})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is second
    assert catalog.definition(first.definition.digest) is first.definition
    assert catalog.binding(first.digest) is second


def test_current_binding_snapshot_has_minimal_wire_shape() -> None:
    snapshot = _snapshot()

    assert set(snapshot.to_payload()) == {
        "version",
        "agent_spec",
        "base_model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    assert snapshot.binding_digest == snapshot.binding_digest


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


def test_restore_accepts_nonsemantic_tool_contract_drift() -> None:
    def sample(value: str) -> str:
        return value

    spec = AgentSpec("agent", allow_tools=("sample",))
    semantic = tool_semantic_metadata(
        effect="none",
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
    assert first_candidate.semantic_contract != second_candidate.semantic_contract
    assert first_candidate.fingerprint == second_candidate.fingerprint

    first_compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(first_candidate,),
        agents={"agent": spec},
    )
    second_compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(second_candidate,),
        agents={"agent": spec},
    )
    original = first_compiler.bind(first_compiler.compile(spec))

    restored = second_compiler.restore(original.snapshot)

    assert restored.digest == original.digest
    assert restored.definition.selected_tools == (second_candidate,)


def test_catalog_reuses_binding_for_nonsemantic_definition_differences() -> None:
    compiler = _compiler()
    first = compiler.bind(
        compiler.compile(AgentSpec("agent", description="first label"))
    )
    second = compiler.bind(
        compiler.compile(AgentSpec("agent", description="second label"))
    )

    assert first.snapshot == second.snapshot
    assert first.digest == second.digest

    catalog = AgentCatalog({"agent": first.definition})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is first


def test_same_json_schema_produces_same_binding_identity() -> None:
    compiler = _compiler()
    definition = compiler.compile(AgentSpec("agent"))
    first = compiler.bind(definition, output=_SchemaTwinA)
    second = compiler.bind(definition, output=_SchemaTwinB)

    assert first.digest == second.digest
    assert first.snapshot == second.snapshot
    assert first.output_binding.schema_definition == second.output_binding.schema_definition

    catalog = AgentCatalog({"agent": definition})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is first
    assert catalog.binding(first.digest) is first


def test_restored_binding_uses_only_snapshot_semantics() -> None:
    compiler = _compiler()
    definition = compiler.compile(AgentSpec("agent"))
    current = compiler.bind(definition, output=_SchemaTwinA)

    restored = compiler.restore(current.snapshot)

    assert restored.digest == current.digest
    assert restored.snapshot == current.snapshot
    assert restored.output_binding.schema_definition == current.output_binding.schema_definition
    assert restored.output_type is not _SchemaTwinA


def test_binding_rejects_selected_definition_snapshot_mismatch() -> None:
    compiler = _compiler()
    binding = compiler.bind(compiler.compile(AgentSpec("agent")))
    mismatched_definition = replace(
        binding.definition,
        selected_tools=(
            SimpleNamespace(
                kind="tool",
                id="unexpected-tool",
                semantic_contract={"version": 1},
            ),
        ),
    )

    with pytest.raises(AIError) as raised:
        AgentBinding(
            mismatched_definition,
            binding.output_binding,
            binding.snapshot,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_binding_preserves_selected_pin_version_error() -> None:
    compiler = _compiler()
    binding = compiler.bind(compiler.compile(AgentSpec("agent")))
    invalid_definition = replace(
        binding.definition,
        selected_tools=(
            SimpleNamespace(
                kind="tool",
                id="future-tool",
                semantic_contract={"version": 2},
            ),
        ),
    )

    with pytest.raises(AIError) as raised:
        AgentBinding(
            invalid_definition,
            binding.output_binding,
            binding.snapshot,
        )
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_execution_binding_digest_is_derived_from_snapshot() -> None:
    value = _execution(planning=True, thinking=True)
    assert value.binding_digest == value.binding.binding_digest
