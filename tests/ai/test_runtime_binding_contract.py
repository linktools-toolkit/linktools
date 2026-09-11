#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

from datetime import datetime, timezone
from typing import Annotated

import pytest
from linktools.ai.agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload
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
        version=1,
        agent_spec=AgentSpec("agent"),
        base_model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


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
        tenant_id="tenant",
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
            1,
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

    assert dict(plain.semantic_payload) == dict(prefixed.semantic_payload)
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


def test_current_binding_snapshot_persists_only_semantic_inputs() -> None:
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


def test_execution_binding_digest_is_derived_from_snapshot() -> None:
    value = _execution(planning=True, thinking=True)
    assert value.binding_digest == value.binding.binding_digest
