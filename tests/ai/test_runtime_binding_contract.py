#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

from datetime import datetime, timezone
from typing import Annotated

import pytest
from linktools.ai.agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec
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


def _snapshot(*, binding_digest: str = "a" * 64) -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest=binding_digest,
    )


def _execution(
    *,
    binding_digest: str = "a" * 64,
    binding: AgentBindingSnapshot | None = None,
    planning: bool = False,
    thinking: bool = False,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=binding_digest,
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
        binding=_snapshot(binding_digest=binding_digest) if binding is None else binding,
    )


def _recovery(
    *,
    binding_digest: str = "a" * 64,
    binding: AgentBindingSnapshot | None = None,
    planning: bool = False,
    thinking: bool = False,
) -> RecoveryExecutionInput:
    return RecoveryExecutionInput(
        user_prompt="prompt",
        user_prompt_codec="text",
        principal_id="principal",
        principal_kind="service",
        session_id=None,
        memory_scope=None,
        binding_digest=binding_digest,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        mode="run",
        planning=planning,
        thinking=thinking,
        binding=_snapshot(binding_digest=binding_digest) if binding is None else binding,
    )


def _compiler() -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(),
        agent_ids=("agent",),
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


def test_current_binding_snapshot_persists_only_v1_semantic_inputs() -> None:
    output = bind_output()
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )

    assert set(snapshot.to_payload()) == {
        "version",
        "agent_spec",
        "model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
        "binding_digest",
    }


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


def test_execution_requires_exact_binding_snapshot() -> None:
    value = _execution(planning=True, thinking=True)
    assert value.binding.binding_digest == value.binding_digest
    with pytest.raises(ValueError, match="execution binding snapshot"):
        _execution(binding_digest="c" * 64, binding=_snapshot(binding_digest="d" * 64))


def test_recovery_requires_exact_binding_snapshot() -> None:
    value = _recovery(planning=True, thinking=True)
    assert value.binding.binding_digest == value.binding_digest
    with pytest.raises(ValueError, match="recovery binding snapshot"):
        _recovery(binding_digest="c" * 64, binding=_snapshot(binding_digest="d" * 64))
