#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

from datetime import datetime, timezone
from typing import Annotated

import pytest
from linktools.ai.agent import AgentBindingSnapshot, AgentCompiler
from linktools.ai.agent._catalog import AgentCatalog
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
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest=binding_digest,
        global_runtime_capability_descriptors=(),
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
        planning=planning,
        thinking=thinking,
        binding=_snapshot(binding_digest=binding_digest) if binding is None else binding,
    )


def _compiler() -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )


def test_current_binding_snapshot_persists_schema_without_python_path() -> None:
    output = bind_output()
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
        output_schema_definition=output.schema_definition,
    )
    payload = snapshot.to_payload()
    assert set(payload) == {
        "version",
        "agent_spec",
        "agent_digest",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
        "global_runtime_capability_descriptors",
        "output_schema_definition",
    }


def test_custom_output_keeps_pydantic_runtime_semantics() -> None:
    binding = bind_output(_PydanticOutput)
    assert binding.runtime_output_type is _PydanticOutput
    parsed = TypeAdapter(binding.runtime_output_type).validate_python({"value": "50"})
    assert isinstance(parsed, _PydanticOutput)
    assert parsed.value == 50
    assert parsed.evidence == []


def test_custom_output_keeps_python_validator_semantics() -> None:
    binding = bind_output(_PythonValidatedOutput)
    with pytest.raises(ValidationError):
        TypeAdapter(binding.runtime_output_type).validate_python({"value": 7})


def test_catalog_rejects_same_durable_schema_with_different_python_semantics() -> None:
    compiler = _compiler()
    definition = compiler.compile(AgentSpec("agent", model="default"))
    first = compiler.bind(definition, output=_SchemaTwinA)
    second = compiler.bind(definition, output=_SchemaTwinB)
    assert first.digest == second.digest
    assert first.output_binding.schema_definition == second.output_binding.schema_definition

    catalog = AgentCatalog({})
    assert catalog.register_binding(first) is first
    with pytest.raises(AIError) as error:
        catalog.register_binding(second)
    assert error.value.code is ErrorCode.BINDING_CONFLICT


def test_catalog_upgrades_restored_schema_binding_to_current_python_type() -> None:
    compiler = _compiler()
    definition = compiler.compile(AgentSpec("agent", model="default"))
    current = compiler.bind(definition, output=_SchemaTwinA)
    restored = compiler.restore(current.snapshot)
    assert restored.output_type is not _SchemaTwinA

    catalog = AgentCatalog({})
    assert catalog.register_binding(restored) is restored
    assert catalog.register_binding(current) is current
    assert catalog.binding(current.digest).output_type is _SchemaTwinA


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
