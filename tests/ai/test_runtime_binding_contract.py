#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec


def _snapshot(*, binding_digest: str = "a" * 64) -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
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
