#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regressions for final Runtime composition invariants."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output, restore_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._approval import DefaultApprovalService
from linktools.ai.runtime._planner import _cancel_execution
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec
from pydantic import BaseModel


class _DenyAuthorization:
    async def authorize(
        self,
        principal: object,
        action: object,
        resource: object,
    ) -> None:
        del principal, action, resource
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _ExecutionHeaders:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)


class _PendingApprovals:
    def __init__(self) -> None:
        self.list_calls = 0

    async def list_pending(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        del execution_id, tenant_id
        self.list_calls += 1
        return ()


class _UncertainExecution:
    async def cancel(self, execution_id: str, request: object) -> object:
        del execution_id, request
        return SimpleNamespace(cancelled=False)

    async def inspect(self, execution_id: str, *, principal: Principal) -> object:
        del execution_id, principal
        return SimpleNamespace(status=ExecutionStatus.STARTED)


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
    )


def _execution(*, binding: AgentBindingSnapshot | None = None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    selected = binding or _binding()
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=selected.binding_digest,
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
        planning=False,
        thinking=False,
        binding=selected,
    )


def _recovery(
    *,
    binding: AgentBindingSnapshot | None = None,
) -> RecoveryExecutionInput:
    selected = binding or _binding()
    return RecoveryExecutionInput(
        user_prompt="prompt",
        principal_id="principal",
        principal_kind="service",
        session_id=None,
        memory_scope=None,
        binding_digest=selected.binding_digest,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        planning=False,
        thinking=False,
        binding=selected,
    )


@pytest.mark.asyncio
async def test_approval_list_authorizes_before_reading_pending_records() -> None:
    approvals = _PendingApprovals()
    service = DefaultApprovalService(
        approvals,
        _ExecutionHeaders(),
        _DenyAuthorization(),
    )

    with pytest.raises(AIError) as error:
        await service.list(
            "execution",
            principal=Principal("principal", "tenant", "service"),
        )

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert approvals.list_calls == 0


def test_output_contract_maps_bind_and_restore_failures() -> None:
    class LocalOutput(BaseModel):
        value: str

    with pytest.raises(AIError) as bind_error:
        bind_output(LocalOutput)
    assert bind_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    descriptor = bind_output().descriptor
    descriptor["module"] = "missing.output.module"
    with pytest.raises(AIError) as restore_error:
        restore_output(descriptor)
    assert restore_error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


@pytest.mark.parametrize(
    ("factory", "target"),
    (
        (_execution, ExecutionRecord),
        (_recovery, RecoveryExecutionInput),
    ),
)
def test_binding_codec_round_trips_mandatory_exact_v1_shape(
    factory: object,
    target: type[object],
) -> None:
    current = factory(binding=_binding())
    assert decode_domain(encode_domain(current), target) == current


@pytest.mark.parametrize(
    ("factory", "target"),
    (
        (_execution, ExecutionRecord),
        (_recovery, RecoveryExecutionInput),
    ),
)
def test_binding_codec_rejects_partial_current_v1_shapes(
    factory: object,
    target: type[object],
) -> None:
    wire = encode_domain(factory(binding=_binding()))
    assert isinstance(wire, dict)
    fields = dict(wire["fields"])
    fields.pop("binding")
    partial = dict(wire)
    partial["fields"] = fields

    with pytest.raises(AIError) as error:
        decode_domain(partial, target)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_task_child_unknown_cancel_requires_recovery() -> None:
    principal = Principal("principal", "tenant", "service")
    execution = _UncertainExecution()

    with pytest.raises(AIError) as error:
        await _cancel_execution(
            execution,
            "execution",
            principal,
            "graph",
            "node",
        )

    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_subagent_unknown_cancel_requires_recovery() -> None:
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._execution = _UncertainExecution()

    with pytest.raises(AIError) as error:
        await dispatcher.cancel_child(
            "execution",
            parent_execution_id="parent",
            principal=Principal("principal", "tenant", "service"),
        )

    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
