#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused durable recovery command regressions."""

from datetime import datetime, timezone

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import ToolEffectNotAppliedError
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    ToolOperationStatus,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._recovery_commands import RuntimeRecoveryCommands
from linktools.ai.spec import AgentSpec
from pydantic_ai.exceptions import ToolFailed


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _execution(now: datetime) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
    )


def _tool(now: datetime, *, operation_id: str = "tool-operation") -> ToolOperationRecord:
    return ToolOperationRecord(
        tool_operation_id=operation_id,
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step-run",
        tool_call_id=f"call:{operation_id}",
        idempotency_key_digest=canonical_sha256({"operation": operation_id}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": operation_id}),
        binding_digest="a" * 64,
        replay_safe=False,
        status=ToolOperationStatus.EFFECT_UNKNOWN,
        owner="worker",
        fence=3,
        lease_expires_at=None,
        error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        created_at=now,
        updated_at=now,
    )


def _commands(state: RuntimeState) -> RuntimeRecoveryCommands:
    return RuntimeRecoveryCommands(
        state.execution.executions,
        state.execution.events,
        state.recovery.operations,
        state.recovery.tools,
        execution_operations=state.execution.operations,
        background_tasks=set(),
    )


@pytest.mark.asyncio
async def test_recovery_status_and_resume_are_durable_nonterminal_events() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="execution-recovery", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    try:
        await state.execution.executions.create(execution)
        commands = _commands(state)
        recovery = await commands.commit_recovery_required(
            execution,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            safe_error_details={"operation_id": "tool-operation", "fence": 3},
        )
        assert recovery.status is ExecutionStatus.RECOVERY_REQUIRED
        assert recovery.result is None
        assert recovery.error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
        assert recovery.error_diagnostics is None

        resumed = await commands.commit_resumed(recovery)
        assert resumed.status is ExecutionStatus.STARTED
        assert resumed.error_code is None
        assert resumed.safe_error_details == {}
        assert resumed.result is None

        events = await state.execution.events.list(
            execution.execution_id,
            tenant_id=execution.tenant_id,
            after_sequence=0,
            limit=10,
        )
        assert tuple(value.event_type for value in events.items) == (
            ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
            ExecutionEventType.EXECUTION_RESUMED,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recovery_cancel_intent_fences_stale_resume() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-cancel", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    try:
        await state.execution.executions.create(execution)
        commands = _commands(state)
        recovery = await commands.commit_recovery_required(
            execution,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            safe_error_details={"operation_id": "tool-operation", "fence": 3},
        )
        operation = OperationLedgerInput(
            "cancel-operation",
            "tenant",
            ResourceKind.EXECUTION,
            "execution",
            "execution",
            OperationKind.EXECUTION_CANCEL,
            OperationStatus.PENDING,
            canonical_sha256({"cancel": True}),
            None,
            None,
            None,
            True,
            now,
            now,
        )
        await commands.commit_cancel_intent(recovery, operation)
        fresh = await state.execution.executions.get(
            "execution",
            tenant_id="tenant",
        )
        assert fresh is not None
        assert fresh.status is ExecutionStatus.RECOVERY_REQUIRED
        assert fresh.revision == recovery.revision + 1
        assert fresh.event_sequence == recovery.event_sequence + 1

        with pytest.raises(AIError) as stale:
            await commands.commit_resumed(recovery)
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT

        resumed = await commands.commit_resumed(fresh)
        cancelling = await commands.commit_cancel_claim(resumed)
        assert cancelling.status is ExecutionStatus.CANCELLING
        assert cancelling.event_sequence == resumed.event_sequence

        events = await state.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_sequence=0,
            limit=10,
        )
        assert tuple(value.event_type for value in events.items) == (
            ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
            ExecutionEventType.CANCEL_REQUESTED,
            ExecutionEventType.EXECUTION_RESUMED,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_not_applied_resolution_reopens_tool_with_next_fence() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="tool-resolution", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    try:
        await state.recovery.tools.reserve(_tool(now))
        operation = OperationLedgerInput(
            "resolution-operation",
            "tenant",
            ResourceKind.TOOL_OPERATION,
            "tool-operation",
            "execution",
            OperationKind.TOOL_EFFECT_RESOLVE,
            OperationStatus.SUCCEEDED,
            canonical_sha256({"resolution": "not-applied"}),
            "tool-operation",
            canonical_sha256({"result": "pending"}),
            None,
            True,
            now,
            now,
        )
        resolved = await _commands(state).resolve_tool_effect(
            operation,
            expected_fence=3,
            target_status=ToolOperationStatus.PENDING,
        )
        assert resolved.status is ToolOperationStatus.PENDING
        assert resolved.owner is None
        assert resolved.fence == 3

        claimed = await state.recovery.tools.claim(
            "tool-operation",
            tenant_id="tenant",
            owner="next-worker",
            lease_seconds=60,
        )
        assert claimed.status is ToolOperationStatus.CLAIMED
        assert claimed.fence == 4
        assert claimed.owner == "next-worker"
    finally:
        await state.close()


def test_explicit_not_applied_marker_is_a_known_tool_failure() -> None:
    assert issubclass(ToolEffectNotAppliedError, ToolFailed)
