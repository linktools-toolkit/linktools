#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable deferred-frontier approval recovery contracts."""

from datetime import datetime, timezone

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    canonical_sha256,
    idempotency_key_digest,
)
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._approval import approval_id_for_call
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._commands import RuntimeStateCommands
from linktools.ai.runtime.state._contracts import (
    ApprovalRecord,
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    PendingDeferredCall,
    PendingToolContinuation,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    StoredUserInput,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", model="default"),
        base_model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _execution(now: datetime) -> ExecutionRecord:
    binding = _binding()
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest=binding.binding_digest,
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
        binding=binding,
        principal_id="owner",
        principal_kind="user",
        stored_user_input=StoredUserInput(
            1,
            "text",
            StoredPayload.inline_text("prompt"),
        ),
    )


def _continuation() -> PendingToolContinuation:
    payload = StoredPayload.inline_json({"path": "file.txt"})
    call = PendingDeferredCall(
        "call-1",
        "read_file",
        payload,
        payload.digest,
        {"source": "workspace"},
    )
    return PendingToolContinuation(
        "step-1",
        canonical_sha256({"call_id": call.tool_call_id}),
        approvals=(call,),
    )


def _checkpoint(now: datetime) -> RecoveryCheckpoint:
    return RecoveryCheckpoint(
        execution_id="execution",
        tenant_id="tenant",
        step_run_id="step-1",
        agent_run_sequence=1,
        state=RecoveryCheckpointState.ACTIVE,
        revision=0,
        created_at=now,
        updated_at=now,
        handoff_phase=RecoveryHandoffPhase.NONE,
    )


def _approval(
    execution: ExecutionRecord,
    continuation: PendingToolContinuation,
    now: datetime,
) -> ApprovalRecord:
    pending = continuation.approvals[0]
    return ApprovalRecord(
        approval_id=approval_id_for_call(
            execution.tenant_id,
            execution.execution_id,
            continuation.source_step_run_id,
            pending.tool_call_id,
        ),
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        status=ApprovalStatus.PENDING,
        idempotency_key_digest=None,
        decision=None,
        decided_by=None,
        decision_digest=None,
        created_at=now,
        decided_at=None,
    )


def _commands(state: RuntimeState, namespace: str) -> RuntimeStateCommands:
    return RuntimeStateCommands(
        state.execution.executions,
        namespace=namespace,
        events=state.execution.events,
        operations=state.execution.operations,
        approvals=state.recovery.approvals,
        external_calls=state.recovery.external_calls,
        conversation=state.conversation.sessions,
        recovery=state.recovery.checkpoints,
        conversation_history=state.conversation.histories,
        tools=state.recovery.tools,
        conversation_steps=state.steps.read_store(RuntimeDomain.CONVERSATION),
        execution_steps=state.steps.read_store(RuntimeDomain.EXECUTION),
        recovery_steps=state.steps.read_store(RuntimeDomain.RECOVERY),
        background_tasks=set(),
    )


async def _enter_waiting(
    namespace: str,
) -> tuple[
    RuntimeState,
    RuntimeStateCommands,
    ExecutionRecord,
    RecoveryCheckpoint,
    PendingToolContinuation,
]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace=namespace, tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    checkpoint = _checkpoint(now)
    continuation = _continuation()
    await state.execution.executions.create(execution)
    await state.recovery.checkpoints.create(checkpoint)
    commands = _commands(state, namespace)
    await commands.commit_deferred_checkpoint(
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        expected_execution_revision=execution.revision,
        expected_event_sequence=execution.event_sequence,
        expected_recovery_revision=checkpoint.revision,
        expected_agent_run_sequence=execution.agent_run_sequence,
        continuation=continuation,
        approval_records=(_approval(execution, continuation, now),),
        occurred_at=now,
    )
    return state, commands, execution, checkpoint, continuation


@pytest.mark.asyncio
async def test_deferred_checkpoint_persists_approval_frontier_atomically() -> None:
    state, _, execution, _, continuation = await _enter_waiting(
        "approval-frontier"
    )
    try:
        current = await state.execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        checkpoint = await state.recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        approval = await state.recovery.approvals.get(
            approval_id_for_call(
                execution.tenant_id,
                execution.execution_id,
                continuation.source_step_run_id,
                continuation.approvals[0].tool_call_id,
            ),
            tenant_id=execution.tenant_id,
        )
        assert current is not None
        assert current.status is ExecutionStatus.WAITING_DEFERRED
        assert checkpoint is not None
        assert checkpoint.state is RecoveryCheckpointState.WAITING
        assert checkpoint.pending_tools == continuation
        assert approval is not None and approval.status is ApprovalStatus.PENDING
        events = await state.execution.events.list(
            execution.execution_id,
            tenant_id=execution.tenant_id,
            after_sequence=0,
            limit=10,
        )
        assert tuple(event.event_type for event in events.items) == (
            ExecutionEventType.APPROVAL_REQUESTED,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_deferred_resume_clears_frontier_and_advances_attempt_once() -> None:
    state, commands, execution, checkpoint, continuation = await _enter_waiting(
        "approval-resume"
    )
    try:
        approval_id = approval_id_for_call(
            execution.tenant_id,
            execution.execution_id,
            continuation.source_step_run_id,
            continuation.approvals[0].tool_call_id,
        )
        now = datetime.now(timezone.utc)
        await state.recovery.approvals.decide(
            approval_id,
            tenant_id=execution.tenant_id,
            expected_status=ApprovalStatus.PENDING,
            idempotency_key_digest=idempotency_key_digest("decision"),
            decision=ApprovalDecision.APPROVE,
            principal_id="approver",
            decision_digest=canonical_sha256({"decision": "approve"}),
            decided_at=now,
        )
        resumed, active = await commands.claim_deferred_resume_checkpoint(
            execution_id=execution.execution_id,
            tenant_id=execution.tenant_id,
            expected_execution_revision=execution.revision + 1,
            expected_event_sequence=execution.event_sequence + 1,
            expected_recovery_revision=checkpoint.revision + 1,
            expected_agent_run_sequence=execution.agent_run_sequence,
            expected_pending_tools=continuation,
        )
        assert resumed.status is ExecutionStatus.STARTED
        assert resumed.agent_run_sequence == 2
        assert active.state is RecoveryCheckpointState.ACTIVE
        assert active.pending_tools is None
        assert active.step_run_id != continuation.source_step_run_id
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_deferred_cancel_cancels_pending_approval_and_clears_frontier() -> None:
    state, commands, execution, checkpoint, continuation = await _enter_waiting(
        "approval-cancel"
    )
    try:
        current = await state.execution.executions.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        assert current is not None
        committed = await commands.commit_deferred_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                execution.execution_id,
                execution.tenant_id,
                current.revision,
                current.event_sequence,
                "cancel-operation",
                datetime.now(timezone.utc),
            ),
            expected_recovery_revision=checkpoint.revision + 1,
            expected_pending_tools=continuation,
        )
        assert committed.status is ExecutionStatus.CANCELLING
        approval = await state.recovery.approvals.get(
            approval_id_for_call(
                execution.tenant_id,
                execution.execution_id,
                continuation.source_step_run_id,
                continuation.approvals[0].tool_call_id,
            ),
            tenant_id=execution.tenant_id,
        )
        active = await state.recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        assert approval is not None and approval.status is ApprovalStatus.CANCELLED
        assert active is not None
        assert active.state is RecoveryCheckpointState.ACTIVE
        assert active.pending_tools is None
    finally:
        await state.close()
