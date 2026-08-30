#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval waiting, decision, resume, and cancellation recovery contracts."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    ApprovalDecisionRequest,
    DefaultApprovalService,
    RuntimeState,
)
from linktools.ai.runtime.state import (
    PendingApprovalContinuation,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RuntimeDomain,
    RuntimeStateCommands,
    ToolApprovalAdmission,
)
from linktools.ai.runtime.state._contracts import (
    ApprovalRecord,
    ExecutionCancelRequestCommit,
    ExecutionRecord,
    RecoveryCheckpoint,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _execution(now: datetime, *, agent_run_sequence: int = 1) -> ExecutionRecord:
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
        agent_run_sequence=agent_run_sequence,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
    )


def _recovery_input() -> RecoveryExecutionInput:
    return RecoveryExecutionInput(
        user_prompt="prompt",
        user_prompt_codec="text",
        principal_id="owner",
        principal_kind="user",
        session_id=None,
        memory_scope=None,
        binding_digest="a" * 64,
        lineage_kind="run",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "digest"),
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
    )


def _commands(state: RuntimeState, namespace: str) -> RuntimeStateCommands:
    return RuntimeStateCommands(
        state.execution.executions,
        namespace=namespace,
        events=state.execution.events,
        operations=state.execution.operations,
        conversation=state.conversation.sessions,
        recovery=state.recovery.checkpoints,
        approvals=state.recovery.approvals,
        conversation_history=state.conversation.histories,
        tools=state.recovery.tools,
        conversation_steps=state.steps.read_store(RuntimeDomain.CONVERSATION),
        execution_steps=state.steps.read_store(RuntimeDomain.EXECUTION),
        recovery_steps=state.steps.read_store(RuntimeDomain.RECOVERY),
        background_tasks=set(),
    )


def _batch_id(source_step_run_id: str, approval_ids: tuple[str, ...]) -> str:
    return canonical_sha256(
        {
            "contract": "tool-approval-batch-v1",
            "execution_id": "execution",
            "source_step_run_id": source_step_run_id,
            "approval_ids": sorted(approval_ids),
        }
    )


def _admission(
    approval_id: str,
    tool_name: str,
    args: object,
    *,
    now: datetime,
) -> ToolApprovalAdmission:
    args_digest = canonical_sha256(args)
    operation_id = canonical_sha256(
        {"contract": "tool-operation-v1", "approval_id": approval_id}
    )
    record = ApprovalRecord(
        approval_id=approval_id,
        execution_id="execution",
        tenant_id="tenant",
        operation_id=operation_id,
        status=ApprovalStatus.PENDING,
        idempotency_key_digest=None,
        decision=None,
        decided_by=None,
        decision_digest=None,
        created_at=now,
        decided_at=None,
    )
    operation = OperationLedgerInput(
        operation_id=canonical_sha256(
            {"contract": "tool-approval-admission-v1", "approval_id": approval_id}
        ),
        tenant_id="tenant",
        resource_kind=ResourceKind.APPROVAL,
        resource_id=approval_id,
        execution_id="execution",
        operation_kind=OperationKind.APPROVAL,
        status=OperationStatus.SUCCEEDED,
        request_digest=canonical_sha256(
            {"contract": "request", "approval_id": approval_id}
        ),
        result_ref=approval_id,
        result_digest=canonical_sha256(
            {"contract": "result", "approval_id": approval_id}
        ),
        error_code=None,
        compactable=True,
        created_at=now,
        updated_at=now,
    )
    return ToolApprovalAdmission(record, operation, tool_name, args_digest)


async def _enter_wait(
    state: RuntimeState,
    *,
    namespace: str,
    approval_ids: tuple[str, ...] = ("approval-1",),
) -> tuple[RuntimeStateCommands, PendingApprovalContinuation, int, int]:
    now = datetime.now(timezone.utc)
    run_id = step_run_id(namespace=namespace, tenant_id="tenant", execution_id="execution", segment_sequence=1)
    execution = _execution(now)
    checkpoint = RecoveryCheckpoint(
        "execution",
        "tenant",
        _recovery_input(),
        run_id,
        1,
        RecoveryCheckpointState.ACTIVE,
        RecoveryHandoffPhase.NONE,
        None,
        None,
        None,
        0,
        now,
        now,
    )
    await state.execution.executions.create(execution)
    await state.recovery.checkpoints.create(checkpoint)
    run = RunRecord(
        run_id=run_id,
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="default",
        metadata={"segment_sequence": "1"},
        started_at=now,
    )
    snapshot = ContinuableSnapshot(
        run_id=run_id,
        step_index=1,
        messages=[ModelRequest(parts=[UserPromptPart(content="prompt")])],
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="default",
        timestamp=now,
        state="interrupted",
    )
    continuation = PendingApprovalContinuation(
        _batch_id(run_id, approval_ids),
        run_id,
    )
    admissions = tuple(
        _admission(
            approval_id,
            f"tool-{index}",
            {"index": index},
            now=now,
        )
        for index, approval_id in enumerate(approval_ids, 1)
    )
    commands = _commands(state, namespace)
    waited_execution, waited_recovery = await commands.commit_approval_wait_checkpoint(
        execution_id="execution",
        tenant_id="tenant",
        expected_execution_revision=0,
        expected_event_sequence=0,
        expected_recovery_revision=0,
        expected_agent_run_sequence=1,
        expected_previous_pending_approval=None,
        continuation=continuation,
        admissions=admissions,
        recovery_run=run,
        recovery_snapshot=snapshot,
        occurred_at=now,
    )
    assert waited_execution.status is ExecutionStatus.WAITING_APPROVAL
    assert waited_recovery.state is RecoveryCheckpointState.WAITING
    assert waited_recovery.pending_approval == continuation
    return (
        commands,
        continuation,
        waited_execution.revision,
        waited_recovery.revision,
    )


@pytest.mark.asyncio
async def test_wait_checkpoint_atomically_persists_snapshot_approvals_and_status(tmp_path: Path) -> None:
    namespace = "approval-wait"
    state = RuntimeState.filesystem(tmp_path / "test_wait_checkpoint_atomically_persists_snapshot_approvals_and_status")
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        _, continuation, execution_revision, recovery_revision = await _enter_wait(
            state,
            namespace=namespace,
        )
        assert execution_revision == 1
        assert recovery_revision == 1
        approval = await state.recovery.approvals.get("approval-1", tenant_id="tenant")
        assert approval is not None and approval.status is ApprovalStatus.PENDING
        recovery = await state.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert recovery is not None and recovery.pending_approval == continuation
        snapshot = await state.steps.read_store(RuntimeDomain.RECOVERY).latest_snapshot(
            run_id=continuation.source_step_run_id,
            include_interrupted=True,
        )
        assert snapshot is not None and snapshot.state == "interrupted"
        page = await state.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_sequence=0,
            limit=10,
        )
        assert tuple(event.event_type for event in page.items) == (
            ExecutionEventType.APPROVAL_REQUESTED,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_resume_requires_terminal_batch_and_increments_sequence_once(tmp_path: Path) -> None:
    namespace = "approval-resume"
    state = RuntimeState.filesystem(tmp_path / "test_resume_requires_terminal_batch_and_increments_sequence_once")
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        commands, continuation, execution_revision, recovery_revision = await _enter_wait(
            state,
            namespace=namespace,
        )
        with pytest.raises(AIError) as pending:
            await commands.claim_approval_resume_checkpoint(
                execution_id="execution",
                tenant_id="tenant",
                expected_execution_revision=execution_revision,
                expected_event_sequence=1,
                expected_recovery_revision=recovery_revision,
                expected_agent_run_sequence=1,
                expected_pending_approval=continuation,
                approval_ids=("approval-1",),
            )
        assert pending.value.code is ErrorCode.STORAGE_CONFLICT

        await state.recovery.approvals.decide(
            "approval-1",
            tenant_id="tenant",
            expected_status=ApprovalStatus.PENDING,
            idempotency_key_digest=idempotency_key_digest("decision-key"),
            decision=ApprovalDecision.APPROVE,
            principal_id="approver",
            decision_digest=canonical_sha256({"decision": "approve"}),
            decided_at=datetime.now(timezone.utc),
        )
        execution, recovery = await commands.claim_approval_resume_checkpoint(
            execution_id="execution",
            tenant_id="tenant",
            expected_execution_revision=execution_revision,
            expected_event_sequence=1,
            expected_recovery_revision=recovery_revision,
            expected_agent_run_sequence=1,
            expected_pending_approval=continuation,
            approval_ids=("approval-1",),
        )
        assert execution.status is ExecutionStatus.STARTED
        assert execution.agent_run_sequence == 2
        assert recovery.state is RecoveryCheckpointState.ACTIVE
        assert recovery.agent_run_sequence == 2
        assert recovery.pending_approval == continuation
        assert recovery.step_run_id == step_run_id(namespace=namespace, tenant_id="tenant", execution_id="execution", segment_sequence=2)

        replay_execution, replay_recovery = await commands.claim_approval_resume_checkpoint(
            execution_id="execution",
            tenant_id="tenant",
            expected_execution_revision=execution_revision,
            expected_event_sequence=1,
            expected_recovery_revision=recovery_revision,
            expected_agent_run_sequence=1,
            expected_pending_approval=continuation,
            approval_ids=("approval-1",),
        )
        assert replay_execution == execution
        assert replay_recovery == recovery
        current = await state.execution.executions.get("execution", tenant_id="tenant")
        assert current is not None and current.agent_run_sequence == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_policy_checkpoint_cancels_only_explicit_denied_pending_subset(tmp_path: Path) -> None:
    namespace = "approval-policy"
    state = RuntimeState.filesystem(tmp_path / "test_policy_checkpoint_cancels_only_explicit_denied_pending_subset")
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        commands, continuation, _, recovery_revision = await _enter_wait(
            state,
            namespace=namespace,
            approval_ids=("approval-1", "approval-2"),
        )
        records = await commands.commit_approval_policy_checkpoint(
            execution_id="execution",
            tenant_id="tenant",
            expected_recovery_revision=recovery_revision,
            expected_pending_approval=continuation,
            batch_approval_ids=("approval-1", "approval-2"),
            denied_approval_ids=("approval-1",),
            decided_at=datetime.now(timezone.utc),
        )
        by_id = {record.approval_id: record for record in records}
        assert by_id == {"approval-1": by_id["approval-1"]}
        assert by_id["approval-1"].status is ApprovalStatus.CANCELLED
        pending = await state.recovery.approvals.get("approval-2", tenant_id="tenant")
        assert pending is not None and pending.status is ApprovalStatus.PENDING
        recovery = await state.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert recovery is not None
        assert recovery.state is RecoveryCheckpointState.WAITING
        assert recovery.pending_approval == continuation
        assert recovery.revision == recovery_revision + 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_waiting_cancel_atomically_cancels_pending_batch_and_execution(tmp_path: Path) -> None:
    namespace = "approval-cancel"
    state = RuntimeState.filesystem(tmp_path / "test_waiting_cancel_atomically_cancels_pending_batch_and_execution")
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        commands, continuation, execution_revision, recovery_revision = await _enter_wait(
            state,
            namespace=namespace,
            approval_ids=("approval-1", "approval-2"),
        )
        current = await state.execution.executions.get("execution", tenant_id="tenant")
        assert current is not None
        committed = await commands.commit_waiting_approval_cancel_checkpoint(
            ExecutionCancelRequestCommit(
                "execution",
                "tenant",
                execution_revision,
                current.event_sequence,
                "cancel-operation",
                datetime.now(timezone.utc),
            ),
            approval_ids=("approval-1", "approval-2"),
            expected_recovery_revision=recovery_revision,
            expected_agent_run_sequence=1,
            expected_pending_approval=continuation,
        )
        assert committed.status is ExecutionStatus.CANCELLING
        records = tuple(
            [
                await state.recovery.approvals.get(value, tenant_id="tenant")
                for value in ("approval-1", "approval-2")
            ]
        )
        assert all(
            record is not None and record.status is ApprovalStatus.CANCELLED
            for record in records
        )
        recovery = await state.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert recovery is not None
        assert recovery.state is RecoveryCheckpointState.WAITING
        assert recovery.pending_approval == continuation
        assert recovery.revision == recovery_revision + 1
    finally:
        await state.close()


class _Executions:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None


class _Continuation:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls = 0
        self.error = error

    async def reconcile_approval(self, execution_id: str, *, tenant_id: str) -> None:
        assert execution_id == "execution" and tenant_id == "tenant"
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_decision_exact_replay_binds_full_principal_identity_and_retries_callback() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="approval-decision", tenant_id="tenant")
    continuation = _Continuation(AIError(ErrorCode.STORAGE_UNAVAILABLE))
    try:
        now = datetime.now(timezone.utc)
        await state.recovery.approvals.create_with_operation(
            ApprovalRecord(
                "approval",
                "execution",
                "tenant",
                "business-operation",
                ApprovalStatus.PENDING,
                None,
                None,
                None,
                None,
                now,
                None,
            ),
            operation=OperationLedgerInput(
                canonical_sha256({"create": "approval"}),
                "tenant",
                ResourceKind.APPROVAL,
                "approval",
                "execution",
                OperationKind.APPROVAL,
                OperationStatus.SUCCEEDED,
                canonical_sha256({"request": "approval"}),
                "approval",
                canonical_sha256({"result": "approval"}),
                None,
                True,
                now,
                now,
            ),
        )
        service = DefaultApprovalService(
            state.recovery.approvals,
            _Executions(),
            TenantAuthorizationPolicy("tenant"),
            continuation=continuation,
        )
        principal = Principal("approver", "tenant", "service")
        request = ApprovalDecisionRequest(
            principal,
            "approval",
            "decision-key",
            ApprovalDecision.APPROVE,
        )
        with pytest.raises(AIError) as callback_error:
            await service.decide("execution", request)
        assert callback_error.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert continuation.calls == 1

        continuation.error = None
        result = await service.decide("execution", request)
        assert result.decision is ApprovalDecision.APPROVE
        assert continuation.calls == 2

        with pytest.raises(AIError) as kind_conflict:
            await service.decide(
                "execution",
                ApprovalDecisionRequest(
                    Principal("approver", "tenant", "user"),
                    "approval",
                    "decision-key",
                    ApprovalDecision.APPROVE,
                ),
            )
        assert kind_conflict.value.code is ErrorCode.APPROVAL_CONFLICT
    finally:
        await state.close()
