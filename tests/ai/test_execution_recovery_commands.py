#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused durable recovery command regressions."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
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
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._recovery_commands import RuntimeRecoveryCommands
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload

from ._runtime_test_helpers import execution_owner_fields


def _binding() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("agent", model="default"),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _execution(now: datetime) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_seq=0,
        agent_run_seq=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        **execution_owner_fields(),
    )


def _tool(now: datetime, *, operation_id: str = "tool-operation") -> ToolOperationRecord:
    return ToolOperationRecord(
        tool_operation_id=operation_id,
        execution_id="execution",
        agent_run_id="agent-run",
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


def _commands(state: RuntimeStorage) -> RuntimeRecoveryCommands:
    return RuntimeRecoveryCommands(
        state.execution.executions,
        state.execution.events,
        state.recovery.operations,
        state.recovery.tools,
        execution_operations=state.execution.operations,
        background_tasks=set(),
    )


def _resolution_operation(
    now: datetime,
    *,
    operation_id: str = "resolution-operation",
) -> OperationLedgerInput:
    return OperationLedgerInput(
        operation_id,
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


@pytest.mark.asyncio
async def test_recovery_status_and_resume_are_durable_nonterminal_events() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="execution-recovery", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    try:
        await state.execution.executions.create_with_history_head(execution)
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
            tenant_id=state.execution.events.tenant_id,
            after_event_seq=0,
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
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="recovery-cancel", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    try:
        await state.execution.executions.create_with_history_head(execution)
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
        assert fresh.event_seq == recovery.event_seq + 1

        with pytest.raises(AIError) as stale:
            await commands.commit_resumed(recovery)
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT

        cancelling = await commands.commit_cancel_claim(fresh)
        assert cancelling.status is ExecutionStatus.CANCELLING
        assert cancelling.event_seq == fresh.event_seq

        events = await state.execution.events.list(
            "execution",
            tenant_id="tenant",
            after_event_seq=0,
            limit=10,
        )
        assert tuple(value.event_type for value in events.items) == (
            ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
            ExecutionEventType.CANCEL_REQUESTED,
        )
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_recovery_cancel_claim_reports_concurrent_intent_as_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="recovery-cancel-race", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    execution = _execution(now)
    try:
        await state.execution.executions.create_with_history_head(execution)
        commands = _commands(state)
        recovery = await commands.commit_recovery_required(
            execution,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            safe_error_details={"operation_id": "tool-operation", "fence": 3},
        )
        concurrent = OperationLedgerInput(
            "concurrent-cancel",
            "tenant",
            ResourceKind.EXECUTION,
            "execution",
            "execution",
            OperationKind.EXECUTION_CANCEL,
            OperationStatus.PENDING,
            canonical_sha256({"cancel": "concurrent"}),
            None,
            None,
            None,
            True,
            now,
            now,
        )
        repository = state.execution.executions
        original_compare_and_swap = repository.compare_and_swap
        advanced = False

        async def race_compare_and_swap(*args, **kwargs):
            nonlocal advanced
            if not advanced:
                advanced = True
                await commands.commit_cancel_intent(recovery, concurrent)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original_compare_and_swap(*args, **kwargs)

        monkeypatch.setattr(repository, "compare_and_swap", race_compare_and_swap)

        with pytest.raises(AIError) as raised:
            await commands.commit_cancel_claim(recovery)

        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        current = await state.execution.executions.get(
            "execution",
            tenant_id="tenant",
        )
        assert current is not None
        assert current.status is ExecutionStatus.RECOVERY_REQUIRED
        assert current.event_seq == recovery.event_seq + 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_not_applied_resolution_reopens_tool_with_next_fence() -> None:
    state = RuntimeStorage.in_memory()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_status", "result_payload", "error_code", "error_payload"),
    (
        (
            ToolOperationStatus.COMPLETED,
            StoredPayload.inline_json({"applied": True}),
            None,
            None,
        ),
        (
            ToolOperationStatus.FAILED,
            None,
            ErrorCode.TOOL_EXECUTION_FAILED.value,
            StoredPayload.inline_bytes(
                b'{"version":1,"kind":"tool_call_failed","message":"failed"}'
            ),
        ),
    ),
)
async def test_applied_and_failed_effect_resolutions_persist_tool_truth(
    target_status: ToolOperationStatus,
    result_payload: StoredPayload | None,
    error_code: str | None,
    error_payload: StoredPayload | None,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="tool-resolution-terminal", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    try:
        await state.recovery.tools.reserve(_tool(now))
        operation = _resolution_operation(now)
        resolved = await _commands(state).resolve_tool_effect(
            operation,
            expected_fence=3,
            target_status=target_status,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )

        assert resolved.status is target_status
        assert resolved.result_payload == result_payload
        assert resolved.error_code == error_code
        assert resolved.error_payload == error_payload
        committed_operation = await state.recovery.operations.get(
            operation.operation_id,
            tenant_id="tenant",
        )
        assert committed_operation is not None
        assert committed_operation.status is OperationStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conflict", "expected_error"),
    (
        ("fence", ErrorCode.TOOL_OPERATION_CONFLICT),
        ("idempotency", ErrorCode.IDEMPOTENCY_CONFLICT),
    ),
)
async def test_tool_effect_resolution_rejects_stale_fence_and_key_reuse(
    conflict: str,
    expected_error: ErrorCode,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="tool-resolution-conflict", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    try:
        await state.recovery.tools.reserve(_tool(now))
        commands = _commands(state)
        operation = _resolution_operation(now)
        if conflict == "idempotency":
            await commands.resolve_tool_effect(
                operation,
                expected_fence=3,
                target_status=ToolOperationStatus.PENDING,
            )
            operation = replace(
                operation,
                request_digest=canonical_sha256({"resolution": "different"}),
            )
            expected_fence = 3
        else:
            expected_fence = 2

        with pytest.raises(AIError) as raised:
            await commands.resolve_tool_effect(
                operation,
                expected_fence=expected_fence,
                target_status=ToolOperationStatus.PENDING,
            )
        assert raised.value.code is expected_error
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_active_restart_admits_a_new_history_producer_and_fences_predecessor() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="restart-producer", tenant_id="tenant")
    execution = _execution(datetime.now(timezone.utc))
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        resumed = await _commands(state).commit_resumed(execution)
        assert resumed.status is ExecutionStatus.STARTED
        assert resumed.revision == execution.revision + 1
        head = await repository.get_history_head(execution.execution_id, tenant_id="tenant")
        assert head.producer_generation == resumed.revision

        async def guard(transaction, generation):
            return await repository.require_open_history_head_in_transaction(
                transaction, execution.execution_id, expected_producer_generation=generation,
            )

        with pytest.raises(AIError) as stale:
            await repository.state_store.mutate(lambda transaction: guard(transaction, execution.revision))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        await repository.state_store.mutate(lambda transaction: guard(transaction, resumed.revision))
        with pytest.raises(AIError) as duplicate:
            await _commands(state).commit_resumed(execution)
        assert duplicate.value.code is ErrorCode.STORAGE_CONFLICT
        successor = await _commands(state).commit_resumed(resumed)
        assert successor.revision == resumed.revision + 1
        with pytest.raises(AIError) as superseded:
            await _commands(state).commit_resumed(execution)
        assert superseded.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as stale:
            await repository.state_store.mutate(lambda transaction: guard(transaction, resumed.revision))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated_update", (False, True))
async def test_resume_recovers_its_own_lost_commit_acknowledgement(monkeypatch, unrelated_update) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="resume-ack", tenant_id="tenant")
    execution = _execution(datetime.now(timezone.utc))
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        store_type = type(repository.state_store)
        original = store_type.mutate
        interrupted = False

        async def lose_ack(store, operation):
            nonlocal interrupted
            result = await original(store, operation)
            if store is repository.state_store and not interrupted:
                interrupted = True
                if unrelated_update:
                    await repository.compare_and_swap(
                        execution.execution_id,
                        tenant_id="tenant",
                        expected_revision=result.revision,
                        next_record=replace(result, revision=result.revision + 1),
                    )
                raise RuntimeError("commit acknowledgement lost")
            return result

        monkeypatch.setattr(store_type, "mutate", lose_ack)
        resumed = await _commands(state).commit_resumed(execution)
        assert interrupted
        assert resumed.revision == execution.revision + 1 + int(unrelated_update)
        head = await repository.get_history_head(execution.execution_id, tenant_id="tenant")
        assert head.producer_generation == execution.revision + 1
    finally:
        await state.close()
