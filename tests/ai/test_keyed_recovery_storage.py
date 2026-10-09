#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyed recovery admissions share the durable execution transaction."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.core import (
    ExecutionStatus,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state._contracts import (
    AgentAttemptClaim,
    ExecutionRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
)
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands

from .test_execution_recovery_commands import _commands, _execution

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NAMESPACE = "keyed-recovery"


@pytest_asyncio.fixture(params=("memory", "filesystem", "sqlite"))
async def storage(
    request: pytest.FixtureRequest, tmp_path: Path,
) -> AsyncIterator[RuntimeStorage]:
    if request.param == "memory":
        state = RuntimeStorage.in_memory()
    elif request.param == "filesystem":
        state = RuntimeStorage.filesystem(tmp_path / "state")
    else:
        state = RuntimeStorage.sqlite(tmp_path / "state.db")
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        yield state
    finally:
        await state.close()


def _operation(*, claim: str | None = "first-producer") -> OperationLedgerInput:
    return OperationLedgerInput(
        operation_id="recovery-key", tenant_id="tenant",
        resource_kind=ResourceKind.EXECUTION, resource_id="execution",
        execution_id="execution", operation_kind=OperationKind.EXECUTION_RECOVER,
        status=OperationStatus.RUNNING, request_digest="a" * 64,
        result_ref=claim, result_digest=None, error_code=None,
        compactable=False, created_at=_NOW, updated_at=_NOW,
    )


async def _admission(
    storage: RuntimeStorage, mode: str,
) -> tuple[ExecutionRecord, Callable[[OperationLedgerInput], Awaitable[ExecutionRecord]]]:
    execution = _execution(_NOW)
    if mode == "attempt":
        execution = replace(execution, agent_run_seq=0)
    await storage.execution.executions.create_with_history_head(execution)
    if mode == "resume":
        async def admit(operation: OperationLedgerInput) -> ExecutionRecord:
            return await _commands(storage).commit_resumed(
                execution, recovery_operation=operation,
            )
        return execution, admit
    checkpoint = RecoveryCheckpoint(
        execution_id=execution.execution_id, agent_run_id=None,
        state=RecoveryCheckpointState.ADMITTED, revision=0,
        created_at=_NOW, updated_at=_NOW,
    )
    await storage.recovery.checkpoints.create(checkpoint)
    commands = RuntimeStateCommands(
        storage.execution.executions, namespace=_NAMESPACE,
        events=storage.execution.events, operations=storage.execution.operations,
        recovery=storage.recovery.checkpoints, background_tasks=set(),
    )
    claim = AgentAttemptClaim(
        execution.execution_id, execution.revision, execution.agent_run_seq,
        checkpoint.revision, checkpoint.state,
    )

    async def admit(operation: OperationLedgerInput) -> ExecutionRecord:
        current, _ = await commands.commit_agent_attempt_checkpoint(
            claim, recovery_operation=operation,
        )
        return current
    return execution, admit


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("resume", "attempt"))
async def test_competing_claims_share_one_recovery_admission(
    storage: RuntimeStorage, mode: str,
) -> None:
    execution, admit = await _admission(storage, mode)
    first = _operation()
    second = _operation(claim="competing-producer")
    results = await asyncio.gather(admit(first), admit(second))
    assert results[0] == results[1]
    admitted = results[0]
    assert admitted.revision == execution.revision + 1
    assert admitted.agent_run_seq == execution.agent_run_seq + int(mode == "attempt")
    receipt = await storage.execution.operations.get(first.operation_id, tenant_id="tenant")
    assert receipt is not None
    assert receipt.result_ref in {first.result_ref, second.result_ref}
    assert receipt.status is OperationStatus.RUNNING
    head = await storage.execution.executions.get_history_head("execution", tenant_id="tenant")
    assert head.producer_generation == admitted.revision
    assert head.producer_claim_id == receipt.result_ref
    events = await storage.execution.events.list("execution", tenant_id="tenant", after_event_seq=0, limit=10)
    assert len(events.items) == int(mode == "resume")
    if mode == "attempt":
        checkpoint = await storage.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert checkpoint is not None
        assert checkpoint.revision == 1
        assert checkpoint.state is RecoveryCheckpointState.ACTIVE

    advanced = await storage.execution.executions.compare_and_swap(
        "execution", tenant_id="tenant", expected_revision=admitted.revision,
        next_record=replace(admitted, revision=admitted.revision + 1, status=ExecutionStatus.CANCELLING),
    )
    assert await admit(second) == advanced
    assert await storage.execution.executions.get_history_head("execution", tenant_id="tenant") == head

    for changed in (
        replace(second, request_digest="b" * 64),
        replace(second, resource_id="different-execution"),
        replace(second, execution_id="different-execution"),
        replace(second, tenant_id="different-tenant"),
        replace(second, operation_kind=OperationKind.EXECUTION_CANCEL),
        replace(second, compactable=True),
    ):
        with pytest.raises(AIError) as conflict:
            await admit(changed)
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert await storage.execution.executions.get("execution", tenant_id="tenant") == advanced


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("resume", "attempt"))
async def test_recovery_receipt_rolls_back_with_producer_checkpoint(
    storage: RuntimeStorage, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, admit = await _admission(storage, mode)
    head = await storage.execution.executions.get_history_head("execution", tenant_id="tenant")
    operations = storage.execution.operations

    async def fail_receipt(*args, **kwargs):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(operations, "append_in_transaction", fail_receipt)
    with pytest.raises(AIError) as failed:
        await admit(_operation())
    assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert await storage.execution.executions.get("execution", tenant_id="tenant") == execution
    assert await storage.execution.executions.get_history_head("execution", tenant_id="tenant") == head
    assert await operations.get("recovery-key", tenant_id="tenant") is None
    assert not (await storage.execution.events.list("execution", tenant_id="tenant", after_event_seq=0, limit=10)).items
    if mode == "attempt":
        checkpoint = await storage.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert checkpoint is not None
        assert checkpoint.state is RecoveryCheckpointState.ADMITTED
        assert checkpoint.revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("resume", "attempt"))
async def test_recovery_receipt_reconciles_lost_commit_acknowledgement(
    storage: RuntimeStorage, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, admit = await _admission(storage, mode)
    group = storage.execution.executions.state_store.storage_group
    original = type(group).mutate
    interrupted = False

    async def lose_ack(current_group, stores, callback):
        nonlocal interrupted
        result = await original(current_group, stores, callback)
        if current_group is group and not interrupted:
            interrupted = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    monkeypatch.setattr(type(group), "mutate", lose_ack)
    admitted = await admit(_operation())
    assert interrupted
    assert admitted.revision == execution.revision + 1
    receipt = await storage.execution.operations.get("recovery-key", tenant_id="tenant")
    assert receipt is not None
    head = await storage.execution.executions.get_history_head("execution", tenant_id="tenant")
    assert head.producer_claim_id == receipt.result_ref == "first-producer"
    assert await admit(_operation(claim="retry-producer")) == admitted


@pytest.mark.asyncio
async def test_cancellation_receipt_guards_revision_without_admitting_producer(
    storage: RuntimeStorage,
) -> None:
    execution = replace(_execution(_NOW), status=ExecutionStatus.RECOVERY_REQUIRED)
    repository = storage.execution.executions
    await repository.create_with_history_head(execution)
    head = await repository.get_history_head("execution", tenant_id="tenant")
    commands = _commands(storage)
    operation = _operation(claim=None)
    receipt = await commands.admit_recovery_without_producer(execution, operation)
    assert receipt.result_ref is None
    assert await repository.get("execution", tenant_id="tenant") == execution
    assert await repository.get_history_head("execution", tenant_id="tenant") == head
    assert not (await storage.execution.events.list("execution", tenant_id="tenant", after_event_seq=0, limit=10)).items
    cancelling = await commands.commit_cancel_claim(execution)
    assert await commands.admit_recovery_without_producer(execution, operation) == receipt
    # A cancellation receipt also wins a later producer claim for the same key.
    assert await commands.commit_resumed(cancelling, recovery_operation=_operation()) == cancelling
    with pytest.raises(AIError) as stale:
        await commands.admit_recovery_without_producer(
            execution, replace(operation, operation_id="new-recovery-key"),
        )
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    assert await storage.execution.operations.get("new-recovery-key", tenant_id="tenant") is None
    assert await repository.get_history_head("execution", tenant_id="tenant") == head


@pytest.mark.asyncio
async def test_recovery_claim_is_immutable_and_survives_operation_compaction(
    storage: RuntimeStorage,
) -> None:
    execution, admit = await _admission(storage, "resume")
    admitted = await admit(_operation())
    operations = storage.execution.operations
    receipt = await operations.get("recovery-key", tenant_id="tenant")
    assert receipt is not None
    for changed in (
        replace(receipt, result_ref="replacement-producer"),
        replace(receipt, result_digest="b" * 64),
    ):
        with pytest.raises(AIError) as conflict:
            await operations.compare_and_swap(
                receipt.operation_id, tenant_id="tenant",
                expected_status=receipt.status, next_record=changed,
            )
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    settled = await operations.compare_and_swap(
        receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
        next_record=replace(receipt, status=OperationStatus.SUCCEEDED),
    )
    for index in range(3):
        terminal = await operations.append(replace(
            _operation(), operation_id=f"compactable-{index}",
            operation_kind=OperationKind.EXECUTION_CANCEL, status=OperationStatus.SUCCEEDED,
            result_ref=None, compactable=True,
        ))
    await operations.compact_terminal(
        ResourceKind.EXECUTION, execution.execution_id, tenant_id="tenant",
        through_sequence=terminal.sequence,
    )
    assert await operations.get("compactable-0", tenant_id="tenant") is None
    assert await operations.get(receipt.operation_id, tenant_id="tenant") == settled
    assert await admit(_operation(claim="post-compaction-retry")) == admitted


@pytest.mark.asyncio
async def test_pending_operation_pages_apply_exclusive_sequence_before_limit(
    storage: RuntimeStorage,
) -> None:
    operations = storage.execution.operations
    rows = []
    for index, status in enumerate((
        OperationStatus.RUNNING, OperationStatus.SUCCEEDED, OperationStatus.PENDING,
        OperationStatus.RUNNING, OperationStatus.CANCELLED, OperationStatus.PENDING,
    )):
        rows.append(await operations.append(replace(
            _operation(), operation_id=f"operation-{index}",
            operation_kind=OperationKind.EXECUTION_CANCEL, result_ref=None,
            status=status, compactable=True,
        )))
    first = await operations.list_pending(
        ResourceKind.EXECUTION, "execution", tenant_id="tenant", limit=2,
    )
    assert first == (rows[0], rows[2])
    second = await operations.list_pending(
        ResourceKind.EXECUTION, "execution", tenant_id="tenant", limit=2,
        after_sequence=first[-1].sequence,
    )
    assert second == (rows[3], rows[5])
    assert await operations.list_pending(
        ResourceKind.EXECUTION, "execution", tenant_id="tenant", limit=2,
        after_sequence=second[-1].sequence,
    ) == ()
    assert await operations.list_pending(
        ResourceKind.EXECUTION, "execution", tenant_id="tenant", limit=1,
        states=frozenset({OperationStatus.PENDING}), after_sequence=rows[0].sequence,
    ) == (rows[2],)
    with pytest.raises(ValueError):
        await operations.list_pending(
            ResourceKind.EXECUTION, "execution", tenant_id="tenant", limit=2,
            after_sequence=-1,
        )
