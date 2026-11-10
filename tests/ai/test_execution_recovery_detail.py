#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery, dispatch, and cancellation share one execution control revision."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ApprovalStatus, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecoveryDetail, ExecutionRecoveryState
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.contracts import (
    AgentAttemptClaim, ApprovalRecord, ExecutionCancelRequestCommit,
    PendingDeferredCall, PendingToolContinuation, deferred_resource_id,
)
from linktools.ai.storage import StoredPayload

from .test_history_projection_conformance import _record


def _commands(state: Persistence) -> StateCommands:
    return StateCommands(
        state.execution.executions, namespace="recovery-detail",
        events=state.execution.events, checkpoint=state.checkpoint,
        operations=state.execution.operations, background_tasks=set(),
    )


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(path / "control.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_attempt_and_source_are_distinct_execution_details(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("recovery-detail", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 0), recovery=ExecutionRecoveryDetail())
        await repository.create_execution(initial)
        # Producer control has no dependency on approval/external-call storage.
        commands = StateCommands(
            repository, namespace="recovery-detail", events=state.execution.events,
            background_tasks=set(),
        )
        first = await commands.commit_agent_attempt_checkpoint(AgentAttemptClaim(
            "execution", 0, 0, ExecutionRecoveryState.ADMITTED,
        ))
        first = first.value
        second = await commands.commit_agent_attempt_checkpoint(AgentAttemptClaim(
            "execution", first.revision, 1, ExecutionRecoveryState.ACTIVE,
        ), source_agent_run_id=first.recovery.agent_run_id)
        second = second.value
        third = await commands.commit_agent_attempt_checkpoint(AgentAttemptClaim(
            "execution", second.revision, 2, ExecutionRecoveryState.ACTIVE,
        ), source_agent_run_id=first.recovery.agent_run_id)
        third = third.value
        assert third.recovery.agent_run_id != third.recovery.source_agent_run_id
        assert third.recovery.source_agent_run_id == first.recovery.agent_run_id
        assert third.recovery.agent_run_id != second.recovery.agent_run_id
        assert third == await repository.get_recovery("execution", tenant_id="tenant")
        assert decode_domain(encode_domain(third.recovery), ExecutionRecoveryDetail) == third.recovery
        records = await repository.state_store.read(lambda tx: tx.scan_records())
        assert [record.kind for record in records] == ["execution"]
        page = await repository.list_recoverable_page(tenant_id="tenant", cursor=None, limit=1)
        assert page.items == (third,)
        assert page.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_deferred_wait_and_cancel_update_the_same_execution(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("recovery-detail", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        current = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1,
                          producer_claim_id="worker", recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(current)
        now = datetime.now(timezone.utc)
        continuation = PendingToolContinuation("run", approvals=(PendingDeferredCall(
            "call", "tool", StoredPayload.inline_json({"argument": 1}),
        ),))
        approval = ApprovalRecord(
            deferred_resource_id("approval-v1", "tenant", "execution", "run", "call"),
            "execution", ApprovalStatus.PENDING, None, None, None, None, now, None,
        )
        waiting = await _commands(state).commit_deferred_checkpoint(
            execution_id="execution", tenant_id="tenant", expected_execution_revision=0,
            expected_event_seq=0, expected_agent_run_seq=1, continuation=continuation,
            approval_records=(approval,), occurred_at=now, producer_generation=1,
        )
        assert waiting.status is ExecutionStatus.WAITING_DEFERRED
        assert waiting.recovery.pending_tools == continuation
        assert waiting.recovery_state is ExecutionRecoveryState.WAITING
        cancelled = await _commands(state).commit_deferred_cancel_checkpoint(
            ExecutionCancelRequestCommit("execution", waiting.revision, waiting.event_seq, "cancel", now),
            expected_pending_tools=continuation,
        )
        assert cancelled.status is ExecutionStatus.CANCELLING
        assert cancelled.recovery.pending_tools is None
        assert cancelled.revision == waiting.revision + 1
        assert (await state.checkpoint.approvals.get(approval.approval_id, tenant_id="tenant")).status is ApprovalStatus.CANCELLED
        records = await state.checkpoint.approvals.state_store.read(lambda tx: tx.scan_records())
        assert all(record.kind != "recovery_checkpoint" for record in records)


@pytest.mark.asyncio
async def test_observed_boundary_is_select_free_and_rejects_stale_cancellation(tmp_path: Path) -> None:
    async with Persistence.open("recovery-detail", tenant_id="tenant", storage=_storage("sqlite", tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1,
                          producer_claim_id="worker", recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(initial)
        observed = await repository.get_observed("execution", tenant_id="tenant")
        statements = []
        engine = repository.state_store.storage_group.context.engine.sync_engine
        def executed(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.lstrip().split(None, 1)[0].upper())
        event.listen(engine, "after_cursor_execute", executed)
        try:
            next_record = replace(initial, revision=1)
            receipt = await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
                tx, observed, next_record=next_record, expected_producer_generation=1,
                expected_producer_claim_id="worker",
            ))
        finally:
            event.remove(engine, "after_cursor_execute", executed)
        assert "SELECT" not in statements
        assert statements.count("UPDATE") == 1
        assert receipt.value == next_record
        await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=1,
                                          next_record=replace(next_record, revision=2, status=ExecutionStatus.CANCELLING))
        with pytest.raises(AIError) as stale:
            await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
                tx, receipt, next_record=replace(next_record, revision=2), expected_producer_generation=1,
                expected_producer_claim_id="worker",
            ))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert (await repository.get("execution", tenant_id="tenant")).status is ExecutionStatus.CANCELLING
        cancelling = await repository.get_observed("execution", tenant_id="tenant")
        terminal = await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
            tx, cancelling, next_record=cancelling.value, expected_producer_generation=1,
            expected_producer_claim_id="worker", allow_cancelling=True,
        ))
        assert terminal.value == cancelling.value
        assert terminal.storage_revision == cancelling.storage_revision + 1
        for candidate, claim in ((replace(terminal.value, revision=3), "worker"),
                                 (terminal.value, "another-worker")):
            with pytest.raises(AIError) as forbidden:
                await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
                    tx, terminal, next_record=candidate, expected_producer_generation=1,
                    expected_producer_claim_id=claim, allow_cancelling=True,
                ))
            assert forbidden.value.code is ErrorCode.STORAGE_CONFLICT


def test_deferred_argument_objects_keep_checkpoint_source_after_control_merge() -> None:
    from linktools.ai.core import PersistenceDomain
    from linktools.ai.persistence.codec import _encode_persisted_domain, iter_runtime_object_refs
    from linktools.ai.storage import ObjectRef
    reference = ObjectRef("runtime", "deferred-arguments", "a" * 64, 123)
    continuation = PendingToolContinuation("run", approvals=(PendingDeferredCall(
        "call", "tool", StoredPayload.object(reference),
    ),))
    execution = replace(_record(ExecutionStatus.STARTED, 1),
                        recovery=ExecutionRecoveryDetail(agent_run_id="run", pending_tools=continuation))
    assert tuple(iter_runtime_object_refs(_encode_persisted_domain(execution),
                                         default_domain=PersistenceDomain.EXECUTION)) == (
        (PersistenceDomain.CHECKPOINT, reference),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_terminal_pending_handoff_remains_in_recoverable_index(tmp_path: Path, backend: str) -> None:
    from linktools.ai.core import ExecutionEventType, StopReason, UsageMetrics
    from linktools.ai.execution import RecoveryHandoffPhase, RecoveryTerminalHandoff, RecoveryTerminalOutcome
    async with Persistence.open("recovery-detail", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        initial = _record(ExecutionStatus.FAILED, 1)
        outcome = RecoveryTerminalOutcome(ExecutionStatus.FAILED, ErrorCode.EXECUTION_FAILED.value, {},
                                          StopReason.ERROR, None, None, UsageMetrics(),
                                          ExecutionEventType.EXECUTION_FAILED, {}, initial.updated_at)
        handoff = RecoveryTerminalHandoff(outcome, "run", None)
        current = replace(initial, session_id="session", recovery=ExecutionRecoveryDetail(
            agent_run_id="run", handoff_phase=RecoveryHandoffPhase.PREPARED, terminal_handoff=handoff,
        ))
        await repository.create_execution(current)
        page = await repository.list_recoverable_page(tenant_id="tenant", cursor=None, limit=1)
        assert page.items == (current,)
        assert current.recovery_state is ExecutionRecoveryState.HANDOFF
        assert await repository.list_by_session("session", tenant_id="tenant", statuses=frozenset({ExecutionStatus.FAILED})) == (current,)
        assert await repository.list_by_session("session", tenant_id="tenant", statuses=frozenset({ExecutionStatus.STARTED})) == ()
        stored = await repository.state_store.read(lambda tx: tx.get_record(repository.execution_record_key("execution")))
        assert stored.state == "execution_handoff"
        completed = replace(current, revision=1, recovery=replace(current.recovery, handoff_phase=RecoveryHandoffPhase.COMPLETED))
        await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=0, next_record=completed)
        assert (await repository.list_recoverable_page(tenant_id="tenant", cursor=None, limit=1)).items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_activation_returns_real_physical_receipt_after_independent_guard(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("recovery-detail", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 0), recovery=ExecutionRecoveryDetail())
        await repository.create_execution(initial)
        before = await repository.get_observed("execution", tenant_id="tenant")
        await repository.state_store.mutate(lambda tx: tx.guard_record_revision(
            repository.execution_record_key("execution"), expected_storage_revision=before.storage_revision,
        ))
        receipt = await _commands(state).commit_agent_attempt_checkpoint(AgentAttemptClaim(
            "execution", initial.revision, 0, ExecutionRecoveryState.ADMITTED,
        ))
        assert receipt.storage_revision == before.storage_revision + 2
        assert receipt.value.revision == initial.revision + 1
        assert receipt == await repository.get_observed("execution", tenant_id="tenant")
        dispatched = await repository.state_store.mutate(lambda tx: repository.commit_boundary_in_transaction(
            tx, receipt, next_record=receipt.value, expected_producer_generation=receipt.value.producer_generation,
            expected_producer_claim_id=receipt.value.producer_claim_id,
        ))
        assert dispatched.storage_revision == receipt.storage_revision + 1
