#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred command outcomes follow coherent durable frontiers."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, ExternalCallStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import (
    ExecutionCancelRequestCommit, ExternalCallRecord, deferred_resource_id,
)
from linktools.ai.storage import StoredPayload

from .test_runtime_approval_recovery import _approval, _checkpoint, _commands, _continuation


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(path / "state.db")


async def _prepare(state: Persistence, phase: str):
    now = datetime.now(timezone.utc)
    initial = _checkpoint(now)
    base = _continuation()
    pending = base.approvals[0]
    continuation = replace(
        base,
        approvals=(pending, replace(pending, tool_call_id="approval-2")),
        calls=(replace(pending, tool_call_id="external-1"), replace(pending, tool_call_id="external-2")),
    )
    approval = _approval(initial, continuation, now)
    approvals = tuple(replace(approval, approval_id=deferred_resource_id(
        "approval-v1", "tenant", initial.execution_id, continuation.source_agent_run_id, item.tool_call_id,
    )) for item in continuation.approvals)
    calls = tuple(ExternalCallRecord(
        deferred_resource_id("external-call-v1", "tenant", initial.execution_id,
                             continuation.source_agent_run_id, item.tool_call_id),
        initial.execution_id, ExternalCallStatus.PENDING, None, now, None,
    ) for item in continuation.calls)
    commands = _commands(state, "readback")
    await state.execution.executions.create_execution(initial)

    async def pause():
        return await commands.commit_deferred_checkpoint(
            execution_id=initial.execution_id, tenant_id="tenant",
            expected_execution_revision=initial.revision, expected_event_seq=initial.event_seq,
            expected_agent_run_seq=initial.agent_run_seq, continuation=continuation,
            approval_records=approvals, external_records=calls, occurred_at=now,
        )

    if phase == "pause":
        operation = pause
    else:
        waiting = await pause()

        async def operation():
            return await commands.commit_deferred_cancel_checkpoint(
                ExecutionCancelRequestCommit(waiting.execution_id, waiting.revision, waiting.event_seq,
                                             "cancel", now),
                expected_pending_tools=continuation,
            )
    return operation, commands, initial, continuation, approvals, calls


async def _facts(state, approvals, calls):
    return (
        await state.execution.executions.get("execution", tenant_id="tenant"),
        tuple([await state.checkpoint.approvals.get(item.approval_id, tenant_id="tenant") for item in approvals]),
        tuple([await state.checkpoint.external_calls.get(item.call_id, tenant_id="tenant") for item in calls]),
        await state.execution.events.list("execution", tenant_id="tenant", after_event_seq=0, limit=20),
    )


async def _resolve_members(state, approvals, calls):
    now = datetime.now(timezone.utc)
    await state.checkpoint.approvals.decide(
        approvals[0].approval_id, tenant_id="tenant", expected_status=ApprovalStatus.PENDING,
        idempotency_key_digest="a" * 64, decision=ApprovalDecision.APPROVE, principal_id="owner",
        decision_digest="b" * 64, decided_at=now,
    )
    await state.checkpoint.external_calls.supply(
        calls[0].call_id, tenant_id="tenant", expected_status=ExternalCallStatus.PENDING,
        idempotency_key_digest="c" * 64, resolution_kind="succeeded",
        result_payload=StoredPayload.inline_text("result"), resolution_metadata={}, supplied_at=now,
    )
    repository = state.checkpoint.external_calls
    await repository.state_store.mutate(lambda tx: repository.cancel_pending_in_transaction(
        tx, (calls[1].call_id,), execution_id="execution", tenant_id="tenant", cancelled_at=now,
    ))


def _fail_before(monkeypatch, group, error):
    original = group.mutate

    async def fail(stores, callback):
        monkeypatch.setattr(group, "mutate", original)
        raise error

    monkeypatch.setattr(group, "mutate", fail)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("phase", ("pause", "cancel"))
async def test_deferred_readback_preserves_uncommitted_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, phase)
        before = await _facts(state, approvals, calls)
        failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)
        _fail_before(monkeypatch, state.execution.executions.state_store.storage_group, failure)
        with pytest.raises(AIError) as caught:
            await operation()
        assert await _facts(state, approvals, calls) == before
        assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("phase", ("pause", "cancel"))
async def test_deferred_readback_accepts_independent_member_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, phase)
        if phase == "pause":
            await state.checkpoint.approvals.create(approvals[0])
        else:
            await _resolve_members(state, approvals, calls)
        before = await _facts(state, approvals, calls)
        failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)
        _fail_before(monkeypatch, state.execution.executions.state_store.storage_group, failure)
        with pytest.raises(AIError) as caught:
            await operation()
        assert await _facts(state, approvals, calls) == before
        assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("phase", ("pause", "cancel"))
async def test_deferred_readback_recognizes_lost_ack_with_resolved_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, phase)
        if phase == "cancel":
            await _resolve_members(state, approvals, calls)
        group = state.execution.executions.state_store.storage_group
        original = group.mutate

        async def lose_ack(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            await original(stores, callback)
            if phase == "pause":
                await _resolve_members(state, approvals, calls)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_ack)
        result = await operation()
        actual, actual_approvals, actual_calls, _ = await _facts(state, approvals, calls)
        assert result == actual
        assert result.status is (ExecutionStatus.WAITING_DEFERRED if phase == "pause" else ExecutionStatus.CANCELLING)
        assert actual_approvals[0].status is ApprovalStatus.APPROVED
        assert actual_calls[0].status is ExternalCallStatus.SUPPLIED
        assert actual_calls[1].status is ExternalCallStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("phase", ("pause", "cancel"))
@pytest.mark.parametrize("defect", ("missing", "wrong_owner"))
async def test_deferred_readback_rejects_inconsistent_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str, defect: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, phase)
        group = state.execution.executions.state_store.storage_group
        original = group.mutate
        repository = state.checkpoint.approvals

        async def break_member(tx):
            key = repository._key("approval", (approvals[0].approval_id,))
            current = await tx.get_record(key)
            if defect == "missing":
                assert await tx.delete_record(key, expected_storage_revision=current.storage_revision)
            else:
                value = await repository.get_in_transaction(tx, approvals[0].approval_id, tenant_id="tenant")
                changed = replace(
                    current,
                    data=repository._stored("approval", (value.approval_id,), replace(value, execution_id="other"),
                                            state=value.status.value).data,
                    storage_revision=current.storage_revision + 1,
                )
                assert await tx.replace_record(changed, expected_storage_revision=current.storage_revision)

        async def lose_ack(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            await original(stores, callback)
            await repository.state_store.mutate(break_member)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_ack)
        with pytest.raises(AIError) as caught:
            await operation()
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        actual = await state.execution.executions.get("execution", tenant_id="tenant")
        assert actual.status is (ExecutionStatus.WAITING_DEFERRED if phase == "pause" else ExecutionStatus.CANCELLING)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_deferred_readback_rejects_cancel_target_with_pending_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, "cancel")
        waiting = await state.execution.executions.get("execution", tenant_id="tenant")
        group = state.execution.executions.state_store.storage_group
        original = group.mutate

        async def partial(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            repository = state.execution.executions
            await repository.state_store.mutate(lambda tx: repository.request_cancel_in_transaction(
                tx, ExecutionCancelRequestCommit("execution", waiting.revision, waiting.event_seq,
                                                 "cancel", datetime.now(timezone.utc)),
                expected_status=ExecutionStatus.WAITING_DEFERRED,
                recovery=replace(waiting.recovery, pending_tools=None),
            ))
            # The real execution component committed, but no member cancellation did.
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", partial)
        with pytest.raises(AIError) as caught:
            await operation()
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        _, actual_approvals, actual_calls, _ = await _facts(state, approvals, calls)
        assert all(item.status is ApprovalStatus.PENDING for item in actual_approvals)
        assert all(item.status is ExternalCallStatus.PENDING for item in actual_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("pause", "cancel"))
async def test_deferred_readback_unavailable_stays_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage("memory", tmp_path)) as state:
        operation, _, _, _, _, _ = await _prepare(state, phase)
        group = state.execution.executions.state_store.storage_group
        _fail_before(monkeypatch, group, AIError(ErrorCode.STORAGE_UNAVAILABLE))

        async def unavailable(stores, callback):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        with monkeypatch.context() as reads:
            reads.setattr(group, "read_many", unavailable)
            with pytest.raises(AIError) as caught:
                await operation()
            assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN


@pytest.mark.asyncio
async def test_deferred_readback_different_source_frontier_stays_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage("memory", tmp_path)) as state:
        operation, _, initial, _, _, _ = await _prepare(state, "pause")
        await state.execution.executions.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=initial.revision,
            next_record=replace(initial, revision=initial.revision + 1,
                                recovery=replace(initial.recovery, agent_run_id="different-source")),
        )
        _fail_before(monkeypatch, state.execution.executions.state_store.storage_group, AIError(ErrorCode.STORAGE_UNAVAILABLE))
        with pytest.raises(AIError) as caught:
            await operation()
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("pause", "cancel"))
async def test_deferred_readback_settles_commit_before_caller_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage("memory", tmp_path)) as state:
        operation, _, _, _, approvals, calls = await _prepare(state, phase)
        group = state.execution.executions.state_store.storage_group
        original = group.mutate
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            entered.set()
            await release.wait()
            await original(stores, callback)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", blocked)
        task = asyncio.create_task(operation())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        actual, _, _, _ = await _facts(state, approvals, calls)
        assert actual.status is (ExecutionStatus.WAITING_DEFERRED if phase == "pause" else ExecutionStatus.CANCELLING)


@pytest.mark.asyncio
async def test_deferred_readback_runtime_cancel_preserves_uncommitted_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.exceptions import ApprovalRequired
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from linktools.ai.capability import AgentContext
    from linktools.ai.execution import CancelExecutionRequest
    from linktools.ai.persistence.commands import StateCommands
    from ._runtime_test_helpers import _wait_for_committed
    from .test_official_runtime_recovery import _runtime

    async def checked(_context: AgentContext[None]) -> str:
        raise ApprovalRequired()

    async def model(_messages, _info):
        return ModelResponse(parts=[ToolCallPart("checked", {}, tool_call_id="approval")])

    async with _runtime(tmp_path / "runtime.db", model, checked) as (runtime, state, _):
        execution = await runtime.agents.get().start("question", idempotency_key="start")
        before = await _wait_for_committed(
            lambda: state.execution.executions.get(execution.execution_id, tenant_id="default"),
            lambda current: current is not None and current.status is ExecutionStatus.WAITING_DEFERRED,
        )
        original = StateCommands.commit_deferred_cancel_checkpoint
        failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def fail_command(self, *args, **kwargs):
            _fail_before(monkeypatch, self._execution.state_store.storage_group, failure)
            return await original(self, *args, **kwargs)

        with monkeypatch.context() as command:
            command.setattr(StateCommands, "commit_deferred_cancel_checkpoint", fail_command)
            with pytest.raises(AIError) as caught:
                await runtime.executions.cancel(execution.execution_id, CancelExecutionRequest(runtime.default_principal, "cancel"))
            assert caught.value is failure
        assert await state.execution.executions.get(execution.execution_id, tenant_id="default") == before
        approvals = await state.checkpoint.approvals.list_pending(execution.execution_id, tenant_id="default")
        assert len(approvals) == 1 and approvals[0].status is ApprovalStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_generation", (1, 3))
async def test_deferred_readback_preserves_producer_conflict(tmp_path: Path, expected_generation: int) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage("memory", tmp_path)) as state:
        _, commands, initial, continuation, approvals, calls = await _prepare(state, "pause")
        current = initial
        for generation in (1, 2):
            current = await state.execution.executions.compare_and_swap(
                "execution", tenant_id="tenant", expected_revision=current.revision,
                next_record=replace(current, revision=current.revision + 1, producer_generation=generation),
            )
        before = await _facts(state, approvals, calls)
        with pytest.raises(AIError) as caught:
            await commands.commit_deferred_checkpoint(
                execution_id=current.execution_id, tenant_id="tenant",
                expected_execution_revision=current.revision, expected_event_seq=current.event_seq,
                expected_agent_run_seq=current.agent_run_seq, continuation=continuation,
                approval_records=approvals, external_records=calls, occurred_at=initial.created_at,
                producer_generation=expected_generation,
            )
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _facts(state, approvals, calls) == before
