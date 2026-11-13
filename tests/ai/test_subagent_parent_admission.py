#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Child admission is conditional on the still-active parent producer."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, ResourceKind
from linktools.ai.persistence.contracts import ExecutionRecoveryDetail, PendingToolContinuation, PendingDeferredCall
from linktools.ai.storage import StoredPayload
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionStartReservation, IdempotencyRecord
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from .test_history_projection_conformance import _record


pytestmark = pytest.mark.asyncio
_BACKENDS = ("memory", "local_sqlite", "sqlite")


def _storage(backend: str, path: Path) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "local_sqlite":
        return PersistenceStorage.from_root(path / "state", exclusive_writer=True)
    return PersistenceStorage.sqlite(path / "state.db")


def _reservation(*, generation: int | None = 1, identity: str = "child") -> ExecutionStartReservation:
    execution = replace(
        _record(ExecutionStatus.PENDING_START, 0), execution_id=identity,
        parent_execution_id="parent", root_execution_id="parent", parent_invocation_id=identity,
        lineage_kind=ExecutionLineageKind.SUBAGENT,
    )
    return ExecutionStartReservation(
        execution,
        IdempotencyRecord(
            "execution.subagent", identity, "request-" + identity, ResourceKind.EXECUTION,
            execution.execution_id, IdempotencyStatus.RESERVED, None, None,
            execution.created_at, execution.updated_at,
        ),
        parent_producer_generation=generation,
    )


async def _parent(state: Persistence):
    repository = state.execution.executions
    parent = replace(_record(ExecutionStatus.STARTED, 1), execution_id="parent", root_execution_id="parent", revision=1, producer_generation=1)
    await repository.create_execution(parent)
    return parent


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_child_admission_keeps_siblings_and_replay_separate(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        first, second = await asyncio.gather(
            repository.reserve_start(_reservation()),
            repository.reserve_start(_reservation(identity="sibling")),
        )
        assert first.created and second.created
        assert {child.execution_id for child in await repository.list_children("parent", tenant_id="tenant")} == {"child", "sibling"}
        await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                                          next_record=replace(parent, status=ExecutionStatus.CANCELLED, revision=2))
        replay = await repository.reserve_start(_reservation(generation=None))
        assert not replay.created and replay.execution == first.execution
        assert await repository.get("child", tenant_id="tenant") == first.execution


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("status", (
    ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED, ExecutionStatus.FINALIZING,
    ExecutionStatus.WAITING_DEFERRED, ExecutionStatus.RECOVERY_REQUIRED,
    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
))
async def test_stopped_parent_cannot_admit_new_child(tmp_path: Path, backend: str, status: ExecutionStatus) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                                          next_record=replace(parent, status=status, revision=2, recovery=(
                                                 ExecutionRecoveryDetail(agent_run_id="run", pending_tools=PendingToolContinuation("run", calls=(
                                                     PendingDeferredCall("call", "tool", StoredPayload.inline_json({}), operation_id="test-operation"),
                                                 ))) if status is ExecutionStatus.WAITING_DEFERRED else parent.recovery)))
        reservation = _reservation()
        with pytest.raises(AIError) as blocked:
            await repository.reserve_start(reservation)
        assert blocked.value.code is (ErrorCode.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLING else ErrorCode.STORAGE_CONFLICT)
        assert await repository.get("child", tenant_id="tenant") is None
        assert await state.execution.idempotency.get("execution.subagent", "child", tenant_id="tenant") is None
        assert await repository.get("child", tenant_id="tenant") is None


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("generation", (None, 2))
async def test_child_admission_requires_exact_parent_generation(tmp_path: Path, backend: str, generation: int | None) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        await _parent(state)
        repository = state.execution.executions
        with pytest.raises(AIError) as blocked:
            await repository.reserve_start(_reservation(generation=generation))
        assert blocked.value.code is (ErrorCode.STORAGE_OWNER_MISMATCH if generation is None else ErrorCode.STORAGE_CONFLICT)
        assert await repository.list_children("parent", tenant_id="tenant") == ()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_parent_takeover_rejects_old_admission_proof(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        replacement = replace(parent, revision=2, agent_run_seq=2, producer_generation=2)

        async def takeover(tx):
            await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=1, next_record=replacement)

        await repository.state_store.mutate(takeover)
        with pytest.raises(AIError) as stale:
            await repository.reserve_start(_reservation())
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert (await repository.reserve_start(_reservation(generation=2))).created


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_parent_cancel_racing_admission_never_misses_an_admitted_child(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        await _parent(state)
        repository = state.execution.executions
        start = asyncio.Event()

        async def admit():
            await start.wait()
            try:
                return await repository.reserve_start(_reservation())
            except AIError as error:
                assert error.code in {ErrorCode.EXECUTION_CANCELLED, ErrorCode.STORAGE_CONFLICT}
                return None

        async def cancel():
            await start.wait()
            while True:
                parent = await repository.state_store.read(lambda tx: repository.get_in_transaction(tx, "parent", tenant_id="tenant"))
                try:
                    await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                        next_record=replace(parent, status=ExecutionStatus.CANCELLING, revision=parent.revision + 1))
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
            return await repository.list_children("parent", tenant_id="tenant")

        admitted = asyncio.create_task(admit())
        cancelled = asyncio.create_task(cancel())
        start.set()
        result, children = await asyncio.wait_for(asyncio.gather(admitted, cancelled), 10)
        assert tuple(child.execution_id for child in children) == (() if result is None else ("child",))
        with pytest.raises(AIError) as stopped:
            await repository.reserve_start(_reservation(identity="late"))
        assert stopped.value.code is ErrorCode.EXECUTION_CANCELLED


async def test_parent_generation_proof_is_not_persisted_or_reconstructed() -> None:
    reservation = _reservation()
    wire = encode_domain(reservation)
    assert set(wire["fields"]) == {"execution", "idempotency"}
    assert decode_domain(wire, ExecutionStartReservation) == replace(reservation, parent_producer_generation=None)
    wire["fields"]["parent_producer_generation"] = 99
    assert decode_domain(wire, ExecutionStartReservation).parent_producer_generation is None


@pytest.mark.parametrize("prepared", (False, True))
async def test_child_attempt_readback_keeps_committed_receipt_after_parent_stops(monkeypatch, prepared: bool) -> None:
    from linktools.ai.persistence.contracts import AgentAttemptClaim, ExecutionRecoveryDetail, ExecutionRecoveryState, RuntimePayloadRef
    from linktools.ai.storage import StoredPayload
    from .test_runtime_approval_recovery import _commands

    async with Persistence.open("child-admission", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        reservation = _reservation()
        await repository.reserve_start(reservation)
        child = replace(reservation.execution, status=ExecutionStatus.STARTED, revision=1, recovery=ExecutionRecoveryDetail())
        await repository.compare_and_swap("child", tenant_id="tenant", expected_revision=0, next_record=child)
        commands = _commands(state, "child-admission")
        group = repository.state_store.storage_group
        mutate = group.mutate
        committed = False

        async def lose_acknowledgement(stores, callback):
            nonlocal committed
            if committed:
                return await mutate(stores, callback)
            committed = True
            await mutate(stores, callback)
            await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                next_record=replace(parent, status=ExecutionStatus.CANCELLED, revision=2))
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_acknowledgement)
        claim = AgentAttemptClaim("child", 1, 0, ExecutionRecoveryState.ADMITTED)
        if prepared:
            active = await commands.commit_prepared_agent_attempt_checkpoint(
                claim, RuntimePayloadRef(StoredPayload.inline_json({"prepared": True}), None),
            )
            active = active.value
        else:
            active = await commands.commit_agent_attempt_checkpoint(claim)
            active = active.value
        assert active.agent_run_seq == 1 and active.recovery_state is ExecutionRecoveryState.ACTIVE
        assert (await repository.get("parent", tenant_id="tenant")).status is ExecutionStatus.CANCELLED
        assert await repository.get("child", tenant_id="tenant") == active


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_reservation_unknown_commit_replays_existing_child_after_parent_stops(tmp_path: Path, backend: str, monkeypatch) -> None:
    async with Persistence.open("child-admission", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        group = repository.state_store.storage_group
        mutate = group.mutate
        committed = False

        async def lose_acknowledgement(stores, callback):
            nonlocal committed
            if committed:
                return await mutate(stores, callback)
            committed = True
            await mutate(stores, callback)
            await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                next_record=replace(parent, status=ExecutionStatus.CANCELLED, revision=2))
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_acknowledgement)
        reservation = _reservation()
        with pytest.raises(AIError) as unknown:
            await repository.reserve_start(reservation)
        assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        replay = await repository.reserve_start(replace(reservation, parent_producer_generation=None))
        assert not replay.created and replay.execution == reservation.execution
        assert await repository.list_children("parent", tenant_id="tenant") == (reservation.execution,)
