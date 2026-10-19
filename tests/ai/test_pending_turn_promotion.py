#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending turns acquire durable producer authority before effects begin."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.core import ExecutionEventType, ExecutionStatus, IdempotencyStatus, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    AgentAttemptClaim, ExecutionHistoryState, ExecutionRecord,
    RecoveryCheckpoint, RecoveryCheckpointState, RuntimePayloadRef,
)
from linktools.ai.runtime.state._store import active_state_scope
from linktools.ai.storage import StoredPayload

from .test_incremental_projection_boundaries import _storage
from .test_pending_turn_commit import _NAMESPACE, _PendingTurn, _TurnState, _pending_turn, _snapshot, _stores, _turn_state
from .test_runtime_storage_contracts import _runtime_commands

pytestmark = pytest.mark.asyncio


def _context() -> RuntimePayloadRef:
    return RuntimePayloadRef(StoredPayload.inline_json({"prepared": True}), None)


async def _promote(
    state: _TurnState, turn: _PendingTurn, context: RuntimePayloadRef | None,
) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
    return await state.commands.promote_pending_turn(
        turn.reservation, expected_session=state.session, captured_input_context=context,
    )


async def _assert_promoted(
    state: _TurnState, turn: _PendingTurn, context: RuntimePayloadRef | None,
    result: tuple[ExecutionRecord, RecoveryCheckpoint],
) -> None:
    execution, recovery = result
    initial = turn.reservation.execution
    prepared = context is not None
    assert execution == replace(
        initial, status=ExecutionStatus.STARTED, revision=3 if prepared else 1,
        event_seq=1, agent_run_seq=1 if prepared else 0, input_context=context,
        started_at=initial.created_at, updated_at=execution.updated_at,
    )
    assert recovery == RecoveryCheckpoint(
        initial.execution_id,
        agent_run_id(namespace=_NAMESPACE, tenant_id="tenant", execution_id=initial.execution_id, agent_run_seq=1) if prepared else None,
        RecoveryCheckpointState.ACTIVE if prepared else RecoveryCheckpointState.ADMITTED,
        1 if prepared else 0, initial.created_at, execution.updated_at,
    )
    storage = state.storage
    assert await storage.execution.executions.get(initial.execution_id, tenant_id="tenant") == execution
    assert await storage.recovery.checkpoints.get(initial.execution_id, tenant_id="tenant") == recovery
    identity = turn.reservation.idempotency
    assert await storage.execution.idempotency.get(identity.scope, identity.idempotency_key_digest, tenant_id="tenant") == replace(
        identity, status=IdempotencyStatus.STARTED,
    )
    head = await storage.execution.executions.get_history_head(initial.execution_id, tenant_id="tenant")
    assert head is not None and head.state is ExecutionHistoryState.OPEN
    assert head.producer_generation == (2 if prepared else 0)
    assert bool(head.producer_claim_id) is prepared
    assert await storage.conversation.sessions.get(state.session.session_id, tenant_id="tenant") == replace(
        state.session, active_execution_id=initial.execution_id,
    )
    events = (await storage.execution.events.list(initial.execution_id, tenant_id="tenant", after_event_seq=0, limit=20)).items
    assert [(item.event_seq, item.event_type, item.payload) for item in events] == [(1, ExecutionEventType.EXECUTION_STARTED, {})]
    assert await storage.conversation.sessions.timeline_head(state.session.session_id, tenant_id="tenant") == 1
    timeline = await storage.conversation.sessions.list_timeline_turns(
        state.session.session_id, tenant_id="tenant", start_turn_seq=1, end_turn_seq=2,
    )
    assert [item.execution_id for item in timeline] == [initial.execution_id]
    assert not await storage.conversation.sessions.list_timeline_commits(
        state.session.session_id, tenant_id="tenant", start_turn_seq=1, end_turn_seq=2,
    )
    archive = storage.run_store.read_store(RuntimeDomain.CONVERSATION)
    assert await archive.checkpoint_count(agent_run_id=turn.run.agent_run_id) == 0
    assert [item async for item in archive.iter_session_messages("history", tenant_id="tenant")] == []


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("prepared", (False, True))
async def test_pending_turn_promotion_commits_one_boundary_and_can_prepare_after_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, prepared: bool,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        context = _context() if prepared else None
        group = _stores(state)[0].storage_group
        original = type(group).mutate
        boundaries = []
        transactions = set()

        async def count_boundaries(self, stores, callback):
            if self is group and active_state_scope() is None:
                boundaries.append(True)
            return await original(self, stores, callback)

        def capture(connection, _cursor, statement, _parameters, _context, _many):
            if statement.split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                transactions.add(connection.get_transaction())

        engine = group.context.engine.sync_engine if backend == "sqlite" else None
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", count_boundaries)
            if engine is not None:
                event.listen(engine, "before_cursor_execute", capture)
            try:
                result = await _promote(state, turn, context)
            finally:
                if engine is not None:
                    event.remove(engine, "before_cursor_execute", capture)
        assert boundaries == [True]
        if engine is not None:
            assert len(transactions) == 1 and None not in transactions
        await _assert_promoted(state, turn, context, result)
        if not prepared:
            execution, recovery = result
            context = _context()
            activated = await state.commands.commit_prepared_agent_attempt_checkpoint(
                AgentAttemptClaim(execution.execution_id, execution.revision, execution.agent_run_seq,
                                  recovery.revision, recovery.state), context,
            )
            await _assert_promoted(state, turn, context, activated)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("conflict", ("existing_request", "stale_session"))
async def test_pending_turn_promotion_rejects_existing_request_or_changed_session_without_writes(
    tmp_path: Path, backend: str, conflict: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        if conflict == "existing_request":
            existing = _pending_turn(state, execution_id="preexisting")
            reserved = await state.storage.execution.executions.reserve_start(existing.reservation)
            assert reserved.created
        else:
            await state.storage.conversation.sessions.compare_and_swap(
                state.session.session_id, tenant_id="tenant", expected_revision=state.session.revision,
                next_record=replace(state.session, revision=state.session.revision + 1, metadata={"changed": True}),
            )
        before = await _snapshot(state)
        with pytest.raises(AIError) as raised:
            await _promote(state, turn, _context())
        assert raised.value.code is (ErrorCode.STORAGE_CONFLICT if conflict == "existing_request" else ErrorCode.SESSION_BUSY)
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("prepared", (False, True))
async def test_pending_turn_promotion_last_stage_failure_rolls_back_all_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, prepared: bool,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        context = _context() if prepared else None
        group = _stores(state)[0].storage_group
        original = type(group).mutate
        before = await _snapshot(state)
        staged = []

        async def fail_before_commit(self, stores, callback):
            outer = self is group and active_state_scope() is None
            async def guarded(transaction):
                result = await callback(transaction)
                if outer:
                    staged.append(result)
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return result
            return await original(self, stores, guarded)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", fail_before_commit)
            with pytest.raises(AIError) as raised:
                await _promote(state, turn, context)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT and len(staged) == 1
        assert staged[0][0].revision == (3 if prepared else 1)
        assert staged[0][1].state is (RecoveryCheckpointState.ACTIVE if prepared else RecoveryCheckpointState.ADMITTED)
        assert await _snapshot(state) == before
        await _assert_promoted(state, turn, context, await _promote(state, turn, context))


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize("readback_available", (False, True))
async def test_pending_turn_promotion_lost_ack_requires_committed_readback_for_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, prepared: bool, readback_available: bool,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        context = _context() if prepared else None
        group = _stores(state)[0].storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        committed = []
        authorized = []

        async def lose_ack(self, stores, callback):
            outer = self is group and active_state_scope() is None
            result = await mutate(self, stores, callback)
            if outer:
                committed.append(result)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def unavailable_read(self, stores, callback):
            if self is group and committed and not readback_available:
                raise OSError("promotion committed but cannot be confirmed")
            return await read_many(self, stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lose_ack)
            patch.setattr(type(group), "read_many", unavailable_read)
            if readback_available:
                authorized.append(await _promote(state, turn, context))
            else:
                with pytest.raises(AIError) as raised:
                    authorized.append(await _promote(state, turn, context))
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert len(committed) == 1
        assert authorized == (committed if readback_available else [])
        await _assert_promoted(state, turn, context, committed[0])
        before = await _snapshot(state)
        with pytest.raises(AIError) as replay:
            await _promote(state, turn, context)
        assert replay.value.code is ErrorCode.SESSION_BUSY
        assert await _snapshot(state) == before


@pytest.mark.parametrize("prepared", (False, True))
async def test_pending_turn_promotion_independent_connections_grant_only_one_candidate_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: bool,
) -> None:
    async with _turn_state("sqlite", tmp_path) as first:
        second_storage = RuntimeState(_storage("sqlite", tmp_path))
        await second_storage.initialize(namespace=_NAMESPACE, tenant_id="tenant")
        try:
            second = _TurnState(second_storage, _runtime_commands(second_storage, _NAMESPACE), first.session)
            turns = (_pending_turn(first, execution_id="first"), _pending_turn(second, execution_id="second"))
            context = _context() if prepared else None
            groups = tuple(_stores(state)[0].storage_group for state in (first, second))
            assert groups[0] is not groups[1] and groups[0].context.engine is not groups[1].context.engine
            original = type(groups[0]).mutate
            ready = asyncio.Event()
            entered = set()

            async def simultaneous_mutation(self, stores, callback):
                if self in groups and active_state_scope() is None and self not in entered:
                    entered.add(self)
                    if len(entered) == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 5)
                return await original(self, stores, callback)

            with monkeypatch.context() as patch:
                patch.setattr(type(groups[0]), "mutate", simultaneous_mutation)
                results = await asyncio.wait_for(asyncio.gather(
                    _promote(first, turns[0], context), _promote(second, turns[1], context),
                    return_exceptions=True,
                ), 15)
            assert len(entered) == 2
            successful = [item for item in results if isinstance(item, tuple)]
            errors = [item for item in results if isinstance(item, BaseException)]
            assert len(successful) == len(errors) == 1, results
            assert isinstance(errors[0], AIError) and errors[0].code in {ErrorCode.SESSION_BUSY, ErrorCode.STORAGE_CONFLICT}
            winner = successful[0]
            winning_turn = next(turn for turn in turns if turn.reservation.execution.execution_id == winner[0].execution_id)
            await _assert_promoted(first, winning_turn, context, winner)
            losing_id = next(turn.reservation.execution.execution_id for turn in turns if turn is not winning_turn)
            assert await first.storage.execution.executions.get(losing_id, tenant_id="tenant") is None
            assert await first.storage.execution.executions.get_history_head(losing_id, tenant_id="tenant") is None
            assert await first.storage.recovery.checkpoints.get(losing_id, tenant_id="tenant") is None
            assert not (await first.storage.execution.events.list(losing_id, tenant_id="tenant", after_event_seq=0, limit=20)).items
        finally:
            await second_storage.close()
