#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending native turns publish one final durable winner atomically."""

import asyncio

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from sqlalchemy import event

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    ResourceKind,
    StopReason,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, Persistence
from linktools.ai.persistence.contracts import (
    ConversationCursor, ExecutionEventAppend, ExecutionHistoryState,
    ExecutionStartReservation, ExecutionTerminalCommit, ExecutionTerminalCommitResult,
    IdempotencyRecord, RecoveryCheckpointState, ResultRecord, SessionRecord,
)
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence.state import StateStore, StateTransaction
from linktools.ai.storage import StoredPayload

from .test_history_projection_conformance import _record
from .test_incremental_projection_boundaries import _storage
from .test_runtime_storage_contracts import _runtime_commands
from .test_session_admission import _session

pytestmark = pytest.mark.asyncio

_NAMESPACE = "pending-turn"


@dataclass(frozen=True)
class _TurnState:
    storage: Persistence
    commands: StateCommands
    session: SessionRecord


@dataclass(frozen=True)
class _PendingTurn:
    reservation: ExecutionStartReservation
    commit: ExecutionTerminalCommit
    cursor: ConversationCursor
    run: AgentRunRecord
    checkpoint: AgentRunCheckpoint
    audit_events: tuple[ExecutionEventAppend, ...]


@asynccontextmanager
async def _turn_state(backend: str, path: Path) -> AsyncIterator[_TurnState]:
    storage = Persistence(_storage(backend, path))
    await storage.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        session = await storage.conversation.sessions.create(replace(
            _session(), agent_id="default", owner_principal_id="principal", history_id="history",
        ))
        assert await storage.conversation.histories.get("history", tenant_id="tenant") is not None
        yield _TurnState(storage, _runtime_commands(storage, _NAMESPACE), session)
    finally:
        await storage.close()


def _pending_turn(
    state: _TurnState, *, execution_id: str = "execution", key: str = "a" * 64,
    request_digest: str = "b" * 64, answer: str = "done",
) -> _PendingTurn:
    now = state.session.updated_at
    run_id = "conversation-" + execution_id
    pending = replace(
        _record(ExecutionStatus.PENDING_START, 0),
        execution_id=execution_id, root_execution_id=execution_id,
        session_id=state.session.session_id, conversation_agent_run_id=run_id,
        created_at=now, updated_at=now, durable_trace=False, conversation=True,
    )
    reservation = ExecutionStartReservation(pending, IdempotencyRecord(
        "execution.start", key, request_digest, ResourceKind.EXECUTION,
        execution_id, IdempotencyStatus.RESERVED, None, None, now, now,
    ))
    finished_at = now + timedelta(seconds=1)
    result = ResultRecord(StoredPayload.inline_text(answer), StopReason.END_TURN, UsageMetrics(), finished_at)
    terminal = replace(
        pending, status=ExecutionStatus.SUCCEEDED, revision=4, event_seq=2,
        agent_run_seq=1, started_at=now, updated_at=finished_at,
    )
    commit = ExecutionTerminalCommit(
        3, 1, terminal, result, ExecutionEventType.EXECUTION_SUCCEEDED,
        {"stop_reason": StopReason.END_TURN.value}, producer_generation=2,
    )
    run = AgentRunRecord(
        run_id, agent_conversation_id="conversation", agent_id="default",
        metadata={"history_id": "history"}, started_at=now,
    )
    checkpoint = AgentRunCheckpoint(
        run_id, 1,
        [ModelRequest(parts=[UserPromptPart("prompt")]), ModelResponse(parts=[TextPart(answer)])],
        agent_conversation_id="conversation", agent_id="default", timestamp=finished_at,
        transcript_message_count_before=0,
    )
    return _PendingTurn(
        reservation, commit, ConversationCursor(run_id, "history", 2), run, checkpoint,
        (ExecutionEventAppend(ExecutionEventType.MODEL_REQUEST_FINISHED, {"model_request_seq": 1}),),
    )


async def _commit(state: _TurnState, turn: _PendingTurn) -> ExecutionTerminalCommitResult:
    return await state.commands.commit_pending_turn(
        turn.reservation, turn.commit, expected_session=state.session,
        next_cursor=turn.cursor, conversation_agent_run=turn.run,
        conversation_checkpoint=turn.checkpoint, audit_events=turn.audit_events,
    )


def _stores(state: _TurnState) -> tuple[StateStore, ...]:
    storage = state.storage
    return tuple(dict.fromkeys((
        storage.execution.executions.state_store,
        storage.conversation.sessions.state_store,
        storage.recovery.checkpoints.state_store,
    )))


async def _snapshot(state: _TurnState) -> tuple[object, ...]:
    async def read(transaction: StateTransaction) -> tuple[object, ...]:
        return (
            tuple(sorted(await transaction.scan_records(), key=lambda value: value.key_digest)),
            tuple(sorted(await transaction.scan_facts(), key=lambda value: (value.stream_digest, value.sequence))),
            dict(await transaction.scan_sequences()),
            tuple(sorted(await transaction.scan_aliases(), key=lambda value: value.alias_digest)),
            tuple(sorted(await transaction.scan_operations(), key=lambda value: value.key_digest)),
        )
    return tuple([await store.read(read) for store in dict.fromkeys(state.storage._stores.values())])


async def _assert_rejected_without_writes(state: _TurnState, turn: _PendingTurn) -> None:
    before = await _snapshot(state)
    with pytest.raises(AIError) as raised:
        await _commit(state, turn)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert await _snapshot(state) == before


async def _assert_committed(state: _TurnState, turn: _PendingTurn, result: ExecutionTerminalCommitResult) -> None:
    storage = state.storage
    execution_id = turn.reservation.execution.execution_id
    assert result.execution.execution_id == execution_id
    assert result.execution.status is ExecutionStatus.SUCCEEDED
    assert result.execution == replace(
        turn.commit.execution, revision=4 + len(turn.audit_events),
        event_seq=2 + len(turn.audit_events), result=turn.commit.result, retention_closed=True,
    )
    assert result.result == turn.commit.result
    assert await storage.execution.executions.get(execution_id, tenant_id="tenant") == result.execution
    identity = turn.reservation.idempotency
    committed_identity = await storage.execution.idempotency.get(identity.scope, identity.idempotency_key_digest, tenant_id="tenant")
    assert committed_identity is not None
    assert committed_identity.status is IdempotencyStatus.COMPLETED
    assert committed_identity.resource_id == execution_id
    assert committed_identity.result_digest == result.result.output.digest
    session = await storage.conversation.sessions.get("session", tenant_id="tenant")
    assert session is not None and session.active_execution_id is None
    assert session.continuation == turn.cursor
    assert session.revision == state.session.revision + 1
    assert await storage.conversation.sessions.timeline_head("session", tenant_id="tenant") == 1
    timeline = await storage.conversation.sessions.list_timeline_turns("session", tenant_id="tenant", start_turn_seq=1, end_turn_seq=2)
    assert [item.execution_id for item in timeline] == [execution_id]
    commits = await storage.conversation.sessions.list_timeline_commits("session", tenant_id="tenant", start_turn_seq=1, end_turn_seq=2)
    assert [(item.execution_id, item.start_message_index, item.end_message_index) for item in commits] == [(execution_id, 0, 2)]
    archive = storage.run_store.read_store(PersistenceDomain.CONVERSATION)
    assert await archive.checkpoint_count(agent_run_id=turn.run.agent_run_id) == 1
    messages = [item async for item in archive.iter_session_messages("history", tenant_id="tenant")]
    assert messages == turn.checkpoint.messages
    head = await storage.execution.executions.get_history_head(execution_id, tenant_id="tenant")
    assert head is not None and head.state is ExecutionHistoryState.CLOSED
    assert head.producer_generation == turn.commit.producer_generation
    assert await storage.execution.executions.get_history_seal(execution_id, tenant_id="tenant") is None
    recovery = await storage.recovery.checkpoints.get(execution_id, tenant_id="tenant")
    assert recovery is not None and recovery.state is RecoveryCheckpointState.COMPLETED
    events = (await storage.execution.events.list(execution_id, tenant_id="tenant", after_event_seq=0, limit=20)).items
    assert [item.event_type for item in events] == [
        ExecutionEventType.EXECUTION_STARTED,
        *(item.event_type for item in turn.audit_events),
        ExecutionEventType.EXECUTION_SUCCEEDED,
    ]
    assert [item.event_seq for item in events] == list(range(1, len(events) + 1))
    assert result.execution.event_seq == len(events)
    for expected in turn.audit_events:
        assert sum(item.event_type == expected.event_type and item.payload == expected.payload for item in events) == 1


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pending_turn_has_one_atomic_write_and_replay_has_no_durable_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        assert await state.storage.execution.executions.get("execution", tenant_id="tenant") is None
        group = _stores(state)[0].storage_group
        original = type(group).mutate
        writes = []

        async def count_writes(self, stores, callback):
            if self is group:
                writes.append(tuple(stores))
            return await original(self, stores, callback)

        transactions = set()
        def capture(connection, _cursor, statement, _parameters, _context, _many):
            if statement.split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                transactions.add(connection.get_transaction())

        engine = group.context.engine.sync_engine if backend == "sqlite" else None
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", count_writes)
            if engine is not None:
                event.listen(engine, "before_cursor_execute", capture)
            try:
                result = await _commit(state, turn)
            finally:
                if engine is not None:
                    event.remove(engine, "before_cursor_execute", capture)
        assert len(writes) == 1
        if engine is not None:
            assert len(transactions) == 1 and None not in transactions
        await _assert_committed(state, turn, result)
        before = await _snapshot(state)
        if engine is not None:
            event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await state.storage.retention.release_execution_handoff("execution", tenant_id="tenant", terminal_commit=True)
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        assert await _snapshot(state) == before
        if engine is not None:
            assert len(transactions) == 1
        with pytest.raises(AIError) as closed:
            await state.storage.execution.executions.acquire_dependency_hold(
                "execution", tenant_id="tenant", hold_id="late-dependency",
            )
        assert closed.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before
        assert await _commit(state, turn) == result
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pending_turn_same_request_returns_first_execution_and_result(tmp_path: Path, backend: str) -> None:
    async with _turn_state(backend, tmp_path) as state:
        winner = _pending_turn(state)
        committed = await _commit(state, winner)
        before = await _snapshot(state)
        loser = _pending_turn(state, execution_id="loser", answer="duplicate model answer")
        assert await _commit(state, loser) == committed
        assert await state.storage.execution.executions.get("loser", tenant_id="tenant") is None
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pending_turn_different_request_cannot_replace_advanced_session(tmp_path: Path, backend: str) -> None:
    async with _turn_state(backend, tmp_path) as state:
        await _commit(state, _pending_turn(state))
        before = await _snapshot(state)
        contender = _pending_turn(state, execution_id="contender", key="c" * 64, request_digest="d" * 64)
        with pytest.raises(AIError) as raised:
            await _commit(state, contender)
        assert raised.value.code is ErrorCode.SESSION_BUSY
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize(("field", "value"), (
    ("durable_trace", True),
    ("restart_checkpoints", True),
    ("requires_task_invocation_capture", True),
    ("dependency_hold_ids", ("task-dependency",)),
))
async def test_pending_turn_rejects_admitted_requirements_removed_from_terminal(
    tmp_path: Path, backend: str, field: str, value: object,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        admitted = replace(turn.reservation.execution, **{field: value})
        assert not getattr(turn.commit.execution, field)
        turn = replace(turn, reservation=replace(turn.reservation, execution=admitted))
        await _assert_rejected_without_writes(state, turn)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize(("target", "field", "value"), (
    ("run", "metadata", {"history_id": "wrong-history"}),
    ("checkpoint", "state", "interrupted"),
    ("checkpoint", "pending_request_index", 0),
    ("run", "agent_id", "wrong-agent"),
    ("checkpoint", "agent_id", "wrong-agent"),
    ("run", "agent_conversation_id", "wrong-conversation"),
    ("checkpoint", "agent_conversation_id", "wrong-conversation"),
    ("run", "parent_agent_run_id", "parent-run"),
    ("checkpoint", "parent_agent_run_id", "parent-run"),
))
async def test_pending_turn_rejects_inconsistent_conversation_checkpoint(
    tmp_path: Path, backend: str, target: str, field: str, value: object,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        changed = replace(getattr(turn, target), **{field: value})
        await _assert_rejected_without_writes(state, replace(turn, **{target: changed}))


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("invalid_event", ("terminal_failure", "control_audit"))
async def test_pending_turn_rejects_events_incompatible_with_effect_free_success(
    tmp_path: Path, backend: str, invalid_event: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        if invalid_event == "terminal_failure":
            turn = replace(turn, commit=replace(turn.commit, terminal_event_type=ExecutionEventType.EXECUTION_FAILED))
        else:
            turn = replace(turn, audit_events=(*turn.audit_events, ExecutionEventAppend(
                ExecutionEventType.APPROVAL_REQUESTED, {"operation_id": "operation"},
            )))
        await _assert_rejected_without_writes(state, turn)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pending_turn_failure_after_staged_writes_rolls_back_every_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        before = await _snapshot(state)
        group = _stores(state)[0].storage_group
        original = type(group).mutate
        failures = []

        async def fail_before_commit(self, stores, callback):
            if self is not group:
                return await original(self, stores, callback)
            async def fail(transaction):
                await callback(transaction)
                failures.append(True)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original(self, stores, fail)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", fail_before_commit)
            with pytest.raises(AIError) as raised:
                await _commit(state, turn)
        assert failures and raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before
        await _assert_committed(state, turn, await _commit(state, turn))


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("readback_available", (False, True))
async def test_pending_turn_lost_ack_needs_committed_readback_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, readback_available: bool,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        group = _stores(state)[0].storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        committed = []

        async def lose_ack(self, stores, callback):
            result = await mutate(self, stores, callback)
            if self is group and not committed:
                committed.append(True)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def unavailable_read(self, stores, callback):
            if self is group and committed and not readback_available:
                raise OSError("committed state is temporarily unreadable")
            return await read_many(self, stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lose_ack)
            patch.setattr(type(group), "read_many", unavailable_read)
            if readback_available:
                result = await _commit(state, turn)
                assert result.execution.status is ExecutionStatus.SUCCEEDED
            else:
                with pytest.raises(AIError) as raised:
                    await _commit(state, turn)
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert committed == [True]
        before = await _snapshot(state)
        result = await _commit(state, turn)
        await _assert_committed(state, turn, result)
        assert await _snapshot(state) == before


@pytest.mark.parametrize("identity", ("different_request", "same_request", "conflicting_digest"))
async def test_pending_turn_independent_connections_choose_one_session_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str,
) -> None:
    async with _turn_state("sqlite", tmp_path) as first:
        second_storage = Persistence(_storage("sqlite", tmp_path))
        await second_storage.initialize(namespace=_NAMESPACE, tenant_id="tenant")
        try:
            second = _TurnState(second_storage, _runtime_commands(second_storage, _NAMESPACE), first.session)
            contenders = (
                _pending_turn(first, execution_id="first", answer="first answer"),
                _pending_turn(second, execution_id="second", answer="second answer",
                              key="c" * 64 if identity == "different_request" else "a" * 64,
                              request_digest="b" * 64 if identity == "same_request" else "d" * 64),
            )
            groups = tuple(_stores(state)[0].storage_group for state in (first, second))
            assert groups[0] is not groups[1]
            assert groups[0].context.engine is not groups[1].context.engine
            original = type(groups[0]).mutate
            ready = asyncio.Event()
            entered = set()

            async def simultaneous_mutation(self, stores, callback):
                if self in groups and self not in entered:
                    entered.add(self)
                    if len(entered) == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 5)
                return await original(self, stores, callback)

            with monkeypatch.context() as patch:
                patch.setattr(type(groups[0]), "mutate", simultaneous_mutation)
                results = await asyncio.wait_for(asyncio.gather(
                    _commit(first, contenders[0]), _commit(second, contenders[1]),
                    return_exceptions=True,
                ), 15)
            assert len(entered) == 2
            successful = [item for item in results if isinstance(item, ExecutionTerminalCommitResult)]
            assert len(successful) == (2 if identity == "same_request" else 1), results
            winner = successful[0]
            if identity == "same_request":
                assert successful[0] == successful[1]
            else:
                errors = [item for item in results if isinstance(item, BaseException)]
                assert len(errors) == 1 and isinstance(errors[0], AIError)
                if identity == "conflicting_digest":
                    assert errors[0].code is ErrorCode.IDEMPOTENCY_CONFLICT
                else:
                    assert errors[0].code in {ErrorCode.SESSION_BUSY, ErrorCode.STORAGE_CONFLICT}
            committed_turn = next(turn for turn in contenders if turn.reservation.execution.execution_id == winner.execution.execution_id)
            await _assert_committed(first, committed_turn, winner)
            losing_id = next(turn.reservation.execution.execution_id for turn in contenders if turn is not committed_turn)
            assert await first.storage.execution.executions.get(losing_id, tenant_id="tenant") is None
            archive = first.storage.run_store.read_store(PersistenceDomain.CONVERSATION)
            losing_run = next(turn.run.agent_run_id for turn in contenders if turn is not committed_turn)
            assert await archive.get_agent_run(agent_run_id=losing_run) is None
            assert await archive.checkpoint_count(agent_run_id=losing_run) == 0
            before = await _snapshot(first)
            assert await _commit(first, committed_turn) == winner
            assert await _snapshot(first) == before
        finally:
            await second_storage.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("disabled", ("reservation", "terminal", "both"))
async def test_pending_turn_requires_opt_in_on_both_records(
    tmp_path: Path, backend: str, disabled: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        if disabled in {"reservation", "both"}:
            turn = replace(turn, reservation=replace(
                turn.reservation, execution=replace(turn.reservation.execution, conversation=False),
            ))
        if disabled in {"terminal", "both"}:
            turn = replace(turn, commit=replace(
                turn.commit, execution=replace(turn.commit.execution, conversation=False),
            ))
        await _assert_rejected_without_writes(state, turn)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_pending_turn_persists_readonly_tool_audit_in_order(
    tmp_path: Path, backend: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        turn = replace(turn, audit_events=(
            *turn.audit_events,
            ExecutionEventAppend(ExecutionEventType.TOOL_CALL_STARTED, {
                "call_id": "readonly-call", "tool_name": "lookup", "arguments_digest": "a" * 64,
            }),
            ExecutionEventAppend(ExecutionEventType.TOOL_CALL_FINISHED, {
                "call_id": "readonly-call", "tool_name": "lookup",
                "result_digest": "b" * 64, "status": "SUCCEEDED",
            }),
        ))
        result = await _commit(state, turn)
        await _assert_committed(state, turn, result)
        before = await _snapshot(state)
        assert await _commit(state, turn) == result
        assert await _snapshot(state) == before
