#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-SQL pending commands and native SQL terminal publication own distinct facts."""

import asyncio

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    ResourceKind,
    StopReason,
    UsageMetrics,
    agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, Persistence
from linktools.ai.persistence.contracts import ConversationCursor, ExecutionEventAppend, ExecutionStartReservation, ExecutionTerminalCommit, ExecutionTerminalCommitResult, IdempotencyRecord, ExecutionRecoveryDetail, ExecutionRecoveryState, ResultRecord, SessionRecord
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
        producer_generation=2, producer_claim_id="pending-turn",
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


async def _commit(
    state: _TurnState, turn: _PendingTurn, *, include_publication: bool = False,
) -> ExecutionTerminalCommitResult:
    return await state.commands.commit_pending_turn(
        turn.reservation, turn.commit, expected_session=state.session,
        next_cursor=turn.cursor, conversation_agent_run=turn.run,
        conversation_checkpoint=turn.checkpoint, audit_events=turn.audit_events,
        include_publication=include_publication,
    )


def _stores(state: _TurnState) -> tuple[StateStore, ...]:
    storage = state.storage
    return tuple(dict.fromkeys((
        storage.execution.executions.state_store,
        storage.conversation.sessions.state_store,
        storage.execution.executions.state_store,
    )))


async def _snapshot(state: _TurnState) -> tuple[object, ...]:
    async def read(transaction: StateTransaction) -> tuple[object, ...]:
        return (
            tuple(sorted(await transaction.scan_records(), key=lambda value: value.key)),
            tuple(sorted(await transaction.scan_facts(), key=lambda value: (value.stream_key, value.sequence))),
            dict(await transaction.scan_sequences()),
            tuple(sorted(await transaction.scan_operations(), key=lambda value: value.key)),
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
        completed_at=turn.commit.result.created_at,
        recovery=ExecutionRecoveryDetail(agent_run_id=agent_run_id(
            namespace=_NAMESPACE, tenant_id="tenant", execution_id=execution_id, agent_run_seq=1,
        )),
    )
    assert result.result == turn.commit.result
    assert await storage.execution.executions.get(execution_id, tenant_id="tenant") == result.execution
    identity = turn.reservation.idempotency
    committed_identity = await storage.execution.idempotency.get(identity.scope, identity.idempotency_key, tenant_id="tenant")
    assert committed_identity is not None
    assert committed_identity.status is IdempotencyStatus.COMPLETED
    assert committed_identity.resource_id == execution_id
    assert committed_identity.result_digest == result.result.output.digest
    session = await storage.conversation.sessions.get("session", tenant_id="tenant")
    assert session is not None and session.active_execution_id is None
    assert session.continuation is not None
    archive = storage.run_store.read_store(PersistenceDomain.CONVERSATION)
    checkpoint = await archive.checkpoint_at_sequence(
        agent_run_id=turn.run.agent_run_id, checkpoint_sequence=session.continuation.checkpoint_sequence,
    )
    assert session.continuation == replace(
        turn.cursor, checkpoint_sequence=1, projection_digest=checkpoint.projection_digest,
    )
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
    head = await storage.execution.executions.get(execution_id, tenant_id="tenant")
    assert head is not None and (head.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and head.history_seal_digest is None)
    assert result.execution.producer_generation == turn.commit.producer_generation
    assert await storage.execution.executions.get_history_seal(execution_id, tenant_id="tenant") is None
    recovery = await storage.execution.executions.get(execution_id, tenant_id="tenant")
    assert recovery is not None and recovery.recovery_state is ExecutionRecoveryState.COMPLETED
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


async def test_pending_turn_has_one_atomic_write_and_replay_has_no_durable_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _pending_turn(state)
        assert await state.storage.execution.executions.get("execution", tenant_id="tenant") is None
        group = _stores(state)[0].storage_group
        original = type(group).mutate
        writes = []

        async def count_writes(self, stores, callback):
            if self is group:
                writes.append(tuple(stores))
            return await original(self, stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", count_writes)
            result = await _commit(state, turn)
        assert len(writes) == 1
        await _assert_committed(state, turn, result)
        before = await _snapshot(state)
        assert await state.storage.retention.release_execution_handoff("execution", tenant_id="tenant", terminal_commit=True)
        assert await _snapshot(state) == before
        with pytest.raises(AIError) as closed:
            await state.storage.execution.executions.acquire_dependency_hold(
                "execution", tenant_id="tenant", hold_id="late-dependency",
            )
        assert closed.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before
        assert await _commit(state, turn) == result
        assert await _snapshot(state) == before


async def test_pending_turn_same_request_returns_first_execution_and_result(tmp_path: Path) -> None:
    from linktools.ai.persistence.contracts import ExecutionTerminalPublication

    async with _turn_state("memory", tmp_path) as state:
        winner = _pending_turn(state)
        committed = await _commit(state, winner, include_publication=True)
        assert isinstance(committed, ExecutionTerminalPublication)
        assert committed.observed.value == committed.execution
        assert committed.session_released
        before = await _snapshot(state)
        loser = _pending_turn(state, execution_id="loser", answer="duplicate model answer")
        replay = await _commit(state, loser, include_publication=True)
        assert not isinstance(replay, ExecutionTerminalPublication)
        assert replay == ExecutionTerminalCommitResult(committed.execution, committed.result)
        assert await state.storage.execution.executions.get("loser", tenant_id="tenant") is None
        assert await _snapshot(state) == before


async def test_pending_turn_different_request_cannot_replace_advanced_session(tmp_path: Path) -> None:
    async with _turn_state("memory", tmp_path) as state:
        await _commit(state, _pending_turn(state))
        before = await _snapshot(state)
        contender = _pending_turn(state, execution_id="contender", key="c" * 64, request_digest="d" * 64)
        with pytest.raises(AIError) as raised:
            await _commit(state, contender)
        assert raised.value.code is ErrorCode.SESSION_BUSY
        assert await _snapshot(state) == before


@pytest.mark.parametrize(("field", "value"), (
    ("durable_trace", True),
    ("restart_checkpoints", True),
    ("requires_task_invocation_capture", True),
    ("dependency_hold_ids", ("task-dependency",)),
))
async def test_pending_turn_rejects_admitted_requirements_removed_from_terminal(
    tmp_path: Path, field: str, value: object,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _pending_turn(state)
        admitted = replace(turn.reservation.execution, **{field: value})
        assert not getattr(turn.commit.execution, field)
        turn = replace(turn, reservation=replace(turn.reservation, execution=admitted))
        await _assert_rejected_without_writes(state, turn)


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
    tmp_path: Path, target: str, field: str, value: object,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _pending_turn(state)
        changed = replace(getattr(turn, target), **{field: value})
        await _assert_rejected_without_writes(state, replace(turn, **{target: changed}))


@pytest.mark.parametrize("invalid_event", ("terminal_failure", "control_audit"))
async def test_pending_turn_rejects_events_incompatible_with_effect_free_success(
    tmp_path: Path, invalid_event: str,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _pending_turn(state)
        if invalid_event == "terminal_failure":
            turn = replace(turn, commit=replace(turn.commit, terminal_event_type=ExecutionEventType.EXECUTION_FAILED))
        else:
            turn = replace(turn, audit_events=(*turn.audit_events, ExecutionEventAppend(
                ExecutionEventType.APPROVAL_REQUESTED, {"operation_id": "operation"},
            )))
        await _assert_rejected_without_writes(state, turn)


async def test_pending_turn_failure_after_staged_writes_rolls_back_every_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
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


@pytest.mark.parametrize("readback_available", (False, True))
async def test_pending_turn_lost_ack_needs_committed_readback_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readback_available: bool,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
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


@pytest.mark.parametrize("disabled", ("reservation", "terminal", "both"))
async def test_pending_turn_requires_opt_in_on_both_records(
    tmp_path: Path, disabled: str,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
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


async def test_pending_turn_persists_readonly_tool_audit_in_order(
    tmp_path: Path,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
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


@pytest.mark.parametrize("committed", (False, True), ids=("rollback", "lost-ack"))
async def test_native_sql_terminal_keeps_occupancy_and_body_in_one_commit(tmp_path, monkeypatch, committed):
    from sqlalchemy import func, select

    from linktools.ai.persistence.turns import SessionTurnKey
    from .test_session_turn_sql import _TRANSCRIPT, accept

    storage = _storage("sqlite", tmp_path)
    key = SessionTurnKey("chat", "turn")
    async with Persistence.open(_NAMESPACE, tenant_id="tenant", storage=storage) as state:
        turns = state.session_turns
        assert turns is not None and state.conversation.sessions is None
        await turns.create("chat", config={})
        assert await accept(turns, "turn")
        session_before = await turns.get_session("chat")
        turn_before = await turns.get_turn(key)
        engine = turns._context.engine
        table = turns._turns.metadata.tables["ai_messages"]

        async def count_messages():
            async with engine.connect() as connection:
                return (await connection.execute(select(func.count()).select_from(table))).scalar_one()

        original_count = await count_messages()
        group = turns.transcript_state_store.storage_group
        original = type(group).mutate
        fault = ErrorCode.STORAGE_COMMIT_UNKNOWN if committed else ErrorCode.STORAGE_CONFLICT

        async def injected(self, stores, callback):
            async def rollback_after_real_writes(transaction):
                assert await callback(transaction)
                raise AIError(fault)

            if self is not group:
                return await original(self, stores, callback)
            if not committed:
                return await original(self, stores, rollback_after_real_writes)
            assert await original(self, stores, callback)
            raise AIError(fault)

        async def finish():
            return await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", injected)
            with pytest.raises(AIError) as error:
                await finish()
        assert error.value.code is fault
        if not committed:
            assert await turns.get_session("chat") == session_before
            assert await turns.get_turn(key) == turn_before
            assert await count_messages() == original_count
            assert await finish()

    async with Persistence.open(_NAMESPACE, tenant_id="tenant", storage=storage) as reopened:
        turns = reopened.session_turns
        session = await turns.get_session("chat")
        published = await turns.get_history_turn(key, lineage=())
        assert session["active_turn_id"] is None and session["last_turn_no"] == 1
        assert published["state"] == "succeeded"
        reply, = published["messages"]
        assert isinstance(reply, ModelResponse) and reply.text == "two"
        assert not await turns.finish(key, generation=1, turn_no=1, state="succeeded",
                                      transcript=(ModelResponse(parts=[TextPart("duplicate")]),))
        assert await turns.get_history_turn(key, lineage=()) == published
        assert await reopened.execution.executions.get(key.turn_id, tenant_id="tenant") is None


@pytest.mark.parametrize("identity", ("different_request", "same_request", "conflicting_digest"))
async def test_native_sql_independent_connections_choose_one_turn_winner(tmp_path, monkeypatch, identity):
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    class HostPolicy(TenantAuthorizationPolicy):
        """Authorize normal canonical admission across both hosts."""

    provider_calls = []
    release = asyncio.Event()

    async def model(messages, _info):
        provider_calls.append(messages[-1].parts[-1].content)
        await release.wait()
        return ModelResponse(parts=[TextPart("one committed answer")])

    features = model_features(_Models(_UsageFunctionModel(model)), RuntimeFeatures(_NAMESPACE))
    features.agent("default", model="default", allow_tools=())
    storage = _storage("sqlite", tmp_path)
    async with (
        Runtime.open(_NAMESPACE, storage=storage, features=features, authorization=HostPolicy()) as first,
        Runtime.open(_NAMESPACE, storage=storage, features=features, authorization=HostPolicy()) as second,
    ):
        principal = first.default_principal
        await first.sessions.create_turn_session("chat", principal=principal)
        stores = (first._state.session_turns, second._state.session_turns)
        assert stores[0]._context.engine is not stores[1]._context.engine
        original = type(stores[0]).accept
        arrived = set()
        winners = []
        both = asyncio.Event()

        async def meet(self, *args, **kwargs):
            arrived.add(self)
            if len(arrived) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 5)
            accepted = await original(self, *args, **kwargs)
            if accepted:
                winners.append(self)
            return accepted

        try:
            with monkeypatch.context() as patch:
                patch.setattr(type(stores[0]), "accept", meet)
                results = await asyncio.wait_for(asyncio.gather(
                    first.sessions.start("chat", "first", agent="default", principal=principal, idempotency_key="request"),
                    second.sessions.start(
                        "chat", "second" if identity == "conflicting_digest" else "first",
                        agent="default", principal=principal,
                        idempotency_key="different" if identity == "different_request" else "request",
                    ),
                    return_exceptions=True,
                ), 10)
        finally:
            release.set()
        assert arrived == set(stores)
        accepted = [item for item in results if not isinstance(item, BaseException)]
        assert len(accepted) == (2 if identity == "same_request" else 1), results
        if identity == "same_request":
            assert accepted[0].key == accepted[1].key
        else:
            error, = [item for item in results if isinstance(item, BaseException)]
            assert isinstance(error, AIError)
            assert error.code in ({ErrorCode.IDEMPOTENCY_CONFLICT} if identity == "conflicting_digest"
                                  else {ErrorCode.SESSION_CONFLICT, ErrorCode.SESSION_BUSY, ErrorCode.STORAGE_CONFLICT})
        winner, = winners
        owner_handle = results[stores.index(winner)]
        assert owner_handle in accepted
        # Only the producer's runtime has a local live subscription. The
        # other connection must still read the same committed result.
        terminal = await owner_handle.wait_final(timeout_seconds=10)
        assert terminal.status is ExecutionStatus.SUCCEEDED and len(provider_calls) == 1
        for result in accepted:
            assert (await result.result()).view == terminal.view
        for host in (first, second):
            history = await host.history.turn_session("chat", principal=principal)
            assert history["session"]["active_turn_id"] is None
            assert [item["turn_id"] for item in history["timeline"]["items"]] == [accepted[0].key.turn_id]
