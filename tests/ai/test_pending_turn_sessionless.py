#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initial standalone turns commit their outcome and canonical fork context together."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.core import ExecutionStatus, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import ExecutionTerminalCommitResult, RecoveryCheckpointState, RuntimeDomain, RuntimePayloadRef
from linktools.ai.storage import StoredPayload

from tests.ai.test_pending_turn_commit import _NAMESPACE, _TurnState, _pending_turn, _runtime_commands, _snapshot, _storage, _turn_state
pytestmark = pytest.mark.asyncio


def _standalone(state, **kwargs):
    turn = _pending_turn(state, **kwargs)
    run_id = agent_run_id(
        namespace=_NAMESPACE, tenant_id="tenant",
        execution_id=turn.reservation.execution.execution_id, agent_run_seq=1,
    )
    return replace(
        turn,
        run=replace(turn.run, agent_run_id=run_id, agent_conversation_id=None, metadata={}),
        checkpoint=replace(turn.checkpoint, agent_run_id=run_id, agent_conversation_id=None),
        reservation=replace(turn.reservation, execution=replace(
            turn.reservation.execution, session_id=None, conversation_agent_run_id=None,
        )),
        commit=replace(turn.commit, execution=replace(
            turn.commit.execution, session_id=None, conversation_agent_run_id=None,
        )),
    )


async def _commit(state, turn, **kwargs):
    return await state.commands.commit_pending_turn(
        turn.reservation, turn.commit, audit_events=turn.audit_events,
        recovery_run=turn.run, recovery_agent_run_checkpoint=turn.checkpoint, **kwargs,
    )


async def _conversation(state):
    store = state.storage.conversation.sessions.state_store

    async def read(transaction):
        return (await transaction.scan_records(), await transaction.scan_facts(),
                await transaction.scan_sequences())

    return await store.read(read)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_sessionless_pending_commit_replay_and_unique_result(tmp_path: Path, backend: str):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state)
        before = await _conversation(state)
        transactions = set()
        engine = state.storage.execution.executions.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None

        def capture(connection, _cursor, sql, _parameters, _context, _many):
            if sql.split()[0] in {"INSERT", "UPDATE", "DELETE"}:
                transactions.add(connection.get_transaction())

        if engine is not None:
            event.listen(engine, "before_cursor_execute", capture)
        try:
            result = await _commit(state, turn)
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        assert result.execution.status is ExecutionStatus.SUCCEEDED
        assert result.execution.session_id is None and result.execution.conversation_agent_run_id is None
        assert result.execution.retention_closed and result.result == turn.commit.result
        if engine is not None:
            assert len(transactions) == 1 and None not in transactions
        recovery = await state.storage.recovery.checkpoints.get("execution", tenant_id="tenant")
        assert recovery is not None and recovery.state is RecoveryCheckpointState.COMPLETED
        assert await _conversation(state) == before
        committed = await _snapshot(state)
        assert await _commit(state, turn) == result
        assert await _commit(state, _standalone(state, execution_id="loser", answer="other")) == result
        assert await _snapshot(state) == committed
        assert await state.storage.execution.executions.get("loser", tenant_id="tenant") is None


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("invalid", ("session", "cursor", "run", "checkpoint", "previous", "fork", "root", "execution_session"))
async def test_sessionless_pending_rejects_partial_context_and_lineage(tmp_path: Path, backend: str, invalid: str):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state)
        kwargs = {}
        if invalid in {"session", "cursor", "run", "checkpoint"}:
            name, value = {
                "session": ("expected_session", state.session),
                "cursor": ("next_cursor", turn.cursor),
                "run": ("conversation_agent_run", turn.run),
                "checkpoint": ("conversation_checkpoint", turn.checkpoint),
            }[invalid]
            kwargs[name] = value
        else:
            field = {"previous": "previous_execution_id", "fork": "fork_base_execution_id",
                     "root": "root_execution_id", "execution_session": "session_id"}[invalid]
            turn = replace(turn, commit=replace(turn.commit, execution=replace(turn.commit.execution, **{field: "other"})))
        before = await _snapshot(state)
        with pytest.raises(AIError) as raised:
            await _commit(state, turn, **kwargs)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_sessionless_pending_failure_rolls_back_every_domain(tmp_path: Path, backend: str, monkeypatch):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state)
        before = await _snapshot(state)
        repository = state.storage.recovery.checkpoints
        original = type(repository).admit_in_transaction

        async def fail(self, transaction, record):
            result = await original(self, transaction, record)
            if self is repository:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(type(repository), "admit_in_transaction", fail)
            with pytest.raises(AIError) as raised:
                await _commit(state, turn)
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("readable", (True, False))
async def test_sessionless_pending_lost_ack_requires_strong_confirmation(tmp_path: Path, backend: str, readable: bool, monkeypatch):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state)
        store = state.storage.execution.executions.state_store
        group = store.storage_group
        original_mutate, original_read = type(group).mutate, type(store).read
        lost = False

        async def mutate(self, stores, callback):
            nonlocal lost
            result = await original_mutate(self, stores, callback)
            if self is group:
                lost = True
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def read(self, callback):
            if self is store and lost and not readable:
                raise RuntimeError("source read unavailable")
            return await original_read(self, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", mutate)
            patch.setattr(type(store), "read", read)
            if readable:
                assert (await _commit(state, turn)).execution.status is ExecutionStatus.SUCCEEDED
            else:
                with pytest.raises(AIError) as raised:
                    await _commit(state, turn)
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert (await _commit(state, turn)).execution.status is ExecutionStatus.SUCCEEDED


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("prepared", (False, True))
async def test_sessionless_pending_promotes_before_effects(tmp_path: Path, backend: str, prepared: bool):
    async with _turn_state(backend, tmp_path) as state:
        turn = _standalone(state)
        before = await _conversation(state)
        context = RuntimePayloadRef(StoredPayload.inline_bytes(b"[]"), RuntimeDomain.EXECUTION) if prepared else None
        execution, recovery = await state.commands.promote_pending_turn(
            turn.reservation, expected_session=None, captured_input_context=context,
        )
        assert execution.session_id is None
        assert execution.revision == (3 if prepared else 1)
        assert execution.agent_run_seq == (1 if prepared else 0)
        assert recovery.state is (RecoveryCheckpointState.ACTIVE if prepared else RecoveryCheckpointState.ADMITTED)
        assert await _conversation(state) == before

@pytest.mark.parametrize("identity", ("same_request", "different_request", "conflicting_digest"))
async def test_sessionless_pending_independent_connections_obey_request_identity(
    tmp_path: Path, monkeypatch, identity: str,
):
    async with _turn_state("sqlite", tmp_path) as first:
        other = RuntimeState(_storage("sqlite", tmp_path))
        await other.initialize(namespace=_NAMESPACE, tenant_id="tenant")
        try:
            second = _TurnState(other, _runtime_commands(other, _NAMESPACE), first.session)
            turns = (
                _standalone(first, execution_id="first"),
                _standalone(second, execution_id="second",
                            key="c" * 64 if identity == "different_request" else "a" * 64,
                            request_digest="d" * 64 if identity == "conflicting_digest" else "b" * 64),
            )
            groups = (first.storage.execution.executions.state_store.storage_group,
                      second.storage.execution.executions.state_store.storage_group)
            assert groups[0].context.engine is not groups[1].context.engine
            original = type(groups[0]).mutate
            ready, entered = asyncio.Event(), set()
            before = await _conversation(first)

            async def mutate(self, stores, callback):
                if self in groups and self not in entered:
                    entered.add(self)
                    if len(entered) == 2:
                        ready.set()
                    await asyncio.wait_for(ready.wait(), 5)
                return await original(self, stores, callback)

            with monkeypatch.context() as patch:
                patch.setattr(type(groups[0]), "mutate", mutate)
                results = await asyncio.wait_for(asyncio.gather(
                    _commit(first, turns[0]), _commit(second, turns[1]), return_exceptions=True,
                ), 15)
            successful = [value for value in results if isinstance(value, ExecutionTerminalCommitResult)]
            assert len(successful) == (1 if identity == "conflicting_digest" else 2), results
            if identity == "same_request":
                assert successful[0] == successful[1]
            elif identity == "different_request":
                assert {value.execution.execution_id for value in successful} == {"first", "second"}
            else:
                errors = [value for value in results if isinstance(value, AIError)]
                assert len(errors) == 1 and errors[0].code is ErrorCode.IDEMPOTENCY_CONFLICT
            if identity != "different_request":
                loser = ({"first", "second"} - {successful[0].execution.execution_id}).pop()
                assert await first.storage.execution.executions.get(loser, tenant_id="tenant") is None
            assert await _conversation(first) == before
        finally:
            await other.close()
