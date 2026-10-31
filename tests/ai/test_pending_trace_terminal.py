#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending Trace publication creates and seals only a reservation-owned head."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._step_archive import PreparedExecutionProjection
from linktools.ai.persistence.contracts import ExecutionHistoryHeadRecord, ExecutionHistoryState

from .test_pending_turn_commit import _NAMESPACE, _pending_turn, _snapshot, _turn_state
from .test_pending_turn_outcomes import _outcome
from .test_pending_turn_sessionless import _standalone

pytestmark = pytest.mark.asyncio


async def _trace_turn(state, status, *, sequence=1, sessionless=False):
    successful = status is ExecutionStatus.SUCCEEDED
    turn = ((_standalone if sessionless else _pending_turn)(state) if successful else
            _outcome(state, status, sequence=sequence, sessionless=sessionless))
    run_id = agent_run_id(namespace=_NAMESPACE, tenant_id="tenant", execution_id="execution", agent_run_seq=1)
    turn = replace(
        turn,
        reservation=replace(turn.reservation, execution=replace(turn.reservation.execution, durable_trace=True)),
        commit=replace(turn.commit, execution=replace(
            turn.commit.execution, durable_trace=True, conversation_agent_run_id=run_id if successful and not sessionless else None,
        )),
        run=replace(turn.run, agent_run_id=run_id),
        checkpoint=replace(turn.checkpoint, agent_run_id=run_id),
        cursor=replace(turn.cursor, agent_run_id=run_id),
    )
    archive = state.storage.run_store.read_store(PersistenceDomain.TRACE)
    if sequence == 0:
        return turn, ()
    batch = await archive.prepare_checkpoints(turn.run, (turn.checkpoint,))
    projection = PreparedExecutionProjection(
        turn.run, (), batch.checkpoints, 0, 0, 0, len(batch.checkpoints),
        batch.target_transcript_message_count, durable_projection_digest=batch.checkpoints[-1].projection.digest,
        producer_generation=turn.commit.producer_generation,
    )
    return turn, (projection,)


async def _publish(state, turn, projections):
    successful = turn.commit.execution.status is ExecutionStatus.SUCCEEDED
    sessionless = turn.reservation.execution.session_id is None
    return await state.commands.commit_pending_turn(
        turn.reservation, turn.commit, expected_session=None if sessionless else state.session,
        next_cursor=turn.cursor if successful and not sessionless else None,
        conversation_agent_run=turn.run if successful and not sessionless else None,
        conversation_checkpoint=turn.checkpoint if successful and not sessionless else None,
        recovery_run=turn.run if successful and sessionless else None,
        recovery_agent_run_checkpoint=turn.checkpoint if successful and sessionless else None,
        audit_events=turn.audit_events, execution_projections=projections,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("sessionless", (False, True))
async def test_pending_trace_seals_the_atomic_terminal(tmp_path: Path, backend, status, sessionless) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn, projections = await _trace_turn(state, status, sessionless=sessionless)
        archive = state.storage.run_store.read_store(PersistenceDomain.TRACE)
        assert await archive.get_agent_run(agent_run_id=turn.run.agent_run_id) is None
        assert await state.storage.execution.executions.get("execution", tenant_id="tenant") is None
        result = await _publish(state, turn, projections)
        assert result.execution.status is status
        head = await state.storage.execution.executions.get_history_head("execution", tenant_id="tenant")
        seal = await state.storage.execution.executions.get_history_seal("execution", tenant_id="tenant")
        assert head.state is ExecutionHistoryState.SEALED and head.seal_digest == seal.seal_digest
        assert seal.execution_event_high_water == result.execution.event_seq
        assert seal.run_heads[0].agent_run_id == turn.run.agent_run_id
        assert await archive.verify_execution_projection_head(projections[0])
        before = await _snapshot(state)
        assert await state.commands.read_pending_turn(
            turn.reservation, expected_session=None if sessionless else state.session,
        ) == result
        assert await _publish(state, turn, projections) == result
        assert await _snapshot(state) == before
        if status is not ExecutionStatus.SUCCEEDED and not sessionless:
            session = await state.storage.conversation.sessions.get(state.session.session_id, tenant_id="tenant")
            assert session.continuation == state.session.continuation


@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("sequence", (0, 1))
async def test_pending_trace_failure_before_agent_has_empty_seal(tmp_path: Path, status, sequence) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn, _projections = await _trace_turn(state, status, sequence=sequence)
        result = await _publish(state, turn, ())
        seal = await state.storage.execution.executions.get_history_seal("execution", tenant_id="tenant")
        assert seal.run_heads == ()
        assert await state.commands.read_pending_turn(turn.reservation, expected_session=state.session) == result


@pytest.mark.parametrize("invalid", ("missing", "baseline", "generation", "target", "run", "run_metadata"))
async def test_pending_trace_rejects_invalid_projection_without_partial_commit(tmp_path: Path, invalid) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn, projections = await _trace_turn(state, ExecutionStatus.SUCCEEDED)
        projection = projections[0]
        if invalid == "missing":
            projections = ()
        else:
            changes = {
                "baseline": {"base_event_offset": 1},
                "generation": {"producer_generation": 77},
                "target": {"target_event_offset": 77},
                "run": {"run": replace(projection.run, agent_run_id="foreign")},
                "run_metadata": {"run": replace(projection.run, metadata={"history_id": "wrong"})},
            }
            projections = (replace(projection, **changes[invalid]),)
        before = await _snapshot(state)
        with pytest.raises(AIError) as error:
            await _publish(state, turn, projections)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await _snapshot(state) == before


@pytest.mark.parametrize("readable", (False, True))
async def test_pending_trace_unknown_commit_uses_sealed_source_readback(tmp_path: Path, monkeypatch, readable) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn, projections = await _trace_turn(state, ExecutionStatus.FAILED)
        group = state.storage.execution.executions.state_store.storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        committed = []

        async def lose_ack(self, stores, callback):
            result = await mutate(self, stores, callback)
            if self is group and not committed:
                committed.append(True)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        async def read(self, stores, callback):
            if self is group and committed and not readable:
                raise OSError("source unavailable")
            return await read_many(self, stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lose_ack)
            patch.setattr(type(group), "read_many", read)
            if readable:
                assert (await _publish(state, turn, projections)).execution.status is ExecutionStatus.FAILED
            else:
                with pytest.raises(AIError) as error:
                    await _publish(state, turn, projections)
                assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert committed == [True]
        before = await _snapshot(state)
        assert (await _publish(state, turn, projections)).execution.status is ExecutionStatus.FAILED
        assert await _snapshot(state) == before


@pytest.mark.parametrize("existing", ("execution_head", "run", "transcript_head"))
async def test_pending_trace_does_not_adopt_existing_source_heads(tmp_path: Path, existing) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn, projections = await _trace_turn(state, ExecutionStatus.FAILED)
        repository = state.storage.execution.executions
        archive = state.storage.run_store.read_store(PersistenceDomain.TRACE)
        if existing == "execution_head":
            store = repository.state_store
            record = repository._stored_history_head(ExecutionHistoryHeadRecord(
                "execution", ExecutionHistoryState.OPEN, 0, None, producer_generation=2,
            ))
        else:
            store = archive.state_store
            record = (archive._stored_agent_run(turn.run) if existing == "run" else
                      archive.transcript_repository.empty_head_record(turn.run.agent_run_id))
        await store.mutate(lambda transaction: transaction.insert_records((record,)))
        before = await _snapshot(state)
        with pytest.raises(AIError):
            await _publish(state, turn, projections)
        assert await _snapshot(state) == before
        assert await repository.get("execution", tenant_id="tenant") is None


@pytest.mark.parametrize("missing", ("seal", "trace_head"))
async def test_pending_trace_readback_rejects_partial_source_without_repair(tmp_path: Path, missing) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn, projections = await _trace_turn(state, ExecutionStatus.CANCELLED)
        await _publish(state, turn, projections)
        repository = state.storage.execution.executions
        archive = state.storage.run_store.read_store(PersistenceDomain.TRACE)
        store = repository.state_store if missing == "seal" else archive.state_store
        key = (repository._key("execution_history_seal", "execution") if missing == "seal" else
               archive.transcript_repository.head_key(turn.run.agent_run_id))
        await store.mutate(lambda transaction: transaction.delete_record(key))
        before = await _snapshot(state)
        with pytest.raises(AIError) as error:
            await state.commands.read_pending_turn(turn.reservation, expected_session=state.session)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await _snapshot(state) == before


async def test_pending_trace_requires_same_group_before_source_writes(tmp_path: Path) -> None:
    async with _turn_state("memory", tmp_path / "first") as state:
        async with _turn_state("memory", tmp_path / "second") as foreign:
            turn, projections = await _trace_turn(state, ExecutionStatus.SUCCEEDED)
            state.commands._execution_run_store = foreign.storage.run_store.read_store(PersistenceDomain.TRACE)
            before = (await _snapshot(state), await _snapshot(foreign))
            with pytest.raises(AIError) as error:
                await _publish(state, turn, projections)
            assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert (await _snapshot(state), await _snapshot(foreign)) == before


async def test_pending_trace_without_checkpoints_cannot_adopt_a_prior_projection(tmp_path: Path) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        turn, projections = await _trace_turn(state, ExecutionStatus.FAILED)
        inherited = replace(projections[0], checkpoints=(), durable_projection_digest="a" * 64)
        with pytest.raises(AIError) as error:
            await _publish(state, turn, (inherited,))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await state.storage.execution.executions.get("execution", tenant_id="tenant") is None
