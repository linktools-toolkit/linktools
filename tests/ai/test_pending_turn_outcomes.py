#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending failures and cancellation publish canonical outcomes, not continuation."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus, IdempotencyStatus, StopReason, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.contracts import RecoveryCheckpointState
from linktools.ai.persistence._runtime_commands import StateCommands

from .test_pending_turn_commit import _NAMESPACE, _commit, _pending_turn, _snapshot, _turn_state
from .test_pending_turn_sessionless import _standalone

pytestmark = pytest.mark.asyncio


def _outcome(state, status, *, sessionless=False, sequence=1, execution_id="execution", key="a" * 64):
    turn = (_standalone if sessionless else _pending_turn)(state, execution_id=execution_id, key=key)
    code = ErrorCode.EXECUTION_CANCELLED.value if status is ExecutionStatus.CANCELLED else ErrorCode.MODEL_API_ERROR.value
    commit = replace(
        turn.commit,
        execution=replace(turn.commit.execution, status=status, agent_run_seq=sequence,
                          conversation_agent_run_id=None, error_code=code,
                          safe_error_details={"phase": "model"}),
        result=replace(turn.commit.result, output=None,
                       stop_reason=StopReason.CANCELLED if status is ExecutionStatus.CANCELLED else StopReason.ERROR),
        terminal_event_type=(ExecutionEventType.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED
                             else ExecutionEventType.EXECUTION_FAILED),
        terminal_event_payload={"error_code": code, "safe_error_details": {"phase": "model"}},
    )
    return replace(turn, commit=commit, audit_events=() if sequence == 0 else turn.audit_events)


async def _publish(state, turn, *, sessionless=False):
    return await state.commands.commit_pending_turn(
        turn.reservation, turn.commit,
        expected_session=None if sessionless else state.session,
        audit_events=turn.audit_events,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("sessionless", (False, True))
@pytest.mark.parametrize("sequence", (0, 1))
async def test_pending_unsuccessful_outcome_and_source_replay(
    tmp_path: Path, backend: str, status: ExecutionStatus, sessionless: bool, sequence: int,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _outcome(state, status, sessionless=sessionless, sequence=sequence)
        result = await _publish(state, turn, sessionless=sessionless)
        assert result.execution.status is status
        assert result.result.output is None and result.execution.retention_closed
        identity = turn.reservation.idempotency
        stored = await state.storage.execution.idempotency.get(
            identity.scope, identity.idempotency_key_digest, tenant_id="tenant",
        )
        assert stored.status is (IdempotencyStatus.FAILED if status is ExecutionStatus.FAILED
                                 else IdempotencyStatus.CANCELLED)
        assert stored.result_digest is None and stored.error_code == result.execution.error_code
        recovery = await state.storage.recovery.checkpoints.get(result.execution.execution_id, tenant_id="tenant")
        assert recovery.state is RecoveryCheckpointState.COMPLETED
        assert recovery.agent_run_id == (None if sequence == 0 else agent_run_id(
            namespace=_NAMESPACE, tenant_id="tenant", execution_id=result.execution.execution_id, agent_run_seq=1,
        ))
        if not sessionless:
            session = await state.storage.conversation.sessions.get(state.session.session_id, tenant_id="tenant")
            assert session.continuation == state.session.continuation
            assert session.history_quality == state.session.history_quality
            committed = await state.storage.conversation.sessions.state_store.read(
                lambda transaction: state.storage.conversation.sessions.pending_turn_commit_in_transaction(
                    transaction, session_id=state.session.session_id, execution_id=result.execution.execution_id,
                )
            )
            assert committed.status is status and committed.output is None
            assert committed.start_message_index == committed.end_message_index == 0
        before = await _snapshot(state)
        assert await state.commands.read_pending_turn(
            turn.reservation, expected_session=None if sessionless else state.session,
        ) == result
        assert await _publish(state, turn, sessionless=sessionless) == result
        assert await _snapshot(state) == before


@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
async def test_pending_failure_keeps_prior_successful_continuation(tmp_path: Path, status: ExecutionStatus) -> None:
    async with _turn_state("memory", tmp_path) as state:
        await _commit(state, _pending_turn(state, execution_id="prior", key="c" * 64))
        session = await state.storage.conversation.sessions.get(state.session.session_id, tenant_id="tenant")
        assert session.continuation is not None
        state = replace(state, session=session)
        await _publish(state, _outcome(state, status))
        after = await state.storage.conversation.sessions.get(session.session_id, tenant_id="tenant")
        assert after.continuation == session.continuation and after.history_quality == session.history_quality


@pytest.mark.parametrize("readback_available", (False, True))
@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
async def test_pending_unsuccessful_lost_ack_requires_source_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readback_available: bool, status: ExecutionStatus,
) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _outcome(state, status, sequence=0)
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
            if self is group and committed and not readback_available:
                raise OSError("source unavailable")
            return await read_many(self, stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", lose_ack)
            patch.setattr(type(group), "read_many", read)
            if readback_available:
                assert (await _publish(state, turn)).execution.status is status
            else:
                with pytest.raises(AIError) as error:
                    await _publish(state, turn)
                assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert committed == [True]
        before = await _snapshot(state)
        assert (await _publish(state, turn)).execution.status is status
        assert await _snapshot(state) == before


@pytest.mark.parametrize("invalid", ("output", "event", "cursor"))
async def test_pending_failure_rejects_success_artifacts(tmp_path: Path, invalid: str) -> None:
    async with _turn_state("memory", tmp_path) as state:
        turn = _outcome(state, ExecutionStatus.FAILED)
        before = await _snapshot(state)
        with pytest.raises((AIError, ValueError)) as error:
            kwargs = {}
            if invalid == "output":
                turn = replace(turn, commit=replace(turn.commit, result=_pending_turn(state).commit.result))
            elif invalid == "event":
                turn = replace(turn, commit=replace(turn.commit, terminal_event_type=ExecutionEventType.EXECUTION_SUCCEEDED))
            else:
                kwargs["next_cursor"] = turn.cursor
            await state.commands.commit_pending_turn(
                turn.reservation, turn.commit, expected_session=state.session, **kwargs,
            )
        if isinstance(error.value, AIError):
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await _snapshot(state) == before



async def test_pending_failure_preserves_session_compare_and_swap(tmp_path: Path) -> None:
    async with _turn_state("memory", tmp_path) as state:
        failed = _outcome(state, ExecutionStatus.FAILED)
        await _commit(state, _pending_turn(state, execution_id="winner", key="d" * 64))
        before = await _snapshot(state)
        with pytest.raises(AIError) as error:
            await _publish(state, failed)
        assert error.value.code is ErrorCode.SESSION_BUSY
        assert await _snapshot(state) == before


@pytest.mark.parametrize("status", (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
async def test_pending_failure_requires_conversation_repository(tmp_path: Path, status: ExecutionStatus) -> None:
    async with _turn_state("memory", tmp_path) as state:
        commands = StateCommands(
            state.storage.execution.executions, namespace=_NAMESPACE,
            events=state.storage.execution.events, recovery=state.storage.recovery.checkpoints,
            background_tasks=set(),
        )
        turn = _outcome(state, status)
        before = await _snapshot(state)
        with pytest.raises(AIError) as error:
            await commands.commit_pending_turn(turn.reservation, turn.commit, expected_session=state.session)
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert await _snapshot(state) == before
