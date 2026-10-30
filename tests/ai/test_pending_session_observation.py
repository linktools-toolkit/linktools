#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local observation detects admission ABA without settling uncertain commits."""

from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, IdempotencyStatus, ResourceKind
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.contracts import ExecutionStartReservation, IdempotencyRecord
from linktools.ai.runtime._pending_turn import _PendingExecutionTurns

from .test_session_timeline import _execution, _session


def _reservation():
    execution = replace(
        _execution("execution", status=ExecutionStatus.PENDING_START, prompt="question"),
        revision=0, event_seq=0, agent_run_seq=0,
    )
    identity = IdempotencyRecord(
        "execution.start", "a" * 64, "b" * 64, ResourceKind.EXECUTION,
        execution.execution_id, IdempotencyStatus.RESERVED, None, None,
        execution.created_at, execution.updated_at,
    )
    return ExecutionStartReservation(execution, identity)


def test_local_session_observation_detects_absent_to_absent_admission() -> None:
    turns = _PendingExecutionTurns()
    absent, before = turns.observe_session("session", tenant_id="tenant")
    assert absent is None
    reservation = _reservation()
    turn, created = turns.admit("tenant", reservation, _session(), now=reservation.execution.created_at)
    assert created
    observed, admitted = turns.observe_session("session", tenant_id="tenant")
    assert observed == turn.execution and admitted > before
    replay, created = turns.admit("tenant", reservation, _session(), now=reservation.execution.created_at)
    assert replay is turn and not created
    assert turns.observe_session("session", tenant_id="tenant")[1] == admitted
    turns.release(turn)
    absent, after = turns.observe_session("session", tenant_id="tenant")
    assert absent is None and after > admitted


def test_local_session_observation_preserves_unknown_without_reconciliation() -> None:
    turns = _PendingExecutionTurns()
    reservation = _reservation()
    turn, _ = turns.admit("tenant", reservation, _session(), now=reservation.execution.created_at)
    failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    turns.fail(turn, failure)

    def forbidden_reconcile():
        raise AssertionError("observation cannot reconcile an unknown commit")

    turn.reconcile = forbidden_reconcile
    with pytest.raises(AIError) as error:
        turns.observe_session("session", tenant_id="tenant")
    assert error.value is failure
    assert turns.get("execution", tenant_id="tenant") is turn
