#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery producer admission and control-only reconciliation boundaries."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import AsyncMock, Mock

import pytest

from linktools.ai.core import ApprovalStatus, ExecutionStatus, Page
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import (
    ExecutionRecoveryDetail, PendingDeferredCall,
    PendingToolContinuation,
)
from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
from linktools.ai.storage import StoredPayload

from .test_execution_recovery_commands import _execution


_T = TypeVar("_T")


class _TurnAdmission:
    def __init__(self, error: AIError | None = None) -> None:
        self.owner_task: asyncio.Task[object] | None = None
        self.caller_task: asyncio.Task[object] | None = None
        self.error = error

    @property
    def active(self) -> bool:
        return self.owner_task is asyncio.current_task()

    async def run(self, execution_id: str, action: Callable[[], Awaitable[_T]]) -> _T:
        assert execution_id == "execution"
        if self.active:
            return await action()
        if self.error is not None:
            raise self.error
        self.caller_task = asyncio.current_task()

        async def own() -> _T:
            self.owner_task = asyncio.current_task()
            assert self.owner_task is not self.caller_task
            try:
                return await action()
            finally:
                self.owner_task = None

        return await asyncio.create_task(own())


def _port(admission: _TurnAdmission) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    execution = replace(_execution(now), recovery=ExecutionRecoveryDetail(agent_run_id="run"))
    return SimpleNamespace(
        tenant_id="tenant", execution=execution,
        run_recovery_turn=admission.run,
        load_execution=AsyncMock(return_value=execution),
        load_recovery_execution=AsyncMock(return_value=execution),
        _reconcile_tool_effects=AsyncMock(return_value=((), 0)),
        _pending_cancel_operations=AsyncMock(return_value=()),
        validate_binding=AsyncMock(return_value=None),
        validate_recovery_inputs=AsyncMock(return_value=None),
        _reconcile_session_recovery=AsyncMock(return_value=True),
        _recovery_idempotency=AsyncMock(return_value=None),
        _ensure_recovery_idempotency=AsyncMock(),
        restore_user_input=AsyncMock(return_value="prompt"),
        _prepare_recovery_relaunch=Mock(return_value=True),
        _reset_local_producer=Mock(),
        _publish_recovery_resumed=Mock(),
        _complete_recovered_cancel=AsyncMock(return_value=execution),
    )


@pytest.mark.asyncio
async def test_checkpoint_recovery_holds_admission_through_producer_launch() -> None:
    admission = _TurnAdmission()
    port = _port(admission)
    events = []

    async def commit(*args, **kwargs):
        assert admission.active
        events.append("producer")
        return port.execution

    async def launch(*args, **kwargs):
        assert admission.active
        events.append("launch")

    port._commit_recovery_resume = commit
    port.launch = launch
    assert await _RecoveryCoordinator(port, None).reconcile_execution(port.execution)
    assert events == ["producer", "launch"]
    assert admission.owner_task is None


@pytest.mark.asyncio
async def test_explicit_recovery_releases_admission_when_producer_commit_fails() -> None:
    admission = _TurnAdmission()
    port = _port(admission)
    failure = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def commit(*args, **kwargs):
        assert admission.active
        raise failure

    port._commit_recovery_resume = commit
    with pytest.raises(AIError) as caught:
        await _RecoveryCoordinator(port, None).recover_execution(
            "execution", tenant_id="tenant", expected_revision=0,
        )
    assert caught.value is failure
    assert admission.owner_task is None


@pytest.mark.asyncio
async def test_deferred_resume_claim_requires_turn_admission() -> None:
    admission = _TurnAdmission()
    port = _port(admission)
    payload = StoredPayload.inline_json({})
    port.execution = replace(
        port.execution, status=ExecutionStatus.WAITING_DEFERRED,
        recovery=replace(port.execution.recovery, pending_tools=PendingToolContinuation(
            "run", approvals=(PendingDeferredCall("call", "tool", payload, operation_id="test-operation"),),
        )),
    )
    port.load_execution.return_value = port.execution
    port.load_recovery_execution.return_value = port.execution
    port.settle_denied_tool_result = AsyncMock()
    port.load_approval = AsyncMock(return_value=SimpleNamespace(
        status=ApprovalStatus.DENIED, decision_message="denied", resolution_metadata={},
    ))
    port.read_deferred_payload = AsyncMock(return_value={})
    from linktools.ai.persistence.runs import AgentRunCheckpoint
    port.load_interrupted_checkpoint = AsyncMock(return_value=AgentRunCheckpoint("run", 1, [], output_retries_used=0))

    async def claim(*args, **kwargs):
        assert admission.active
        return replace(port.execution, status=ExecutionStatus.STARTED, revision=1,
                       agent_run_seq=2, recovery=replace(port.execution.recovery,
                       agent_run_id="resumed-run", source_agent_run_id="run"))

    port.claim_deferred_resume = claim
    result = await _RecoveryCoordinator(port, None).reconcile_deferred(
        "execution", tenant_id="tenant",
    )
    assert result is not None
    assert result.producer_generation == 1
    assert result.execution.recovery.agent_run_id == "resumed-run"
    assert result.execution.recovery.source_agent_run_id == "run"
    assert result.execution.recovery.pending_tools.source_agent_run_id == "run"
    port.settle_denied_tool_result.assert_awaited()
    assert result.results.approvals["call"].message == "denied"
    assert admission.owner_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("terminal", "cancel"))
async def test_control_only_reconciliation_does_not_require_turn_admission(control: str) -> None:
    admission = _TurnAdmission(AIError(ErrorCode.STORAGE_CONFLICT))
    port = _port(admission)
    if control == "terminal":
        port.execution = replace(port.execution, status=ExecutionStatus.SUCCEEDED)
    else:
        port._pending_cancel_operations.return_value = (object(),)
    assert not await _RecoveryCoordinator(port, None).reconcile_execution(port.execution)
    assert admission.owner_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "phase", "skip"), (
    (ErrorCode.STORAGE_CONFLICT, "turn_recovery_busy", True),
    (ErrorCode.STORAGE_CONFLICT, "subagent_parent_unavailable", True),
    (ErrorCode.STORAGE_CONFLICT, "unrelated", False),
    (ErrorCode.STORAGE_INTEGRITY_ERROR, "turn_recovery_busy", False),
))
async def test_startup_defers_only_known_recovery_admission_conflicts(
    code: ErrorCode, phase: str, skip: bool,
) -> None:
    failure = AIError(code, safe_details={"phase": phase})
    port = _port(_TurnAdmission(failure))
    port._list_recoverable_executions = AsyncMock(return_value=Page((port.execution,)))
    coordinator = _RecoveryCoordinator(port, None)
    if skip:
        await coordinator.reconcile()
    else:
        with pytest.raises(AIError) as caught:
            await coordinator.reconcile()
        assert caught.value is failure


@pytest.mark.asyncio
async def test_busy_turn_recovery_preserves_existing_producer() -> None:
    failure = AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"phase": "turn_recovery_busy"})
    port = _port(_TurnAdmission(failure))
    port._commit_recovery_resume = AsyncMock()
    with pytest.raises(AIError) as caught:
        await _RecoveryCoordinator(port, None).recover_execution(
            "execution", tenant_id="tenant", expected_revision=0,
        )
    assert caught.value is failure
    port._reset_local_producer.assert_not_called()
    port._commit_recovery_resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_resume_and_nested_relaunch_share_owner_task() -> None:
    admission = _TurnAdmission()
    port = _port(admission)
    producers = []

    async def commit(*args, **kwargs):
        assert admission.active
        producers.append(asyncio.current_task())
        port.execution = replace(port.execution, revision=1)
        port.load_execution.return_value = port.execution
        return port.execution

    async def admitted(execution, **kwargs):
        assert admission.active
        return execution

    async def launch(*args, **kwargs):
        assert admission.active
        assert asyncio.current_task() is producers[0]

    port._commit_recovery_resume = commit
    port._load_admitted_recovery_execution = admitted
    port.launch = launch
    latest, launched = await _RecoveryCoordinator(port, None).recover_execution(
        "execution", tenant_id="tenant", expected_revision=0,
    )
    assert launched
    assert latest.revision == 1
    assert admission.owner_task is None


@pytest.mark.asyncio
async def test_deferred_recovery_does_not_replay_an_already_advanced_attempt() -> None:
    port = _port(_TurnAdmission())
    pending = PendingToolContinuation("run", approvals=(
        PendingDeferredCall("call", "tool", StoredPayload.inline_json({}), operation_id="test-operation"),
    ))
    waiting = replace(port.execution, status=ExecutionStatus.WAITING_DEFERRED,
                      recovery=replace(port.execution.recovery, pending_tools=pending))
    current = replace(waiting, status=ExecutionStatus.STARTED, revision=1, agent_run_seq=2,
                      recovery=replace(waiting.recovery, agent_run_id="next-run", source_agent_run_id="run"))
    port.load_recovery_execution.return_value = current
    assert await _RecoveryCoordinator(port, None).reconcile_waiting_deferred(waiting) is None
    assert current.recovery.source_agent_run_id == pending.source_agent_run_id
