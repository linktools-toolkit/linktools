#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold release reuses committed terminal status, not retention authority."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.core import ExecutionStatus, HandoffGate
from linktools.ai.execution import TaskExecutionService


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.STARTED))
@pytest.mark.parametrize("conversation", (False, True))
async def test_hold_release_reuses_only_terminal_status(status: ExecutionStatus, conversation: bool) -> None:
    service = object.__new__(TaskExecutionService)
    released = SimpleNamespace(status=status, conversation=conversation)
    service._repositories = SimpleNamespace(executions=SimpleNamespace(release_dependency_hold=AsyncMock(return_value=released)))
    service._handoff = HandoffGate()
    service.load_execution = AsyncMock(return_value=SimpleNamespace(status=ExecutionStatus.SUCCEEDED, conversation=conversation))
    service.request_terminal_handoff = AsyncMock()
    await service.release_dependency_hold("execution", tenant_id="tenant", hold_id="hold")
    service._repositories.executions.release_dependency_hold.assert_awaited_once_with(
        "execution", tenant_id="tenant", hold_id="hold")
    if status is ExecutionStatus.STARTED:
        service.load_execution.assert_awaited_once_with("execution", tenant_id="tenant")
    else:
        service.load_execution.assert_not_awaited()
    service.request_terminal_handoff.assert_awaited_once_with(
        "execution", tenant_id="tenant", terminal_commit=conversation)


@pytest.mark.asyncio
async def test_hold_release_failure_does_not_request_cleanup() -> None:
    service = object.__new__(TaskExecutionService)
    service._repositories = SimpleNamespace(executions=SimpleNamespace(release_dependency_hold=AsyncMock(side_effect=RuntimeError("conflict"))))
    service._handoff = HandoffGate()
    service.load_execution = AsyncMock()
    service.request_terminal_handoff = AsyncMock()
    with pytest.raises(RuntimeError, match="conflict"):
        await service.release_dependency_hold("execution", tenant_id="tenant", hold_id="hold")
    service.load_execution.assert_not_awaited()
    service.request_terminal_handoff.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (RuntimeError("conflict"), asyncio.CancelledError()))
async def test_failed_durable_release_does_not_strand_local_consumers(failure: BaseException) -> None:
    service = object.__new__(TaskExecutionService)
    service._repositories = SimpleNamespace(executions=SimpleNamespace(
        release_dependency_hold=AsyncMock(side_effect=failure)))
    service._handoff = HandoffGate()
    service.request_terminal_handoff = AsyncMock()
    key = ("tenant", "execution")
    await service._handoff.acquire_hold(key, "hold")
    await service._handoff.request_release(key)
    with pytest.raises(type(failure)):
        await service.release_dependency_hold("execution", tenant_id="tenant", hold_id="hold")
    # A failed persistence await must not leave cleanup owning the local gate.
    consumer = await asyncio.wait_for(service._handoff.enter(key), 1)
    await service._handoff.acquire_hold(key, "other-hold")
    assert not await service._handoff.leave(key, consumer)
    _, owner = await service._handoff.release_hold(key, "other-hold")
    assert owner
    service.request_terminal_handoff.assert_not_awaited()
