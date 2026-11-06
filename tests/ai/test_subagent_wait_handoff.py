#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Worker handoff stops child observers without cancelling logical children."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.core import ExecutionStatus, Principal, UsageMetrics
from linktools.ai.execution import ExecutionResult
from linktools.ai.runtime._handoff import ExecutionHandoffScope, execution_handoff_scope
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.spec import SubagentRef

pytestmark = pytest.mark.asyncio


def _dispatcher(wait: AsyncMock) -> SubagentDispatcher:
    dispatcher = SubagentDispatcher(None, None, SimpleNamespace(wait=wait))
    dispatcher.cancel_child = AsyncMock()
    return dispatcher


async def _wait(dispatcher: SubagentDispatcher) -> dict:
    return await dispatcher._wait_child(
        "child", parent_execution_id="parent",
        principal=Principal("principal", "tenant", "service"),
        ref=SubagentRef("agent", "child"),
    )


@pytest.mark.parametrize("stop_before_registration", [False, True])
async def test_handoff_releases_child_wait_without_logical_cancel(
    stop_before_registration: bool,
) -> None:
    scope = ExecutionHandoffScope()
    entered, begin_wait = asyncio.Event(), asyncio.Event()

    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    dispatcher = _dispatcher(AsyncMock(side_effect=wait))

    async def delegate():
        await begin_wait.wait()
        return await _wait(dispatcher)

    token = execution_handoff_scope.set(scope)
    try:
        task = asyncio.create_task(scope.call_tool(delegate))
        await asyncio.sleep(0)
        if stop_before_registration:
            scope.stop()
            begin_wait.set()
        else:
            begin_wait.set()
            await asyncio.wait_for(entered.wait(), 2)
            scope.stop()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    finally:
        execution_handoff_scope.reset(token)
    dispatcher.cancel_child.assert_not_called()
    assert not scope._child_waits
    assert not scope._tools
    assert dispatcher.pending_background_tasks == ()
    if stop_before_registration:
        dispatcher._execution.wait.assert_not_called()


@pytest.mark.parametrize("cancel_mode", ["caller", "explicit_during_stop", "error"])
async def test_child_wait_retains_normal_cleanup_contract(cancel_mode: str) -> None:
    scope = ExecutionHandoffScope()
    entered = asyncio.Event()

    async def wait(*args, **kwargs):
        entered.set()
        if cancel_mode == "error":
            raise RuntimeError("wait failed")
        await asyncio.Event().wait()

    dispatcher = _dispatcher(AsyncMock(side_effect=wait))
    token = execution_handoff_scope.set(scope)
    try:
        task = asyncio.create_task(scope.call_tool(lambda: _wait(dispatcher)))
        await asyncio.wait_for(entered.wait(), 2)
        if cancel_mode == "explicit_during_stop":
            scope.stop()
            scope.cancel_tools()
        elif cancel_mode == "caller":
            task.cancel()
        expected = RuntimeError if cancel_mode == "error" else asyncio.CancelledError
        with pytest.raises(expected):
            await asyncio.wait_for(task, 2)
        await asyncio.gather(*dispatcher.pending_background_tasks)
    finally:
        execution_handoff_scope.reset(token)
    dispatcher.cancel_child.assert_awaited_once()
    assert not scope._child_waits


async def test_completed_child_wait_is_unregistered_before_later_stop() -> None:
    result = ExecutionResult("child", ExecutionStatus.SUCCEEDED, "done", UsageMetrics())
    dispatcher = _dispatcher(AsyncMock(return_value=result))
    scope = ExecutionHandoffScope()
    token = execution_handoff_scope.set(scope)
    try:
        assert (await _wait(dispatcher))["output"] == "done"
        scope.stop()
        await asyncio.sleep(0)
    finally:
        execution_handoff_scope.reset(token)
    assert not scope._child_waits
    dispatcher.cancel_child.assert_not_called()


async def test_stopping_child_observer_does_not_cancel_entered_external_tool() -> None:
    scope = ExecutionHandoffScope()
    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def external_tool():
        entered.set()
        await release.wait()
        settled.set()
        return "done"

    task = asyncio.create_task(scope.call_tool(external_tool))
    await asyncio.wait_for(entered.wait(), 2)
    scope.stop()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not settled.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert settled.is_set()
    assert not scope._tools
