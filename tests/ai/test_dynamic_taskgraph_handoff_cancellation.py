#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation contracts for Task-to-Execution handoff ownership."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.runtime._planner import _AgentTaskNodeHandler


@pytest.mark.asyncio
async def test_cancelled_handoff_continuation_releases_start_hold_after_commit() -> None:
    released: list[tuple[str, str, str]] = []

    async def release_hold(
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
        released.append((execution_id, tenant_id, hold_id))

    handler = _AgentTaskNodeHandler(
        object(),
        object(),
        object(),
        release_dependency_hold=release_hold,
    )
    entered = asyncio.Event()
    finish = asyncio.Event()

    class Control:
        async def handoff_execution(self, execution_id: str) -> None:
            assert execution_id == "execution"
            entered.set()
            await finish.wait()

    caller = asyncio.create_task(
        handler._handoff_execution(
            Control(),
            "execution",
            key=("tenant", "graph", "node"),
        )
    )
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    finish.set()
    pending = handler.pending_background_tasks
    assert pending
    await asyncio.gather(*pending)

    assert released == [("execution", "tenant", "task:graph:node")]
    assert handler.background_failure is None


@pytest.mark.asyncio
async def test_cancelled_launch_continuation_releases_start_hold_after_handoff() -> None:
    released: list[tuple[str, str, str]] = []
    handed_off: list[str] = []

    async def release_hold(
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> None:
        released.append((execution_id, tenant_id, hold_id))

    handler = _AgentTaskNodeHandler(
        object(),
        object(),
        object(),
        release_dependency_hold=release_hold,
    )

    async def launched() -> object:
        return SimpleNamespace(execution_id="execution")

    class Control:
        async def handoff_execution(self, execution_id: str) -> None:
            handed_off.append(execution_id)

    await handler._handoff_after_launch(
        asyncio.create_task(launched()),
        ("tenant", "graph", "node"),
        Control(),
    )

    assert handed_off == ["execution"]
    assert released == [("execution", "tenant", "task:graph:node")]
    assert handler.background_failure is None
