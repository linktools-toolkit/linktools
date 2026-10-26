#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Deferred input replay across execution completion and graph publication."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task_engine import TaskEngine
from linktools.ai.task import TaskGraph, TaskNode


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("same_value", (True, False))
async def test_deferred_input_replays_execution_committed_before_graph_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, same_value: bool,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    async with Persistence.open('input-race', storage=storage) as persistence_25, TaskEngine.open(persistence=persistence_25) as engine:
        graph = await engine.start(
            TaskGraph("input", (TaskNode.wait("value"),)),
            principal=Principal("owner", "default"), idempotency_key="start",
        )
        waiting = (await graph.wait(timeout_seconds=10)).result.node_states[0]
        assert waiting.status is TaskStatus.WAITING
        assert waiting.execution_id is not None
        service = engine._execution_service
        complete = service._complete_task_output
        entered, release = asyncio.Event(), asyncio.Event()
        pause = True

        async def interleave(current, **kwargs):
            nonlocal pause
            if pause:
                pause = False
                assert current.status is ExecutionStatus.WAITING_DEFERRED
                entered.set()
                await asyncio.wait_for(release.wait(), 10)
            return await complete(current, **kwargs)

        monkeypatch.setattr(service, "_complete_task_output", interleave)
        accepted = {"answer": 1}
        supplied = accepted if same_value else {"answer": 2}
        stale = asyncio.create_task(service.supply_task_input(
            waiting.execution_id, principal=Principal("owner", "default"), value=supplied,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            winner = await service.supply_task_input(
                waiting.execution_id, principal=Principal("owner", "default"), value=accepted,
            )
            assert winner.status is ExecutionStatus.SUCCEEDED
            assert (await graph.state()).node_states[0].status is TaskStatus.WAITING
            release.set()
            if same_value:
                replay = await stale
                assert replay.execution_id == waiting.execution_id
                assert replay.status is ExecutionStatus.SUCCEEDED
            else:
                with pytest.raises(AIError) as raised:
                    await stale
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            result = await service.result(waiting.execution_id, principal=Principal("owner", "default"))
            assert result.output == accepted
        finally:
            release.set()
            await asyncio.gather(stale, return_exceptions=True)
