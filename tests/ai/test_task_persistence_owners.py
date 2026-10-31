#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone task producers retain their borrowed persistence until cleanup."""

from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "filesystem"))
async def test_direct_task_engine_prevents_premature_persistence_close(tmp_path: Path, backend: str) -> None:
    async def work(context):
        return "still-open"

    task = Task("work", work, effect_policy="none")
    storage = (PersistenceStorage.in_memory() if backend == "memory" else
               PersistenceStorage.from_root(tmp_path) if backend == "sqlite" else
               PersistenceStorage.filesystem(tmp_path))
    async with Persistence.open("task-owner", storage=storage) as persistence:
        engine = TaskEngine(task, persistence=persistence)
        try:
            with pytest.raises(AIError) as caught:
                await persistence.close()
            assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert persistence.ready
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="run")
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
            assert await run.result("node") == "still-open"
        finally:
            await engine.close()
        assert persistence.ready


@pytest.mark.asyncio
async def test_task_open_views_and_peers_release_only_their_own_lifetime(tmp_path: Path) -> None:
    async with Persistence.open("task-views", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        async with TaskEngine.open(persistence=persistence) as first:
            peer = TaskEngine(persistence=persistence)
            view = first.with_definitions()
            assert len(persistence._task_owners) == 2
            await view.close()
            with pytest.raises(AIError):
                await persistence.close()
            await peer.close()
            assert len(persistence._task_owners) == 1
        assert not persistence._task_owners


@pytest.mark.asyncio
async def test_task_close_notification_failure_retains_borrow_until_retry() -> None:
    async with Persistence.open("task-notifications", storage=PersistenceStorage.in_memory()) as persistence:
        engine = TaskEngine(persistence=persistence)
        calls = []
        failing = True

        def notify(current):
            calls.append(current)
            assert persistence.ready
            if failing:
                raise ValueError("notification incomplete")

        engine.add_close_callback(notify)
        try:
            with pytest.raises(ValueError, match="notification incomplete"):
                await engine.close()
            assert len(persistence._task_owners) == 1
            with pytest.raises(AIError):
                await persistence.close()
            assert persistence.ready
        finally:
            failing = False
            await engine.close()
        await engine.close()
        assert calls == [engine, engine]
        assert not persistence._task_owners


@pytest.mark.asyncio
async def test_task_cleanup_failure_and_cancelled_waiter_keep_borrow_until_quiescent(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    entered, finish = asyncio.Event(), asyncio.Event()
    async with Persistence.open("task-cleanup", storage=PersistenceStorage.in_memory()) as persistence:
        engine = TaskEngine(persistence=persistence)
        original = engine._launcher.shutdown
        attempts = 0

        async def shutdown():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("shutdown incomplete")
            entered.set()
            await finish.wait()
            await original()

        monkeypatch.setattr(engine._launcher, "shutdown", shutdown)
        with pytest.raises(ValueError, match="shutdown incomplete"):
            await engine.close()
        with pytest.raises(AIError):
            await persistence.close()
        closing = asyncio.create_task(engine.close())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(AIError):
                await persistence.close()
            assert len(persistence._task_owners) == 1
        finally:
            finish.set()
            await engine.close()
        assert attempts == 2 and not persistence._task_owners


@pytest.mark.asyncio
async def test_failed_task_construction_does_not_register_a_borrow(monkeypatch: pytest.MonkeyPatch) -> None:
    import linktools.ai.task.engine._engine as module

    def fail(*args, **kwargs):
        raise ValueError("binding failed")

    async with Persistence.open("task-construction", storage=PersistenceStorage.in_memory()) as persistence:
        monkeypatch.setattr(module, "ExecutionBinding", fail)
        with pytest.raises(ValueError, match="binding failed"):
            TaskEngine(persistence=persistence)
        assert not persistence._task_owners


@pytest.mark.asyncio
async def test_task_engine_read_only_observation_remains_supported(tmp_path: Path) -> None:
    async def work(context):
        return "retained"

    task = Task("work", work, effect_policy="none")
    storage = PersistenceStorage.from_root(tmp_path)
    async with Persistence.open("task-reader", storage=storage) as persistence:
        async with TaskEngine.open(task, persistence=persistence) as engine:
            run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="run")
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
    async with Persistence.open("task-reader", storage=storage, read_only=True) as persistence:
        async with TaskEngine.open(task, persistence=persistence) as engine:
            run = await engine.get("graph")
            assert await run.result("node") == "retained"
            with pytest.raises(AIError):
                await persistence.close()


@pytest.mark.asyncio
async def test_owned_runtime_waits_for_independent_task_borrower() -> None:
    from linktools.ai.runtime import Runtime

    runtime = Runtime.open("owned-task", storage=PersistenceStorage.in_memory())
    await runtime.__aenter__()
    bound = await runtime.tasks.bind()
    persistence = bound.persistence
    independent = TaskEngine(persistence=persistence)
    try:
        with pytest.raises(AIError) as blocked:
            await runtime.close()
        assert blocked.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert persistence.ready and len(persistence._task_owners) == 1
        assert len(persistence._runtime_owners) == 1
    finally:
        await independent.close()
        await runtime.close()
    assert not persistence.ready and not persistence._task_owners and not persistence._runtime_owners
