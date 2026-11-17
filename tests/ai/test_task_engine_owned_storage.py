#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Task storage ownership and Evaluation borrowing lifetimes."""

import pytest
from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine


async def answer(context):
    return {"answer": 42}


@pytest.mark.asyncio
async def test_standalone_callable_task_owns_storage(monkeypatch) -> None:
    from linktools.ai.runtime import Runtime

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("standalone callable Task opened a hidden Runtime")

    monkeypatch.setattr(Runtime, "open", unexpected_runtime)
    task = Task("answer", answer, effect_policy="none")
    async with TaskEngine.open(task, storage=PersistenceStorage.in_memory(), namespace="standalone") as engine:
        state = engine.persistence
        run = await engine.start(TaskGraph("graph", (TaskNode("answer", task=task),)), idempotency_key="once")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
    assert not state.ready
    await engine.close()


@pytest.mark.asyncio
async def test_borrowed_storage_survives_task_close() -> None:
    async with Persistence.open("borrowed", storage=PersistenceStorage.in_memory()) as state:
        async with TaskEngine.open(persistence=state):
            pass
        assert state.ready


@pytest.mark.asyncio
async def test_evaluation_borrows_root_and_exact_definition_view() -> None:
    async with TaskEngine.open(storage=PersistenceStorage.in_memory(), namespace="borrow-count") as engine:
        view = engine.with_definitions()
        other = engine.with_definitions()
        first = view.bind_evaluation()
        second = view.bind_evaluation()
        for producer in (engine, view):
            with pytest.raises(AIError) as error:
                await producer.close()
            assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert not producer._closing and not producer._closed
        await other.close()
        first()
        first()
        with pytest.raises(AIError):
            await engine.close()
        second()
        await view.close()
        assert engine.persistence.ready


@pytest.mark.asyncio
async def test_owned_storage_retained_after_context_cleanup_failure(monkeypatch) -> None:
    context = TaskEngine.open(storage=PersistenceStorage.in_memory(), namespace="retry-close")
    engine = await context.__aenter__()
    state = engine.persistence
    cleanup = engine._graph_service.preflight_close
    attempts = 0

    async def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic cleanup failure")
        await cleanup()

    monkeypatch.setattr(engine._graph_service, "preflight_close", fail_once)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        await context.__aexit__(None, None, None)
    assert state.ready
    await engine.close()
    assert not state.ready


@pytest.mark.asyncio
async def test_owned_storage_close_failure_is_retryable(monkeypatch) -> None:
    context = TaskEngine.open(storage=PersistenceStorage.in_memory(), namespace="retry-storage")
    engine = await context.__aenter__()
    state = engine.persistence
    close = state.close
    attempts = 0

    async def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic storage failure")
        await close()

    monkeypatch.setattr(state, "close", fail_once)
    with pytest.raises(RuntimeError, match="synthetic storage failure"):
        await context.__aexit__(None, None, None)
    assert state.ready
    await engine.close()
    assert not state.ready


@pytest.mark.asyncio
async def test_standalone_task_constructor_failure_closes_owned_storage(monkeypatch) -> None:
    closed = []
    close = Persistence.close

    async def capture_close(state):
        closed.append(state)
        await close(state)

    monkeypatch.setattr(Persistence, "close", capture_close)
    with pytest.raises(ValueError):
        async with TaskEngine.open(storage=PersistenceStorage.in_memory(), namespace="bad-config",
                                   model_refresh_interval_seconds=0):
            pytest.fail("invalid configuration opened")
    assert len(closed) == 1 and not closed[0].ready


@pytest.mark.asyncio
async def test_constructor_failure_remains_primary_when_cleanup_fails(monkeypatch) -> None:
    states = []
    close = Persistence.close

    async def failed_close(state):
        states.append(state)
        raise RuntimeError("synthetic constructor cleanup failure")

    monkeypatch.setattr(Persistence, "close", failed_close)
    try:
        with pytest.raises(ValueError, match="model_refresh_interval_seconds") as error:
            async with TaskEngine.open(storage=PersistenceStorage.in_memory(), namespace="double-failure",
                                       model_refresh_interval_seconds=0):
                pytest.fail("invalid configuration opened")
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "synthetic constructor cleanup failure"
        assert len(states) == 1 and states[0].ready
    finally:
        for state in states:
            await close(state)
