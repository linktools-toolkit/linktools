#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed producer lifetimes do not accumulate in a long-lived composition."""

import asyncio
import gc
import weakref

import pytest

from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.observe import Metrics
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine


class _Payload:
    def __init__(self) -> None:
        self.data = bytearray(1024 * 1024)


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("metrics", [False, True])
async def test_closed_producers_release_executed_definitions(evaluation: bool, metrics: bool) -> None:
    refs = []
    async with Runtime.open("release", storage=PersistenceStorage.in_memory(),
                            metrics=Metrics.in_memory() if metrics else None) as runtime:
        for index in range(3):
            payload = _Payload()

            async def handler(context, captured=payload):
                return len(captured.data)

            definition = Task(f"task-{index}", handler, effect_policy="none")
            engine = await runtime.tasks.bind(definition)
            run = await engine.start(TaskGraph(f"graph-{index}", (TaskNode("node", task=definition, input={}),)),
                                     idempotency_key=f"request-{index}")
            await run.wait(timeout_seconds=5)
            assert await run.result("node") == 1024 * 1024
            refs.extend((weakref.ref(engine), weakref.ref(payload)))
            if evaluation:
                evaluations = await runtime.evaluations.bind(engine)
                refs.append(weakref.ref(evaluations))
                await evaluations.close()
                del evaluations
            await engine.close()
            del run, engine, definition, handler, payload
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert runtime.metrics.status().recorders == ()
        assert (await runtime.metrics.flush()).completed


@pytest.mark.asyncio
async def test_active_producers_remain_owned_and_closed_views_do_not_release_owner() -> None:
    async with Runtime.open("active", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        view = engine.with_definitions()
        evaluations = await runtime.evaluations.bind(view)
        engine_ref, evaluation_ref = weakref.ref(engine), weakref.ref(evaluations)
        await view.close()
        del engine, evaluations, view
        gc.collect()
        assert engine_ref() is not None and evaluation_ref() is not None
        await evaluation_ref().close()
        await engine_ref().close()
        gc.collect()
        assert engine_ref() is None and evaluation_ref() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
async def test_failed_close_notification_can_be_retried_without_reclosing_resources(evaluation: bool) -> None:
    async with Runtime.open("notification", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        producer = await Evaluations.create(engine) if evaluation else TaskEngine(persistence=engine.persistence)
        calls = []
        fail = True

        def first(resource):
            calls.append("first")

        def second(resource):
            calls.append("second")
            if fail:
                raise ValueError("notification failed")

        producer.add_close_callback(first)
        producer.add_close_callback(second)
        with pytest.raises(ValueError, match="notification failed"):
            await producer.close()
        fail = False
        await producer.close()
        await producer.close()
        assert calls == ["first", "second", "second"]


@pytest.mark.asyncio
async def test_cancelled_close_releases_owner_only_after_cleanup_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Runtime.open("cancel-release", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        ref = weakref.ref(engine)
        started, finish = asyncio.Event(), asyncio.Event()
        original = engine._launcher.shutdown

        async def shutdown():
            started.set()
            await finish.wait()
            await original()

        monkeypatch.setattr(engine._launcher, "shutdown", shutdown)
        close = asyncio.create_task(engine.close())
        await started.wait()
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        del engine, close
        gc.collect()
        assert ref() is not None
        finish.set()
        await ref().close()
        gc.collect()
        assert ref() is None


@pytest.mark.asyncio
async def test_failed_later_close_releases_successful_producer_and_preserves_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("partial-release", storage=PersistenceStorage.in_memory())
    await runtime.__aenter__()
    first = await runtime.tasks.bind()
    second = await runtime.tasks.bind()
    second_ref = weakref.ref(second)
    persistence = first.persistence
    original = first.close
    fail = True

    async def close():
        if fail:
            raise ValueError("close failed")
        await original()

    monkeypatch.setattr(first, "close", close)
    del second
    with pytest.raises(ValueError, match="close failed"):
        await runtime.close()
    gc.collect()
    assert second_ref() is None
    assert persistence.ready
    fail = False
    await runtime.close()
    assert not persistence.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_evaluation_creation_closes_reader_but_not_borrowed_engine(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    from linktools.ai.history import History

    closes = []
    original = History.close

    async def close(history):
        closes.append(True)
        await original(history)

    class BrokenEvaluations(Evaluations):
        def __init__(self, *args, **kwargs):
            if cancelled:
                raise asyncio.CancelledError("creation cancelled")
            raise ValueError("creation failed")

    monkeypatch.setattr(History, "close", close)
    async with Runtime.open("creation", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        with pytest.raises(asyncio.CancelledError if cancelled else ValueError):
            await BrokenEvaluations.create(engine)
        assert closes == [True]
        view = engine.with_definitions()
        await view.close()
        assert engine.persistence.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
async def test_close_notification_can_register_a_later_observer(evaluation: bool) -> None:
    async with Runtime.open("late-observer", storage=PersistenceStorage.in_memory()) as runtime:
        engine = await runtime.tasks.bind()
        producer = await Evaluations.create(engine) if evaluation else engine
        calls = []

        def second(resource):
            calls.append("second")

        def first(resource):
            calls.append("first")
            resource.add_close_callback(second)

        producer.add_close_callback(first)
        await producer.close()
        await producer.close()
        assert calls == ["first", "second"]
