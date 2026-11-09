#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation tuning reduces idle SQL reads without delaying final truth."""

import asyncio
from contextvars import ContextVar

import pytest
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import event

from linktools.ai.core import TaskStatus
from linktools.ai.execution import TaskGraphProjection, TaskModelProjection
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

from .test_evaluation_consumers import FixtureModels
from .test_graph_metadata_projection import _runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", (0, -1, float("nan"), float("inf"), True, "1"))
async def test_model_refresh_interval_rejects_invalid_periods(interval) -> None:
    async with Persistence.open("interval", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(ValueError, match="finite and positive"):
            TaskEngine(persistence=persistence, model_refresh_interval_seconds=interval)


@pytest.mark.asyncio
async def test_longer_refresh_period_reduces_idle_reads_and_keeps_terminal_projection(tmp_path) -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    owner = ContextVar("observation_io_owner", default=None)
    counts = {"default": {"select": 0, "transaction": 0}, "tuned": {"select": 0, "transaction": 0}}
    measuring = False

    class Models(FixtureModels):
        def materialize(self):
            async def respond(messages, info):
                started.set()
                await release.wait()
                yield "complete"
            return FunctionModel(stream_function=respond)

    async with _runtime(tmp_path, backend="sqlite", models=Models()) as (runtime, base):
        task = Task.from_agent("interval.agent", runtime.agents.get())
        engine = base.with_definitions(task)
        async with (
            TaskEngine.open(task, persistence=base.persistence) as default,
            TaskEngine.open(task, persistence=base.persistence, model_refresh_interval_seconds=60) as tuned,
        ):
            assert tuned.with_definitions().model_refresh_interval_seconds == 60
            run = await engine.start(TaskGraph("interval", (
                TaskNode("node", task=task, input=AgentTaskInput("wait")),
            )), idempotency_key="interval")
            await asyncio.wait_for(started.wait(), 10)
            default_view = await default.get(run.graph_id)
            peer = await tuned.get(run.graph_id)
            sql = base.persistence.execution.executions.state_store.storage_group.context.engine.sync_engine

            def statement(_connection, _cursor, text, _parameters, _context, _many):
                label = owner.get()
                if measuring and label in counts and text.lstrip().upper().startswith("SELECT"):
                    counts[label]["select"] += 1

            def begin(_connection):
                label = owner.get()
                if measuring and label in counts:
                    counts[label]["transaction"] += 1

            ready = {name: asyncio.Event() for name in counts}
            results = {name: [] for name in counts}
            terminal_models = {name: asyncio.Event() for name in counts}

            async def observe(label, handle):
                token = owner.set(label)
                try:
                    async for item in handle.watch():
                        results[label].append(item.event)
                        if isinstance(item.event, TaskGraphProjection) and item.event.coverage is not None:
                            ready[label].set()
                        if isinstance(item.event, TaskModelProjection) and item.event.item.status == "SUCCEEDED":
                            terminal_models[label].set()
                finally:
                    owner.reset(token)

            event.listen(sql, "before_cursor_execute", statement)
            event.listen(sql, "begin", begin)
            watchers = [asyncio.create_task(observe("default", default_view)), asyncio.create_task(observe("tuned", peer))]
            try:
                await asyncio.wait_for(asyncio.gather(*(item.wait() for item in ready.values())), 10)
                measuring = True
                await asyncio.sleep(3.2)
                measuring = False
                print("idle observation SQL, 3.2 seconds:", counts)
                assert counts["tuned"]["select"] < counts["default"]["select"]
                assert counts["tuned"]["transaction"] < counts["default"]["transaction"]
                release.set()
                outcome = await run.wait(timeout_seconds=10)
                assert outcome.result.status is TaskStatus.SUCCEEDED
                # Terminal model metadata must not wait for the 60-second refresh.
                await asyncio.wait_for(asyncio.gather(*(item.wait() for item in terminal_models.values())), 5)
                # These engines do not own the graph: its event stream uses
                # independent polling with a maximum 30-second idle backoff.
                await asyncio.wait_for(asyncio.gather(*watchers), 35)
                for rows in results.values():
                    assert isinstance(rows[-1], TaskGraphProjection)
                    assert rows[-1].phase == "final"
                    assert rows[-1].graph.status is TaskStatus.SUCCEEDED
                    assert any(isinstance(row, TaskModelProjection) and row.item.status == "SUCCEEDED" for row in rows)
            finally:
                measuring = False
                release.set()
                for watcher in watchers:
                    watcher.cancel()
                await asyncio.gather(*watchers, return_exceptions=True)
                event.remove(sql, "before_cursor_execute", statement)
                event.remove(sql, "begin", begin)
