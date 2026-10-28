#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation retries retain acknowledged progress and independent business work."""

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import Task, TaskEvent, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine, TaskGraphRun


@pytest.mark.asyncio
async def test_stream_fault_allows_serial_cursor_resume_without_restarting_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_a, release_b = asyncio.Event(), asyncio.Event()
    acknowledged, disconnect, source_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first_result_observed = asyncio.Event()
    calls, delivered, resumed = [], [], []

    async def first(context):
        calls.append("a")
        await release_a.wait()
        return {"answer": "a"}

    async def second(context):
        calls.append("b")
        await release_b.wait()
        return {"answer": "b"}

    a, b = Task("retry.first", first, effect_policy="none"), Task("retry.second", second, effect_policy="none")
    original_watch = TaskGraphRun._watch_prepared

    async def failing_watch(self, cursor, include_content, ready, initial_ready=None):
        source = original_watch(self, cursor, include_content, ready, initial_ready)
        try:
            async for event in source:
                yield event
                if isinstance(event.event, TaskEvent):
                    break
            if ready is not None:
                ready.set()
            if initial_ready is not None:
                initial_ready.set()
            await disconnect.wait()
            raise ObservationError("stream", cursor=event.cursor, safe_details={"phase": "source_disconnect"})
        finally:
            await source.aclose()
            source_closed.set()

    async def observe(event):
        delivered.append(event)
        if isinstance(event.event, TaskEvent):
            acknowledged.set()

    async def retry_observer(event):
        assert source_closed.is_set()
        if isinstance(event.event, TaskEvent):
            resumed.append((event.event.graph_id, event.event.event_seq))
            if event.event.node_id == "a" and event.event.status is TaskStatus.SUCCEEDED:
                assert await run.result("a") == {"answer": "a"}
                first_result_observed.set()

    path = tmp_path / "state.sqlite"
    async with (
        Persistence.open("observation-retry", storage=PersistenceStorage.sqlite(path)) as persistence,
        TaskEngine.open(a, b, persistence=persistence) as engine,
    ):
        run = await engine.start(TaskGraph("independent", (
            TaskNode("a", task=a), TaskNode("b", task=b),
        )), idempotency_key="once")
        waiting = None
        try:
            with monkeypatch.context() as fault:
                fault.setattr(TaskGraphRun, "_watch_prepared", failing_watch)
                waiting = asyncio.create_task(run.wait(
                    on_event=observe, raise_on_stream_error=True, timeout_seconds=10,
                ))
                await asyncio.wait_for(acknowledged.wait(), 5)
                release_a.set()
                async with asyncio.timeout(5):
                    while True:
                        states = {node.node_id: node for node in (await run.state()).node_states}
                        if states["a"].status is TaskStatus.SUCCEEDED:
                            break
                        await asyncio.sleep(0.01)
                disconnect.set()
                with pytest.raises(ObservationError) as failure:
                    await asyncio.wait_for(waiting, 2)
                assert failure.value.origin == "stream"
                assert failure.value.safe_details["phase"] == "source_disconnect"
                cursor = failure.value.cursor
                assert cursor is not None and cursor == delivered[-1].cursor
                assert source_closed.is_set()

            assert not release_b.is_set()
            states = {node.node_id: node for node in (await run.state()).node_states}
            assert states["b"].status in {TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.READY}
            assert await run.result("a") == {"answer": "a"}
            reference = await run.result_ref("a")
            with pytest.raises(AIError) as invalid:
                await run.wait(on_event=retry_observer, cursor="not-a-cursor", raise_on_stream_error=True)
            assert invalid.value.code is ErrorCode.CURSOR_INVALID
            assert not resumed

            waiting = asyncio.create_task(run.wait(
                on_event=retry_observer, cursor=cursor, raise_on_stream_error=True, timeout_seconds=10,
            ))
            await asyncio.wait_for(first_result_observed.wait(), 2)
            assert not waiting.done() and not release_b.is_set()
            release_b.set()
            outcome = await waiting
            assert outcome.result.status is TaskStatus.SUCCEEDED
            assert outcome.observation_error is None
            delivered_facts = {(event.event.graph_id, event.event.event_seq) for event in delivered
                               if isinstance(event.event, TaskEvent)}
            assert delivered_facts and resumed and not delivered_facts.intersection(resumed)
            assert await run.result_ref("a") == reference
            assert sorted(calls) == ["a", "b"]
        finally:
            release_a.set()
            release_b.set()
            disconnect.set()
            if waiting is not None and not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    code = textwrap.dedent("""
        import asyncio
        import json
        import sys
        from pathlib import Path
        from linktools.ai.core import TaskStatus
        from linktools.ai.persistence import Persistence, PersistenceStorage
        from linktools.ai.task import Task, TaskEvent
        from linktools.ai.task.engine import TaskEngine

        async def first(context):
            raise AssertionError("Completed task a must not execute again")

        async def second(context):
            raise AssertionError("Completed task b must not execute again")

        async def main():
            a = Task("retry.first", first, effect_policy="none")
            b = Task("retry.second", second, effect_policy="none")
            replayed = []
            async def observe(event):
                if isinstance(event.event, TaskEvent):
                    replayed.append((event.event.graph_id, event.event.event_seq))
            async with (
                Persistence.open("observation-retry", storage=PersistenceStorage.sqlite(Path(sys.argv[1]))) as persistence,
                TaskEngine.open(a, b, persistence=persistence) as engine,
            ):
                run = await engine.get("independent")
                outcome = await run.wait(
                    on_event=observe, cursor=sys.argv[2], raise_on_stream_error=True, timeout_seconds=10,
                )
                assert outcome.result.status is TaskStatus.SUCCEEDED
                assert outcome.observation_error is None
                assert await run.result("a") == {"answer": "a"}
                reference = await run.result_ref("a")
                print(json.dumps({"events": replayed, "digest": reference.result_digest}))

        asyncio.run(main())
    """)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, str(path), cursor,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, errors = await asyncio.wait_for(process.communicate(), 20)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, errors.decode()
    replayed = json.loads(output)
    assert replayed["events"] and not delivered_facts.intersection(map(tuple, replayed["events"]))
    assert replayed["digest"] == reference.result_digest
    assert sorted(calls) == ["a", "b"]
