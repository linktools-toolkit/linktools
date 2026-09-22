#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    TaskStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Execution, Runtime, TaskGraphRun, TaskGraphRunEvent
from linktools.ai.runtime.service_api import (
    ExecutionEvent,
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
)
from linktools.ai.task import TaskEvent, TaskEventType, TaskGraphResult


class _ExecutionService:
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences=None,
        include_content: bool = False,
    ):
        del principal, include_content
        after = 0 if after_sequences is None else after_sequences.get(execution_id, 0)

        async def values():
            if after >= 1:
                return
            yield ExecutionTreeEvent(
                execution_id,
                "agent",
                ExecutionLineageKind.RUN,
                None,
                execution_id,
                None,
                0,
                ExecutionStreamEvent(
                    execution_id,
                    1,
                    ExecutionEventType.EXECUTION_SUCCEEDED.value,
                    {},
                ),
            )

        return values()


class _TaskGraphService:
    async def snapshot(self, graph_id: str, *, principal: Principal):
        del principal

        class State:
            node_id = "node"
            execution_id = "execution"

        class Snapshot:
            node_states = (State(),)

        assert graph_id == "graph"
        return Snapshot()

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ):
        del principal

        async def values():
            now = datetime.now(timezone.utc)
            events = (
                TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.GRAPH_ADMITTED,
                    now,
                    TaskStatus.PENDING,
                ),
                TaskEvent(
                    1,
                    graph_id,
                    2,
                    TaskEventType.NODE_CHANGED,
                    now,
                    TaskStatus.RUNNING,
                    TaskStatus.READY,
                    "node",
                    "worker",
                    1,
                ),
                TaskEvent(
                    1,
                    graph_id,
                    3,
                    TaskEventType.NODE_CHANGED,
                    now,
                    TaskStatus.WAITING,
                    TaskStatus.RUNNING,
                    "node",
                    None,
                    1,
                    "execution",
                ),
                TaskEvent(
                    1,
                    graph_id,
                    4,
                    TaskEventType.GRAPH_CHANGED,
                    now,
                    TaskStatus.SUCCEEDED,
                    TaskStatus.RUNNING,
                ),
            )
            for event in events:
                if event.sequence > after_sequence:
                    yield event

        return values()


class _HistoryService:
    async def list_events(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: str | None = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> Page[ExecutionEvent]:
        assert execution_id == "execution"
        assert principal.tenant_id == "tenant"
        assert cursor is None
        assert include_content is False
        assert limit == 1
        return Page(
            (
                ExecutionEvent(
                    execution_id,
                    1,
                    ExecutionEventType.EXECUTION_SUCCEEDED.value,
                    {},
                ),
            ),
            "next",
        )


class _Runtime:
    namespace = "watch-test"

    def __init__(self) -> None:
        self.execution = _ExecutionService()
        self.graph = _TaskGraphService()
        self.history = _HistoryService()


def _watch_tree(
    execution_id,
    *,
    principal,
    after_sequences=None,
    include_content=False,
):
    return _ExecutionService().stream(
        execution_id,
        principal=principal,
        after_sequences=after_sequences,
        include_content=include_content,
    )


@pytest.mark.asyncio
async def test_execution_list_events_uses_runtime_history_query_surface() -> None:
    execution = Execution(
        _Runtime(),
        "execution",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    page = await execution.list_events(limit=1)

    assert len(page.items) == 1
    assert page.items[0].event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value
    assert page.next_cursor == "next"


@pytest.mark.asyncio
async def test_execution_watch_projects_complete_execution_tree() -> None:
    runtime = _Runtime()
    execution = Execution(
        runtime,
        "execution",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    values = [item async for item in execution.watch()]
    assert len(values) == 1
    assert values[0].execution_id == "execution"
    assert values[0].event.event_type == ExecutionEventType.EXECUTION_SUCCEEDED.value
    assert values[0].cursor is not None
    assert [item async for item in execution.watch(cursor=values[0].cursor)] == []
    with pytest.raises(AIError) as raised:
        execution.watch(cursor=values[0].cursor, include_content=True)
    assert raised.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
async def test_task_graph_watch_starts_execution_before_binding_event_yield() -> None:
    started: list[str] = []

    def watch_tree(
        execution_id: str,
        *,
        principal: Principal,
        after_sequences=None,
        include_content: bool = False,
    ):
        del principal, after_sequences, include_content
        started.append(execution_id)

        async def values():
            await asyncio.Event().wait()
            if False:
                yield None

        return values()

    run = TaskGraphRun(
        _Runtime(),
        "graph",
        Principal("owner", "tenant"),
        watch_tree,
    )
    stream = run.watch()
    try:
        await anext(stream)
        assert started == []
        await anext(stream)
        assert started == []
        binding = await anext(stream)
        assert isinstance(binding.event, TaskEvent)
        assert binding.event.execution_id == "execution"
        assert started == ["execution"]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_task_graph_run_watch_merges_task_and_execution_events() -> None:
    run = TaskGraphRun(
        _Runtime(),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    values = [item async for item in run.watch()]
    assert all(isinstance(item, TaskGraphRunEvent) for item in values)
    assert [type(item.event) for item in values].count(TaskEvent) == 4
    execution = [
        item for item in values if isinstance(item.event, ExecutionTreeEvent)
    ]
    assert len(execution) == 1
    assert execution[0].node_id == "node"
    assert execution[0].event.execution_id == "execution"
    assert (
        execution[0].event.event.event_type
        == ExecutionEventType.EXECUTION_SUCCEEDED.value
    )
    assert all(item.cursor is not None for item in values)
    assert all(item.event.cursor is not None for item in execution)
    assert values[-1].cursor is not None
    assert [item async for item in run.watch(cursor=values[-1].cursor)] == []
    with pytest.raises(AIError) as raised:
        run.watch(cursor=values[-1].cursor, include_content=True)
    assert raised.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
async def test_task_graph_watch_rejects_unbound_cursor_before_starting_stream() -> None:
    class GraphService:
        def __init__(self) -> None:
            self.stream_calls = 0

        async def snapshot(self, graph_id: str, *, principal: Principal):
            del principal
            assert graph_id == "graph"
            state = type(
                "State",
                (),
                {"node_id": "node", "execution_id": None},
            )()
            return type("Snapshot", (), {"node_states": (state,)})()

        def stream_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
        ):
            del graph_id, principal, after_sequence
            self.stream_calls += 1

            async def values():
                await asyncio.Event().wait()
                if False:
                    yield None

            return values()

    service = GraphService()
    runtime = type(
        "Runtime",
        (),
        {"namespace": "watch-test", "graph": service},
    )()
    run = TaskGraphRun(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    stream = run.watch(
        after_execution_sequences={"node": {"execution": 1}},
    )
    try:
        with pytest.raises(AIError) as raised:
            await anext(stream)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert service.stream_calls == 0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_task_graph_replay_delivers_pages_without_buffering_all_events() -> None:
    observed_first = asyncio.Event()
    now = datetime.now(timezone.utc)

    class GraphService:
        async def snapshot(self, graph_id: str, *, principal: Principal):
            del principal
            assert graph_id == "graph"
            return type(
                "Snapshot",
                (),
                {
                    "graph_id": graph_id,
                    "status": TaskStatus.RUNNING,
                    "event_sequence": 2,
                    "node_states": (),
                },
            )()

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            assert graph_id == "graph"
            if after_sequence == 0:
                return Page(
                    (
                        TaskEvent(
                            1,
                            graph_id,
                            1,
                            TaskEventType.GRAPH_ADMITTED,
                            now,
                            TaskStatus.PENDING,
                        ),
                    )
                )
            await observed_first.wait()
            return Page(
                (
                    TaskEvent(
                        1,
                        graph_id,
                        2,
                        TaskEventType.GRAPH_CHANGED,
                        now,
                        TaskStatus.RUNNING,
                        TaskStatus.PENDING,
                    ),
                )
            )

    runtime = type(
        "Runtime",
        (),
        {
            "namespace": "watch-test",
            "graph": GraphService(),
            "execution": object(),
            "event": object(),
        },
    )()
    run = TaskGraphRun(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[int] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        assert isinstance(event.event, TaskEvent)
        observed.append(event.event.sequence)
        if event.event.sequence == 1:
            observed_first.set()

    await asyncio.wait_for(run.replay(observer), timeout=1)
    assert observed == [1, 2]


@pytest.mark.asyncio
async def test_task_graph_replay_uses_captured_durable_cutoffs() -> None:
    now = datetime.now(timezone.utc)

    class ReplayGraphService:
        async def snapshot(self, graph_id: str, *, principal: Principal):
            del principal
            assert graph_id == "graph"
            return type(
                "Snapshot",
                (),
                {
                    "graph_id": "graph",
                    "status": TaskStatus.RUNNING,
                    "event_sequence": 2,
                    "node_states": (
                        type(
                            "State",
                            (),
                            {
                                "node_id": "node",
                                "status": TaskStatus.WAITING,
                                "result_digest": None,
                                "execution_id": "execution",
                                "error_code": None,
                                "error_digest": None,
                            },
                        )(),
                    ),
                },
            )()

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            assert graph_id == "graph"
            values = (
                TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.GRAPH_ADMITTED,
                    now,
                    TaskStatus.PENDING,
                ),
                TaskEvent(
                    1,
                    graph_id,
                    2,
                    TaskEventType.NODE_CHANGED,
                    now,
                    TaskStatus.WAITING,
                    TaskStatus.RUNNING,
                    "node",
                    None,
                    1,
                    "execution",
                ),
                TaskEvent(
                    1,
                    graph_id,
                    3,
                    TaskEventType.GRAPH_CHANGED,
                    now,
                    TaskStatus.SUCCEEDED,
                    TaskStatus.RUNNING,
                ),
            )
            return Page(tuple(value for value in values if value.sequence > after_sequence))

    class ReplayExecutionService:
        async def inspect(self, execution_id: str, *, principal: Principal):
            del principal
            assert execution_id == "execution"
            return ExecutionView(
                "execution",
                "agent",
                ExecutionStatus.STARTED,
                ExecutionLineageKind.RUN,
                None,
                "execution",
                None,
                event_sequence=2,
            )

        async def list_children(
            self,
            execution_id: str,
            *,
            principal: Principal,
        ):
            del principal
            assert execution_id == "execution"
            return ()

    class ReplayEventService:
        async def list(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            assert execution_id == "execution"
            values = (
                ExecutionEvent(execution_id, 1, "EXECUTION_STARTED", {"raw": "one"}),
                ExecutionEvent(execution_id, 2, "EXECUTION_SUCCEEDED", {"raw": "two"}),
                ExecutionEvent(execution_id, 3, "LATE_EVENT", {"raw": "late"}),
            )
            return Page(tuple(value for value in values if value.sequence > after_sequence))

    runtime = type(
        "ReplayRuntime",
        (),
        {
            "namespace": "watch-test",
            "graph": ReplayGraphService(),
            "execution": ReplayExecutionService(),
            "event": ReplayEventService(),
        },
    )()
    run = TaskGraphRun(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[TaskGraphRunEvent] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        observed.append(event)

    result = await run.replay(observer)

    assert result.status is TaskStatus.WAITING
    assert [
        event.event.sequence
        for event in observed
        if isinstance(event.event, TaskEvent)
    ] == [1, 2]
    execution_events = [
        event.event.event
        for event in observed
        if isinstance(event.event, ExecutionTreeEvent)
    ]
    assert [event.durable_sequence for event in execution_events] == [1, 2]
    assert all(event.payload == {} for event in execution_events)
    assert all(event.cursor is not None for event in observed)
    assert all(
        item.event.cursor is not None
        for item in observed
        if isinstance(item.event, ExecutionTreeEvent)
    )



@pytest.mark.asyncio
async def test_task_graph_replay_keeps_direct_execution_tree_boundary() -> None:
    now = datetime.now(timezone.utc)

    class GraphService:
        async def snapshot(self, graph_id: str, *, principal: Principal):
            del principal
            assert graph_id == "graph"
            return type(
                "Snapshot",
                (),
                {
                    "graph_id": graph_id,
                    "status": TaskStatus.RUNNING,
                    "event_sequence": 1,
                    "node_states": (
                        type(
                            "State",
                            (),
                            {
                                "node_id": "node",
                                "status": TaskStatus.RUNNING,
                                "result_digest": None,
                                "execution_id": "root",
                                "error_code": None,
                                "error_digest": None,
                            },
                        )(),
                    ),
                },
            )()

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            values = (
                TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.GRAPH_ADMITTED,
                    now,
                    TaskStatus.PENDING,
                ),
            )
            return Page(tuple(value for value in values if value.sequence > after_sequence))

    views = {
        "root": ExecutionView(
            "root",
            "agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.RUN,
            None,
            "root",
            None,
            event_sequence=1,
        ),
        "child": ExecutionView(
            "child",
            "child-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.SUBAGENT,
            "root",
            "root",
            "call-child",
            event_sequence=1,
        ),
        "grandchild": ExecutionView(
            "grandchild",
            "grandchild-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.SUBAGENT,
            "child",
            "root",
            "call-grandchild",
            event_sequence=1,
        ),
    }

    class ExecutionService:
        async def inspect(self, execution_id: str, *, principal: Principal):
            del principal
            return views[execution_id]

        async def list_children(
            self,
            execution_id: str,
            *,
            principal: Principal,
        ):
            del principal
            assert execution_id == "root"
            return (views["child"],)

    class EventService:
        async def list(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            values = (
                ExecutionEvent(execution_id, 1, "EXECUTION_STARTED", {"raw": "one"}),
                ExecutionEvent(execution_id, 2, "LATE_EVENT", {"raw": "late"}),
            )
            return Page(tuple(value for value in values if value.sequence > after_sequence))

    runtime = type(
        "ReplayRuntime",
        (),
        {
            "namespace": "watch-test",
            "graph": GraphService(),
            "execution": ExecutionService(),
            "event": EventService(),
        },
    )()
    run = TaskGraphRun(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[TaskGraphRunEvent] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        observed.append(event)

    await run.replay(observer)

    execution_events = [
        event.event
        for event in observed
        if isinstance(event.event, ExecutionTreeEvent)
    ]
    assert [
        (event.execution_id, event.depth, event.event.durable_sequence)
        for event in execution_events
    ] == [
        ("root", 0, 1),
        ("child", 1, 1),
    ]
    assert all(event.event.payload == {} for event in execution_events)


class _WaitGraphService:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.wait_started = asyncio.Event()
        self.wait_cancelled = asyncio.Event()
        self.stream_release = asyncio.Event()

    async def snapshot(self, graph_id: str, *, principal: Principal):
        del principal
        assert graph_id == "graph"
        return type("Snapshot", (), {"node_states": ()})()

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ):
        del principal, after_sequence

        async def values():
            if self.mode == "observer_error":
                yield TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.GRAPH_ADMITTED,
                    datetime.now(timezone.utc),
                    TaskStatus.PENDING,
                )
            if self.mode == "recovery":
                yield TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.GRAPH_CHANGED,
                    datetime.now(timezone.utc),
                    TaskStatus.RECOVERY_REQUIRED,
                    TaskStatus.RUNNING,
                )
                return
            await self.stream_release.wait()
            if False:
                yield None

        return values()

    async def wait(
        self,
        graph_id: str,
        *,
        principal: Principal,
        timeout_seconds: float | None = None,
    ) -> TaskGraphResult:
        del principal, timeout_seconds
        assert graph_id == "graph"
        self.wait_started.set()
        if self.mode == "waiting":
            return TaskGraphResult(graph_id, TaskStatus.WAITING, ())
        if self.mode == "recovery":
            return TaskGraphResult(graph_id, TaskStatus.RECOVERY_REQUIRED, ())
        if self.mode == "timeout":
            raise AIError(
                ErrorCode.TASK_WAIT_TIMEOUT,
                safe_details={"graph_id": graph_id},
            )
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.wait_cancelled.set()
            raise
        raise AssertionError("unreachable")


def _wait_runtime(service: _WaitGraphService):
    return type(
        "WaitRuntime",
        (),
        {
            "namespace": "watch-test",
            "graph": service,
            "execution": _ExecutionService(),
        },
    )()


async def _assert_no_graph_observer_tasks() -> None:
    await asyncio.sleep(0)
    names = {
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    }
    assert "task-graph-wait-graph" not in names
    assert "task-graph-observer-graph" not in names
    assert "task-run-graph-graph" not in names


@pytest.mark.asyncio
async def test_task_graph_wait_cleans_observer_on_stable_waiting() -> None:
    service = _WaitGraphService("waiting")
    run = TaskGraphRun(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    async def observer(_event: TaskGraphRunEvent) -> None:
        raise AssertionError("idle observer should be cancelled before an event")

    result = await run.wait(observer=observer)

    assert result.status is TaskStatus.WAITING
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_wait_delivers_recovery_required_boundary() -> None:
    service = _WaitGraphService("recovery")
    run = TaskGraphRun(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[TaskGraphRunEvent] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        observed.append(event)

    result = await run.wait(observer=observer)

    assert result.status is TaskStatus.RECOVERY_REQUIRED
    assert len(observed) == 1
    assert isinstance(observed[0].event, TaskEvent)
    assert observed[0].event.status is TaskStatus.RECOVERY_REQUIRED
    assert observed[0].cursor is not None
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_wait_cleans_observer_on_timeout() -> None:
    service = _WaitGraphService("timeout")
    run = TaskGraphRun(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    async def observer(_event: TaskGraphRunEvent) -> None:
        return None

    with pytest.raises(AIError) as raised:
        await run.wait(observer=observer)

    assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_observer_error_cleans_waiter_without_cancelling_graph() -> None:
    service = _WaitGraphService("observer_error")
    run = TaskGraphRun(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    async def observer(_event: TaskGraphRunEvent) -> None:
        raise RuntimeError("observer failed")

    with pytest.raises(AIError) as raised:
        await run.wait(observer=observer)
    assert raised.value.code is ErrorCode.TASK_OBSERVER_FAILED

    await asyncio.wait_for(service.wait_cancelled.wait(), 1)
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_wait_outer_cancel_cleans_waiter_and_observer() -> None:
    service = _WaitGraphService("block")
    run = TaskGraphRun(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    async def observer(_event: TaskGraphRunEvent) -> None:
        return None

    task = asyncio.create_task(
        run.wait(observer=observer),
        name="test-task-graph-wait-cancel",
    )
    await service.wait_started.wait()
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(service.wait_cancelled.wait(), 1)
    await _assert_no_graph_observer_tasks()


def test_runtime_does_not_expose_stream_tree() -> None:
    assert not hasattr(Runtime, "stream_tree")


def test_task_run_event_rejects_execution_without_node() -> None:
    event = ExecutionTreeEvent(
        "execution",
        "agent",
        ExecutionLineageKind.RUN,
        None,
        "execution",
        None,
        0,
        ExecutionStreamEvent(
            "execution",
            1,
            ExecutionEventType.EXECUTION_SUCCEEDED.value,
            {},
        ),
    )
    with pytest.raises(ValueError):
        TaskGraphRunEvent("graph", None, event)
