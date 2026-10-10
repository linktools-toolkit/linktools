#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Page,
    Principal,
    TaskStatus,
)
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.runtime import Execution, Runtime, TaskGraphRun, TaskGraphRunEvent
from linktools.ai.runtime import _cursor
from linktools.ai.runtime._domains import RuntimeExecutions, RuntimeSessions
from linktools.ai.runtime._execution_tree import ExecutionTreeBroker, ExecutionTreeStreamer
from linktools.ai.runtime._watch_cursor import decode_graph_watch_cursor, encode_graph_watch_cursor
from linktools.ai.runtime.service_api import (
    ExecutionEvent,
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
    ModelInteractionReadBoundary,
    TaskGraphProjection,
    TaskModelProjection,
)
from linktools.ai.runtime.service_api import _ExecutionStreamFailure
from linktools.ai.task import TaskEvent, TaskEventType, TaskGraphState, TaskNode, TaskNodeView


class _ExecutionService:
    async def inspect(self, execution_id: str, *, principal: Principal):
        del principal
        return ExecutionView(execution_id, "agent", ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.RUN, None, execution_id, None, event_seq=1)

    async def list_children(self, execution_id: str, *, principal: Principal):
        return ()

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs=None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ):
        if ready is not None:
            ready.set()
        del principal, include_content
        after = 0 if after_event_seqs is None else after_event_seqs.get(execution_id, 0)

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
    async def list_events(self, graph_id: str, *, principal: Principal, after_event_seq=0, limit=100):
        values = [item async for item in _TaskGraphService.stream_events(
            self, graph_id, principal=principal, after_event_seq=after_event_seq)]
        return Page(tuple(values[:limit]))

    async def wait(self, graph_id: str, *, principal: Principal, timeout_seconds=None):
        await asyncio.Event().wait()

    async def state(self, graph_id: str, *, principal: Principal):
        assert graph_id == "graph"
        return _graph_snapshot(graph_id, (("node", "execution"),), event_seq=4,
                               status=TaskStatus.SUCCEEDED)

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
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
                if event.event_seq > after_event_seq:
                    yield event

        return values()


class _HistoryService:
    async def subscribe_model_interactions(self, execution_id, *, principal):
        return None

    async def capture_model_interaction_cutoffs(self, execution_id, *, principal):
        return ModelInteractionReadBoundary((), (), True, True)

    async def read_model_interaction_metadata(self, execution_id, **kwargs):
        return ()

    async def list_execution_events(
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


def _graph_snapshot(graph_id, bindings, *, event_seq, status=TaskStatus.RUNNING,
                    node_status=TaskStatus.RUNNING):
    nodes = tuple(TaskNode(node_id) for node_id, _ in bindings)
    states = tuple(TaskNodeView(graph_id, node_id, (), node_status, None, 1,
        None, None, None, None, execution_id=execution_id) for node_id, execution_id in bindings)
    return TaskGraphState(graph_id, status, nodes, states, event_seq)


class _EventService:
    async def list(self, execution_id, *, principal, after_event_seq=0, limit=100):
        kind = (ExecutionEventType.EXECUTION_FAILED if execution_id.endswith("-failed")
                else ExecutionEventType.EXECUTION_SUCCEEDED)
        events = (ExecutionEvent(execution_id, 1, kind.value, {}),)
        return Page(tuple(event for event in events if event.event_seq > after_event_seq)[:limit])


def _prepare_observation_runtime(runtime):
    if not hasattr(runtime, "_ensure_open"):
        runtime._ensure_open = lambda: None
    if not hasattr(runtime, "history"):
        runtime.history = _HistoryService()
    executions = getattr(runtime, "executions", _ExecutionService())
    events = getattr(runtime, "events", _EventService())
    streamer = ExecutionTreeStreamer(executions, events, ExecutionTreeBroker())
    runtime._capture_execution_tree = streamer.capture
    runtime._replay_execution_tree = streamer.replay
    return runtime


class _Runtime:
    namespace = "watch-test"

    def __init__(self) -> None:
        self.execution = _ExecutionService()
        self.executions = self.execution
        self.graph = _TaskGraphService()
        self.history = _HistoryService()
        self.observations = set()
        _prepare_observation_runtime(self)

    def _register_observation(self, session) -> None:
        self.observations.add(session)

    def _release_observation(self, session) -> None:
        self.observations.discard(session)


def _task_graph_run(
    runtime,
    graph_id: str,
    principal: Principal,
    watch_tree,
) -> TaskGraphRun:
    return TaskGraphRun(
        _prepare_observation_runtime(runtime),
        runtime.graph,
        graph_id,
        principal,
        watch_tree,
    )


def _watch_tree(
    execution_id,
    *,
    principal,
    after_event_seqs=None,
    include_content=False,
    ready: asyncio.Event | None = None,
):
    if ready is not None:
        ready.set()
    return _ExecutionService().stream(
        execution_id,
        principal=principal,
        after_event_seqs=after_event_seqs,
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
async def test_task_graph_watch_forwards_model_request_progress_with_execution_identity() -> None:
    runtime = _Runtime()

    def watch_tree(
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs=None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ):
        if ready is not None:
            ready.set()
        del principal, after_event_seqs, include_content

        async def values():
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
                    ExecutionEventType.MODEL_REQUEST_STARTED.value,
                    {
                        "execution_id": execution_id,
                        "agent_run_seq": 2,
                        "model_request_seq": 1,
                        "purpose": "agent",
                        "status": "RUNNING",
                    },
                ),
            )

        return values()

    run = _task_graph_run(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        watch_tree,
    )
    events = [event async for event in run.watch()]
    progress = [
        event.event
        for event in events
        if isinstance(event.event, ExecutionTreeEvent)
        and event.event.event.event_type
        == ExecutionEventType.MODEL_REQUEST_STARTED.value
    ]

    assert len(progress) == 1
    assert progress[0].execution_id == "execution"
    assert progress[0].event.payload["execution_id"] == "execution"  # type: ignore[index]
    assert progress[0].event.payload["model_request_seq"] == 1  # type: ignore[index]


@pytest.mark.asyncio
async def test_task_graph_watch_starts_execution_before_binding_event_yield() -> None:
    started: list[str] = []

    def watch_tree(
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs=None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ):
        if ready is not None:
            ready.set()
        del principal, after_event_seqs, include_content
        started.append(execution_id)

        async def values():
            await asyncio.Event().wait()
            if False:
                yield None

        return values()

    run = _task_graph_run(
        _Runtime(),
        "graph",
        Principal("owner", "tenant"),
        watch_tree,
    )
    stream = run.watch()
    try:
        await anext(stream)
        assert started == ["execution"]
        async for binding in stream:
            if isinstance(binding.event, TaskEvent) and binding.event.execution_id is not None:
                assert binding.event.execution_id == "execution"
                assert started == ["execution"]
                break
        else:
            raise AssertionError("the binding event must be delivered")
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_task_graph_watch_refreshes_nodes_added_after_subscription() -> None:
    graph_id = "dynamic-watch"
    principal = Principal("owner", "tenant")
    expand = asyncio.Event()

    class DynamicGraphService:
        expanded = False

        def snapshot(self):
            bindings = [("expand", None)]
            if self.expanded:
                bindings.extend((("child-failed", "execution-failed"),
                                 ("child-succeeded", "execution-succeeded")))
            return _graph_snapshot(graph_id, bindings, event_seq=3 if self.expanded else 1)

        async def state(self, requested_graph_id: str, *, principal: Principal):
            assert requested_graph_id == graph_id
            assert principal == principal_arg
            return self.snapshot()

        def stream_events(
            self,
            requested_graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
        ):
            assert requested_graph_id == graph_id
            assert principal == principal_arg

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
                        "child-succeeded",
                        "worker",
                        1,
                        "execution-succeeded",
                    ),
                    TaskEvent(
                        1,
                        graph_id,
                        3,
                        TaskEventType.NODE_CHANGED,
                        now,
                        TaskStatus.RUNNING,
                        TaskStatus.READY,
                        "child-failed",
                        "worker",
                        1,
                        "execution-failed",
                    ),
                )
                for event in events:
                    if event.event_seq == 2:
                        await expand.wait()
                        self.expanded = True
                    if event.event_seq > after_event_seq:
                        yield event

            return values()

    principal_arg = principal
    graph = DynamicGraphService()

    class DynamicExecutions(_ExecutionService):
        async def inspect(self, execution_id: str, *, principal: Principal):
            assert principal == principal_arg
            assert execution_id in {"execution-succeeded", "execution-failed"}
            return await super().inspect(execution_id, principal=principal)

    class DynamicRuntime:
        namespace = "watch-test"

        def __init__(self, graph_service: DynamicGraphService) -> None:
            self.graph = graph_service
            self.executions = DynamicExecutions()

    def watch_tree(
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs=None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ):
        if ready is not None:
            ready.set()
        del include_content
        assert principal == principal_arg
        after_event_seq = (after_event_seqs or {}).get(execution_id, 0)
        event_type = (
            ExecutionEventType.EXECUTION_FAILED.value
            if execution_id == "execution-failed"
            else ExecutionEventType.EXECUTION_SUCCEEDED.value
        )

        async def values():
            if after_event_seq >= 1:
                return
            yield ExecutionTreeEvent(
                execution_id,
                "agent",
                ExecutionLineageKind.RUN,
                None,
                execution_id,
                None,
                0,
                ExecutionStreamEvent(execution_id, 1, event_type, {}),
            )

        return values()

    runtime = DynamicRuntime(graph)
    run = _task_graph_run(runtime, graph_id, principal, watch_tree)
    first_watch = run.watch()
    try:
        first = await anext(first_watch)
        while not isinstance(first.event, TaskEvent):
            first = await anext(first_watch)
        assert first.event.event_seq == 1
        expand.set()
        observed = [first]
        observed.extend([item async for item in first_watch])
    finally:
        await first_watch.aclose()

    task_events = [item for item in observed if isinstance(item.event, TaskEvent)]
    assert [item.event.event_seq for item in task_events] == [1, 2, 3]
    execution_events = [
        item.event.event
        for item in observed
        if isinstance(item.event, ExecutionTreeEvent)
    ]
    assert {
        (item.execution_id, item.event_type)
        for item in execution_events
    } == {
        ("execution-succeeded", ExecutionEventType.EXECUTION_SUCCEEDED.value),
        ("execution-failed", ExecutionEventType.EXECUTION_FAILED.value),
    }

    first_execution = next(
        item for item in observed if isinstance(item.event, ExecutionTreeEvent)
    )
    assert first_execution.cursor is not None
    resumed = [item async for item in run.watch(cursor=first_execution.cursor)]
    assert {
        item.event.execution_id
        for item in resumed
        if isinstance(item.event, ExecutionTreeEvent)
    } == {
        "execution-succeeded",
        "execution-failed",
    } - {first_execution.event.execution_id}


@pytest.mark.asyncio
async def test_task_graph_watch_preserves_missing_dynamic_node_integrity_error() -> None:
    cause_details = {"graph_id": "graph", "node_id": "late-node"}

    class MissingNodeGraphService:
        async def state(self, graph_id: str, *, principal: Principal):
            return _graph_snapshot(graph_id, (("existing", None),), event_seq=4)

        def stream_events(self, graph_id: str, *, principal: Principal, after_event_seq: int = 0):
            del principal, after_event_seq

            async def values():
                yield TaskEvent(
                    1,
                    graph_id,
                    1,
                    TaskEventType.NODE_CHANGED,
                    datetime.now(timezone.utc),
                    TaskStatus.RUNNING,
                    TaskStatus.READY,
                    "late-node",
                    "worker",
                    1,
                    "late-execution",
                )

            return values()

    class MissingNodeRuntime:
        namespace = "watch-test"
        graph = MissingNodeGraphService()
        executions = _ExecutionService()

    run = _task_graph_run(
        MissingNodeRuntime(),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    with pytest.raises(AIError) as raised:
        async for _event in run.watch():
            pass

    assert not isinstance(raised.value, ObservationError)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert raised.value.safe_details == cause_details


@pytest.mark.asyncio
async def test_task_graph_run_watch_merges_task_and_execution_events(monkeypatch) -> None:
    now = datetime.now(timezone.utc).timestamp()
    monkeypatch.setattr(_cursor, "time", SimpleNamespace(time=lambda: now))
    run = _task_graph_run(
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
    position = decode_graph_watch_cursor(
        "watch-test", "tenant", "graph", values[-1].cursor, include_content=False,
    )
    assert position == (4, {"node": {"execution": 1}})
    now += 1
    resumed = [item async for item in run.watch(cursor=values[-1].cursor)]
    assert resumed
    assert all(isinstance(item.event, (TaskGraphProjection, TaskModelProjection)) for item in resumed)
    assert all(decode_graph_watch_cursor(
        "watch-test", "tenant", "graph", item.cursor, include_content=False,
    ) == position for item in resumed)
    with pytest.raises(AIError) as raised:
        run.watch(cursor=values[-1].cursor, include_content=True)
    assert raised.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
async def test_task_graph_watch_rejects_unbound_cursor_before_starting_stream() -> None:
    class GraphService:
        def __init__(self) -> None:
            self.stream_calls = 0

        async def state(self, graph_id: str, *, principal: Principal):
            assert graph_id == "graph"
            return _graph_snapshot(graph_id, (("node", None),), event_seq=4)

        def stream_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
        ):
            del graph_id, principal, after_event_seq
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
    run = _task_graph_run(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    stream = run.watch(cursor=encode_graph_watch_cursor(
        "watch-test", "tenant", "graph", include_content=False,
        graph_event_seq=0, execution_event_seqs={"node": {"execution": 1}},
    ))
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
        async def state(self, graph_id: str, *, principal: Principal):
            assert graph_id == "graph"
            return _graph_snapshot(graph_id, (), event_seq=2)

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            assert graph_id == "graph"
            if after_event_seq == 0:
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
    run = _task_graph_run(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[int] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        assert isinstance(event.event, TaskEvent)
        observed.append(event.event.event_seq)
        if event.event.event_seq == 1:
            observed_first.set()

    await asyncio.wait_for(run.replay(observer), timeout=1)
    assert observed == [1, 2]


@pytest.mark.asyncio
async def test_task_graph_replay_uses_captured_durable_cutoffs() -> None:
    now = datetime.now(timezone.utc)

    class ReplayGraphService:
        async def state(self, graph_id: str, *, principal: Principal):
            assert graph_id == "graph"
            return _graph_snapshot(graph_id, (("node", "execution"),), event_seq=2,
                                   node_status=TaskStatus.WAITING)

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
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
            return Page(tuple(value for value in values if value.event_seq > after_event_seq))

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
                event_seq=2,
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
            after_event_seq: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            assert execution_id == "execution"
            values = (
                ExecutionEvent(execution_id, 1, "TOOL_CALL_STARTED", {"call_id": "call", "tool_name": "tool", "arguments": "secret"}),
                ExecutionEvent(execution_id, 2, "EXECUTION_SUCCEEDED", {"raw": "two"}),
                ExecutionEvent(execution_id, 3, "LATE_EVENT", {"raw": "late"}),
            )
            return Page(tuple(value for value in values if value.event_seq > after_event_seq))

    replay_execution = ReplayExecutionService()
    replay_events = ReplayEventService()
    runtime = type(
        "ReplayRuntime",
        (),
        {
            "namespace": "watch-test",
            "graph": ReplayGraphService(),
            "execution": replay_execution,
            "executions": replay_execution,
            "event": replay_events,
            "events": replay_events,
        },
    )()
    run = _task_graph_run(
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
        event.event.event_seq
        for event in observed
        if isinstance(event.event, TaskEvent)
    ] == [1, 2]
    execution_events = [
        event.event.event
        for event in observed
        if isinstance(event.event, ExecutionTreeEvent)
    ]
    assert [event.durable_seq for event in execution_events] == [1, 2]
    assert execution_events[0].payload == {"call_id": "call", "tool_name": "tool"}
    assert execution_events[1].payload == {}
    assert all(event.cursor is not None for event in observed)
    assert all(
        item.event.cursor is not None
        for item in observed
        if isinstance(item.event, ExecutionTreeEvent)
    )



@pytest.mark.asyncio
@pytest.mark.parametrize("root_id", ["root", "child"])
async def test_task_graph_replay_captures_recursive_members_with_relative_depth(root_id: str) -> None:
    now = datetime.now(timezone.utc)

    class GraphService:
        async def state(self, graph_id: str, *, principal: Principal):
            assert graph_id == "graph"
            return _graph_snapshot(graph_id, (("node", root_id),), event_seq=1)

        async def list_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
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
            return Page(tuple(value for value in values if value.event_seq > after_event_seq))

    views = {
        "root": ExecutionView(
            "root",
            "agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.RUN,
            None,
            "root",
            None,
            event_seq=1,
        ),
        "child": ExecutionView(
            "child",
            "child-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.SUBAGENT,
            "root",
            "root",
            "call-child",
            event_seq=1,
        ),
        "grandchild": ExecutionView(
            "grandchild",
            "grandchild-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.SUBAGENT,
            "child",
            "root",
            "call-grandchild",
            event_seq=1,
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
            return tuple(view for view in views.values() if view.parent_execution_id == execution_id)

    class EventService:
        async def list(
            self,
            execution_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
            limit: int = 100,
        ):
            del principal, limit
            values = (
                ExecutionEvent(execution_id, 1, "EXECUTION_STARTED", {"raw": "one"}),
                ExecutionEvent(execution_id, 2, "LATE_EVENT", {"raw": "late"}),
            )
            return Page(tuple(value for value in values if value.event_seq > after_event_seq))

    replay_execution = ExecutionService()
    replay_events = EventService()
    runtime = type(
        "ReplayRuntime",
        (),
        {
            "namespace": "watch-test",
            "graph": GraphService(),
            "execution": replay_execution,
            "executions": replay_execution,
            "event": replay_events,
            "events": replay_events,
        },
    )()
    run = _task_graph_run(
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
        (event.execution_id, event.depth, event.event.durable_seq)
        for event in execution_events
    ] == (
        [("root", 0, 1), ("child", 1, 1), ("grandchild", 2, 1)]
        if root_id == "root" else [("child", 0, 1), ("grandchild", 1, 1)]
    )
    assert all(event.event.payload == {} for event in execution_events)


class _WaitGraphService:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.wait_started = asyncio.Event()
        self.wait_cancelled = asyncio.Event()
        self.stream_release = asyncio.Event()

    async def state(self, graph_id: str, *, principal: Principal):
        assert graph_id == "graph"
        return _graph_snapshot(graph_id, (), event_seq=1,
            status=TaskStatus.RECOVERY_REQUIRED if self.mode == "recovery" else TaskStatus.RUNNING)

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_event_seq: int = 0,
    ):
        del principal, after_event_seq

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
    ) -> TaskGraphState:
        del principal, timeout_seconds
        assert graph_id == "graph"
        self.wait_started.set()
        if self.mode == "waiting":
            return TaskGraphState(graph_id, TaskStatus.WAITING, (), ())
        if self.mode == "recovery":
            return TaskGraphState(graph_id, TaskStatus.RECOVERY_REQUIRED, (), ())
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
    runtime = _Runtime()
    runtime.graph = service
    return runtime


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
async def test_task_graph_wait_does_not_start_observer_on_stable_waiting() -> None:
    service = _WaitGraphService("waiting")
    run = _task_graph_run(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    result = await run.wait()

    assert result.result.wait_status is TaskStatus.WAITING
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_wait_and_watch_are_independent_at_recovery_boundary() -> None:
    service = _WaitGraphService("recovery")
    run = _task_graph_run(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )
    observed: list[TaskGraphRunEvent] = []

    async def observer(event: TaskGraphRunEvent) -> None:
        observed.append(event)

    result = await run.wait()

    assert result.result.wait_status is TaskStatus.RECOVERY_REQUIRED
    assert not observed
    async for event in run.watch():
        await observer(event)
    durable = [item for item in observed if isinstance(item.event, TaskEvent)]
    assert len(durable) == 1
    assert durable[0].event.status is TaskStatus.RECOVERY_REQUIRED
    assert durable[0].cursor is not None
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_wait_cleans_up_on_timeout() -> None:
    service = _WaitGraphService("timeout")
    run = _task_graph_run(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    with pytest.raises(AIError) as raised:
        await run.wait()

    assert raised.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_observer_error_does_not_start_graph_wait() -> None:
    service = _WaitGraphService("observer_error")
    run = _task_graph_run(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    async def observer(_event: TaskGraphRunEvent) -> None:
        raise RuntimeError("observer failed")

    stream = run.watch()
    try:
        with pytest.raises(RuntimeError, match="observer failed"):
            async for event in stream:
                await observer(event)
    finally:
        await stream.aclose()

    assert not service.wait_started.is_set()
    assert not service.wait_cancelled.is_set()
    await _assert_no_graph_observer_tasks()


@pytest.mark.asyncio
async def test_task_graph_live_stream_failure_keeps_stream_origin_and_cursor() -> None:
    cause = RuntimeError("optional broker is unavailable")
    delivered_first = asyncio.Event()

    def failed_watch_tree(
        execution_id,
        *,
        principal,
        after_event_seqs=None,
        include_content=False,
        ready: asyncio.Event | None = None,
    ):
        if ready is not None:
            ready.set()
        del execution_id, principal, after_event_seqs, include_content

        async def values():
            await delivered_first.wait()
            if False:
                yield None
            raise _ExecutionStreamFailure(cause)

        return values()

    run = _task_graph_run(
        _Runtime(),
        "graph",
        Principal("owner", "tenant"),
        failed_watch_tree,
    )
    stream = run.watch()
    delivered: list[TaskGraphRunEvent] = []
    with pytest.raises(ObservationError) as raised:
        while True:
            delivered.append(await anext(stream))
            delivered_first.set()

    assert raised.value.origin == "stream"
    assert raised.value.cause_code is None
    assert raised.value.safe_details == {"cause_type": "RuntimeError"}
    assert delivered
    assert raised.value.cursor == delivered[-1].cursor
    assert raised.value.__cause__ is cause


@pytest.mark.asyncio
async def test_task_graph_durable_stream_integrity_error_remains_raw() -> None:
    class BrokenGraphService(_TaskGraphService):
        def stream_events(
            self,
            graph_id: str,
            *,
            principal: Principal,
            after_event_seq: int = 0,
        ):
            del graph_id, principal, after_event_seq

            async def values():
                if False:
                    yield None
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            return values()

    runtime = _Runtime()
    runtime.graph = BrokenGraphService()
    run = _task_graph_run(
        runtime,
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    with pytest.raises(AIError) as raised:
        async for _event in run.watch():
            pass

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not isinstance(raised.value, ObservationError)


@pytest.mark.asyncio
async def test_task_graph_wait_outer_cancel_cleans_waiter() -> None:
    service = _WaitGraphService("block")
    run = _task_graph_run(
        _wait_runtime(service),
        "graph",
        Principal("owner", "tenant"),
        _watch_tree,
    )

    task = asyncio.create_task(
        run.wait(),
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


def test_runtime_domain_facades_match_the_supported_method_sets() -> None:
    execution_methods = {
        name
        for name, value in vars(RuntimeExecutions).items()
        if callable(value) and not name.startswith("_")
    }
    session_methods = {
        name
        for name, value in vars(RuntimeSessions).items()
        if callable(value) and not name.startswith("_")
    }

    assert execution_methods == {
        "get",
        "inspect",
        "list",
        "list_children",
        "result",
        "wait",
        "retry",
        "fork",
        "cancel",
        "recovery_effects",
        "resolve_tool_effect",
        "recover",
        "trace",
        "transcript",
        "history",
        "model_interactions",
        "capture_input",
        "budget_usage",
    }
    assert session_methods == {
        "get",
        "create",
        "reconcile",
        "list",
        "history",
        "timeline",
        "fork",
        "update",
        "close",
    }


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


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["initial_state", "dynamic_state", "inspect"])
@pytest.mark.parametrize("typed", [False, True])
async def test_task_graph_watch_preserves_authoritative_read_failures(
    boundary: str,
    typed: bool,
) -> None:
    cause = AIError(ErrorCode.STORAGE_UNAVAILABLE) if typed else OSError("read failed")

    class BrokenGraphService(_TaskGraphService):
        async def state(self, graph_id: str, *, principal: Principal):
            if boundary == "initial_state":
                raise cause
            snapshot = await super().state(graph_id, principal=principal)
            if boundary == "dynamic_state":
                if getattr(self, "read", False):
                    raise cause
                self.read = True
                snapshot = replace(snapshot, nodes=(), node_states=())
            return snapshot

    class BrokenExecutions(_ExecutionService):
        async def inspect(self, execution_id: str, *, principal: Principal):
            if boundary == "inspect":
                raise cause
            return await super().inspect(execution_id, principal=principal)

    runtime = _Runtime()
    runtime.graph = BrokenGraphService()
    runtime.executions = BrokenExecutions()
    run = _task_graph_run(runtime, "graph", Principal("owner", "tenant"), _watch_tree)

    with pytest.raises(type(cause)) as raised:
        async for _event in run.watch():
            pass

    assert raised.value is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["encode_graph_watch_cursor", "encode_execution_watch_cursor"])
async def test_task_graph_watch_preserves_cursor_protocol_failures(
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    cause = ValueError("invalid cursor identity")

    def fail(*args, **kwargs):
        raise cause

    monkeypatch.setattr(f"linktools.ai.runtime._task.{encoding}", fail)
    run = _task_graph_run(_Runtime(), "graph", Principal("owner", "tenant"), _watch_tree)

    with pytest.raises(ValueError) as raised:
        async for _event in run.watch():
            pass

    assert raised.value is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["sync", "async", "cancel"])
async def test_task_graph_observer_preserves_cause_and_last_acknowledged_cursor(
    failure: str,
) -> None:
    cause = (
        asyncio.CancelledError("callback cancelled")
        if failure == "cancel"
        else AIError(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"source": "callback"})
    )
    run = _task_graph_run(_Runtime(), "graph", Principal("owner", "tenant"), _watch_tree)
    delivered: list[TaskGraphRunEvent] = []

    async def callback(event: TaskGraphRunEvent) -> None:
        if isinstance(event.event, (TaskGraphProjection, TaskModelProjection)):
            return
        await asyncio.sleep(0)
        if delivered:
            raise cause
        delivered.append(event)

    def observer(event: TaskGraphRunEvent):
        if delivered and failure == "sync" and not isinstance(event.event, (TaskGraphProjection, TaskModelProjection)):
            raise cause
        return callback(event)

    with pytest.raises(asyncio.CancelledError if failure == "cancel" else ObservationError) as raised:
        await run.wait(on_event=observer)

    assert len(delivered) == 1
    if failure == "cancel":
        assert raised.value is cause
    else:
        assert raised.value.origin == "callback"
        assert raised.value.__cause__ is cause
        assert raised.value.cursor == delivered[0].cursor
        assert raised.value.cause_code == ErrorCode.STORAGE_UNAVAILABLE.value
        assert raised.value.safe_details == {"source": "callback"}


@pytest.mark.asyncio
async def test_task_graph_watch_preserves_same_round_durable_failure() -> None:
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    class BrokenGraphService(_TaskGraphService):
        def stream_events(self, *args, **kwargs):
            async def values():
                raise cause
                yield
            return values()

    def broken_tree(*args, **kwargs):
        async def values():
            raise _ExecutionStreamFailure(OSError("broker failed"))
            yield
        return values()

    runtime = _Runtime()
    runtime.graph = BrokenGraphService()
    run = _task_graph_run(runtime, "graph", Principal("owner", "tenant"), broken_tree)

    with pytest.raises(AIError) as raised:
        await anext(run.watch(cursor=encode_graph_watch_cursor(
            "watch-test", "tenant", "graph", include_content=False,
            graph_event_seq=1, execution_event_seqs={},
        )))

    assert raised.value is cause


@pytest.mark.asyncio
async def test_task_graph_watch_preserves_authoritative_failure_during_cleanup() -> None:
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    started = asyncio.Event()

    class BrokenGraphService(_TaskGraphService):
        def stream_events(self, *args, **kwargs):
            async def values():
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise cause
                yield
            return values()

    def broken_tree(*args, **kwargs):
        async def values():
            await started.wait()
            raise _ExecutionStreamFailure(OSError("broker failed"))
            yield
        return values()

    runtime = _Runtime()
    runtime.graph = BrokenGraphService()
    run = _task_graph_run(runtime, "graph", Principal("owner", "tenant"), broken_tree)

    with pytest.raises(AIError) as raised:
        await anext(run.watch(cursor=encode_graph_watch_cursor(
            "watch-test", "tenant", "graph", include_content=False,
            graph_event_seq=1, execution_event_seqs={},
        )))

    assert raised.value is cause


@pytest.mark.asyncio
async def test_graph_live_subagent_root_cursor_binds_selected_execution():
    from types import SimpleNamespace
    from linktools.ai.runtime._watch_cursor import decode_execution_watch_cursor
    from linktools.ai.task import TaskNode, TaskNodeView

    principal = Principal("owner", "tenant")
    node = TaskNode("node")
    state = TaskGraphState("graph", TaskStatus.SUCCEEDED, (node,), (
        TaskNodeView("graph", "node", (), TaskStatus.SUCCEEDED, None, 1, None, None, None, None,
                     execution_id="selected"),
    ), 1)
    class Graph:
        async def state(self, graph_id, *, principal):
            return state
        async def stream_events(self, graph_id, *, principal, after_event_seq=0):
            if after_event_seq < 1:
                yield TaskEvent(1, "graph", 1, TaskEventType.GRAPH_ADMITTED,
                                datetime.now(timezone.utc), TaskStatus.PENDING)
    async def inspect(execution_id, *, principal):
        assert execution_id == "selected"
        return ExecutionView("selected", "agent", ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.SUBAGENT, "parent", "lineage-root", "invocation", event_seq=1)
    async def tree(execution_id, *, principal, after_event_seqs=None, include_content=False, ready=None):
        assert execution_id == "selected"
        if ready is not None:
            ready.set()
        yield ExecutionTreeEvent(
            "selected", "agent", ExecutionLineageKind.SUBAGENT,
            "parent", "lineage-root", "invocation", 0,
            ExecutionStreamEvent("selected", 1, ExecutionEventType.EXECUTION_SUCCEEDED.value, {}),
        )
    async def list_children(execution_id, *, principal):
        return ()
    runtime = SimpleNamespace(namespace="selected-subtree", executions=SimpleNamespace(
        inspect=inspect, list_children=list_children))
    handle = TaskGraphRun(_prepare_observation_runtime(runtime), Graph(), "graph", principal, tree)
    items = [item async for item in handle.watch()]
    nested = next(item.event for item in items if isinstance(item.event, ExecutionTreeEvent))
    assert nested.root_execution_id == "lineage-root" and nested.depth == 0
    assert decode_execution_watch_cursor("selected-subtree", "tenant", "selected", nested.cursor,
                                         include_content=False) == {"selected": 1}
    with pytest.raises(AIError) as error:
        decode_execution_watch_cursor("selected-subtree", "tenant", "lineage-root", nested.cursor,
                                      include_content=False)
    assert error.value.code is ErrorCode.CURSOR_INVALID
