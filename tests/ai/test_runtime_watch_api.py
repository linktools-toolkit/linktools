#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from linktools.ai.runtime import Execution, Runtime, TaskGraphRun, TaskGraphRunEvent
from linktools.ai.runtime.service_api import (
    ExecutionEvent,
    ExecutionStreamEvent,
    ExecutionTreeEvent,
    ExecutionView,
)
from linktools.ai.task import TaskEvent, TaskEventType


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


class _Runtime:
    namespace = "watch-test"

    def __init__(self) -> None:
        self.execution = _ExecutionService()
        self.graph = _TaskGraphService()


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
async def test_execution_watch_projects_complete_execution_tree() -> None:
    runtime = _Runtime()
    execution = Execution(
        runtime,
        "execution",
        "binding",
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
    assert values[-1].cursor is not None
    assert [item async for item in run.watch(cursor=values[-1].cursor)] == []
    with pytest.raises(AIError) as raised:
        run.watch(cursor=values[-1].cursor, include_content=True)
    assert raised.value.code is ErrorCode.CURSOR_INVALID


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
                ExecutionStatus.RUNNING,
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
