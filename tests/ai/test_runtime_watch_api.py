#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone

import pytest

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    Principal,
    TaskStatus,
)
from linktools.ai.runtime import Execution, TaskGraphRunEvent
from linktools.ai.runtime._task import TaskGraphRun
from linktools.ai.runtime.service_api import (
    ExecutionStreamEvent,
    ExecutionTreeEvent,
)
from linktools.ai.task import TaskEvent, TaskEventType


class _ExecutionService:
    def stream_tree(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequences=None,
    ):
        del principal, after_sequences

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
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    {},
                ),
            )

        return values()


class _TaskService:
    async def snapshot_graph(self, graph_id: str, *, principal: Principal):
        del principal

        class Snapshot:
            node_states = ()

        assert graph_id == "graph"
        return Snapshot()

    def stream_graph_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ):
        del principal

        async def values():
            now = datetime.now(timezone.utc)
            yield TaskEvent(
                1,
                graph_id,
                after_sequence + 1,
                TaskEventType.GRAPH_ADMITTED,
                now,
                TaskStatus.PENDING,
            )
            yield TaskEvent(
                1,
                graph_id,
                after_sequence + 2,
                TaskEventType.NODE_CHANGED,
                now,
                TaskStatus.RUNNING,
                TaskStatus.READY,
                "node",
                "worker",
                1,
                "execution",
            )
            yield TaskEvent(
                1,
                graph_id,
                after_sequence + 3,
                TaskEventType.GRAPH_CHANGED,
                now,
                TaskStatus.SUCCEEDED,
                TaskStatus.RUNNING,
            )

        return values()


class _Runtime:
    def __init__(self) -> None:
        self.execution = _ExecutionService()
        self.task = _TaskService()

    def stream_tree(self, execution_id, *, principal, after_sequences=None):
        return self.execution.stream_tree(
            execution_id,
            principal=principal,
            after_sequences=after_sequences,
        )


@pytest.mark.asyncio
async def test_execution_watch_projects_complete_execution_tree() -> None:
    runtime = _Runtime()
    execution = Execution(
        runtime,
        "execution",
        "binding",
        Principal("owner", "tenant"),
    )
    values = [item async for item in execution.watch()]
    assert len(values) == 1
    assert values[0].execution_id == "execution"


@pytest.mark.asyncio
async def test_task_graph_run_watch_merges_task_and_execution_events() -> None:
    run = TaskGraphRun(_Runtime(), "graph", Principal("owner", "tenant"))
    values = [item async for item in run.watch()]
    assert all(isinstance(item, TaskGraphRunEvent) for item in values)
    assert [type(item.event) for item in values].count(TaskEvent) == 3
    execution = [
        item for item in values if isinstance(item.event, ExecutionTreeEvent)
    ]
    assert len(execution) == 1
    assert execution[0].node_id == "node"
    assert execution[0].event.execution_id == "execution"


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
            ExecutionEventType.EXECUTION_SUCCEEDED,
            {},
        ),
    )
    with pytest.raises(ValueError):
        TaskGraphRunEvent("graph", None, event)
