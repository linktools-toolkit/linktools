#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Task event history is the only authority for Task metrics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import Page, TaskStatus, canonical_sha256
from linktools.ai.observe import Observation
from linktools.ai.task._event import TaskEvent, TaskEventType
from linktools.ai.task._metrics import _TaskMetricProjector

pytestmark = pytest.mark.asyncio


class _Repository:
    def __init__(self, events: tuple[TaskEvent, ...]) -> None:
        self.events = events

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        del tenant_id
        selected = tuple(
            event
            for event in self.events
            if event.graph_id == graph_id and event.sequence > after_sequence
        )
        page = selected[:limit]
        return Page(page, "more" if len(selected) > limit else None)

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None:
        del tenant_id
        selected = tuple(event for event in self.events if event.graph_id == graph_id)
        return selected[-1] if selected else None


class _Recorder:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.observations: list[Observation] = []
        self.calls = 0
        self.fail_first = fail_first

    def try_record(self, observation: Observation) -> bool:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("synthetic metric failure")
        self.observations.append(observation)
        return True


def _admitted(graph_id: str, at: datetime, *, terminal: bool = False) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        1,
        TaskEventType.GRAPH_ADMITTED,
        at,
        TaskStatus.SUCCEEDED if terminal else TaskStatus.PENDING,
    )


def _running(
    graph_id: str,
    sequence: int,
    at: datetime,
    *,
    fence: int,
    execution_id: str | None = None,
    previous_status: TaskStatus = TaskStatus.READY,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        TaskStatus.RUNNING,
        previous_status=previous_status,
        node_id="node",
        owner="owner",
        fence=fence,
        execution_id=execution_id,
    )


def _succeeded(
    graph_id: str,
    sequence: int,
    at: datetime,
    *,
    fence: int,
    execution_id: str | None,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        TaskStatus.SUCCEEDED,
        previous_status=TaskStatus.RUNNING,
        node_id="node",
        fence=fence,
        execution_id=execution_id,
        result_digest=canonical_sha256({"result": sequence}),
    )


def _graph_terminal(
    graph_id: str,
    sequence: int,
    at: datetime,
    status: TaskStatus = TaskStatus.SUCCEEDED,
) -> TaskEvent:
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.GRAPH_CHANGED,
        at,
        status,
        previous_status=TaskStatus.PENDING,
    )


async def _project(events: tuple[TaskEvent, ...], recorder: _Recorder) -> None:
    projector = _TaskMetricProjector(
        _Repository(events),
        recorder,
        source_namespace="workspace",
    )
    await projector._project(events[0].graph_id, tenant_id="tenant")


async def test_empty_graph_has_exact_zero_latency() -> None:
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    event = _admitted("empty", now, terminal=True)
    recorder = _Recorder()

    await _project((event,), recorder)

    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.task.graph.terminal"
    assert observation.status == "SUCCEEDED"
    assert observation.measurements[0].name == "latency_ns"
    assert observation.measurements[0].value == 0


async def test_node_attempt_uses_first_running_event_for_same_fence() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("graph", start),
        _running("graph", 2, start + timedelta(seconds=1), fence=1),
        _running(
            "graph",
            3,
            start + timedelta(seconds=3),
            fence=1,
            execution_id="execution",
            previous_status=TaskStatus.RUNNING,
        ),
        _succeeded(
            "graph",
            4,
            start + timedelta(seconds=6),
            fence=1,
            execution_id="execution",
        ),
        _graph_terminal("graph", 5, start + timedelta(seconds=7)),
    )
    recorder = _Recorder()

    await _project(events, recorder)

    attempt = next(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    )
    assert attempt.measurements[0].value == 5_000_000_000
    assert dict(attempt.correlation) == {
        "execution_id": "execution",
        "fence": 1,
        "graph_id": "graph",
        "node_id": "node",
    }


async def test_unmatched_old_fence_is_not_paired_with_new_attempt() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("reclaim", start),
        _running("reclaim", 2, start + timedelta(seconds=1), fence=1),
        _running(
            "reclaim",
            3,
            start + timedelta(seconds=4),
            fence=2,
            execution_id="execution-2",
            previous_status=TaskStatus.RUNNING,
        ),
        _succeeded(
            "reclaim",
            4,
            start + timedelta(seconds=9),
            fence=2,
            execution_id="execution-2",
        ),
        _graph_terminal("reclaim", 5, start + timedelta(seconds=10)),
    )
    recorder = _Recorder()

    await _project(events, recorder)

    attempts = tuple(
        observation
        for observation in recorder.observations
        if observation.kind == "linktools.task.node.attempt"
    )
    assert len(attempts) == 1
    assert attempts[0].correlation["fence"] == 2
    assert attempts[0].measurements[0].value == 5_000_000_000


async def test_one_rejected_task_fact_does_not_abort_later_attempts() -> None:
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    events = (
        _admitted("fail-open", start),
        _running("fail-open", 2, start + timedelta(seconds=1), fence=1),
        _succeeded(
            "fail-open",
            3,
            start + timedelta(seconds=2),
            fence=1,
            execution_id=None,
        ),
        _graph_terminal("fail-open", 4, start + timedelta(seconds=3)),
    )
    recorder = _Recorder(fail_first=True)

    await _project(events, recorder)

    assert recorder.calls == 2
    assert [observation.kind for observation in recorder.observations] == [
        "linktools.task.node.attempt"
    ]
