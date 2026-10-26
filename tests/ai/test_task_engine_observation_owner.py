#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph observation, control and budget reads belong to their TaskEngine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone

import pytest

from linktools.ai.core import BudgetUsage, Page, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task_engine._run import TaskGraphRun
from linktools.ai.execution import ExecutionTreeEvent
from linktools.ai.task import CancelGraphRequest, TaskEvent, TaskEventType, TaskGraphState


class _Engine:
    def _bind_graph_history(self, state):
        pass

    namespace = "independent"
    _history = None

    def __init__(self) -> None:
        self._cancel_settlements: set[object] = set()
        self.activated: tuple[str, Principal] | None = None
        self.budget_request: tuple[str, Principal] | None = None

    def _ensure_open(self) -> None:
        pass

    async def _activate_graph(self, graph_id: str, principal: Principal) -> None:
        self.activated = graph_id, principal

    async def budget_usage(self, graph_id: str, *, principal: Principal) -> BudgetUsage | None:
        self.budget_request = graph_id, principal
        return None


class _Graph:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.cancelled: tuple[str, Principal] | None = None
        self.event_cutoff = 1
        self.stream_closed = False

    async def cancel(
        self, graph_id: str, request: CancelGraphRequest, *,
        admission_guard: Callable[[], None] | None = None,
    ) -> None:
        if admission_guard is not None:
            admission_guard()
        assert self.engine.activated == (graph_id, request.principal)
        self.cancelled = graph_id, request.principal

    async def state(self, graph_id: str, *, principal: Principal) -> TaskGraphState:
        return TaskGraphState(graph_id, TaskStatus.CANCELLED, (), (), self.event_cutoff)

    async def stream_events(
        self, graph_id: str, *, principal: Principal, after_event_seq: int = 0,
    ) -> AsyncIterator[TaskEvent]:
        try:
            page = await self.list_events(graph_id, principal=principal, after_event_seq=after_event_seq)
            for event in page.items:
                yield event
        finally:
            self.stream_closed = True

    async def list_events(
        self, graph_id: str, *, principal: Principal, after_event_seq: int = 0, limit: int = 100,
    ) -> Page[TaskEvent]:
        return Page(tuple(TaskEvent(
            1, graph_id, sequence, TaskEventType.GRAPH_CHANGED, datetime.now(timezone.utc),
            TaskStatus.CANCELLED, TaskStatus.RUNNING,
        ) for sequence in range(after_event_seq + 1, min(self.event_cutoff, after_event_seq + limit) + 1)))


async def _unused_tree(*args: object, **kwargs: object) -> AsyncIterator[ExecutionTreeEvent]:
    if False:
        yield


def _build() -> tuple[_Engine, _Graph, TaskGraphRun, Principal]:
    engine = _Engine()
    principal = Principal("worker", "tenant")
    graph = _Graph(engine)
    return engine, graph, TaskGraphRun(engine, graph, "graph", principal, _unused_tree), principal


@pytest.mark.asyncio
async def test_budget_read_uses_graph_owner_and_admitted_principal() -> None:
    engine, _, run, principal = _build()
    assert await run.budget_usage() is None
    assert engine.budget_request == ("graph", principal)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 1.0])
async def test_cancellation_activates_owner_and_releases_settlement(timeout: float | None) -> None:
    engine, graph, run, principal = _build()
    result = await run.cancel(settle_timeout_seconds=timeout)
    assert result.status is TaskStatus.CANCELLED
    assert graph.cancelled == ("graph", principal)
    assert not engine._cancel_settlements


@pytest.mark.asyncio
async def test_zero_deadline_does_not_submit_control() -> None:
    engine, graph, run, _ = _build()
    with pytest.raises(AIError) as error:
        await run.cancel(settle_timeout_seconds=0)
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert engine.activated is None and graph.cancelled is None
    assert not engine._cancel_settlements


@pytest.mark.asyncio
async def test_replay_freezes_graph_cutoff_before_consumption() -> None:
    _, graph, run, _ = _build()
    stream = await run.replay_events()
    graph.event_cutoff = 5
    assert [event.event.event_seq async for event in stream] == [1]


@pytest.mark.asyncio
async def test_watch_signals_ready_and_closes_owned_subscription() -> None:
    _, graph, run, _ = _build()
    ready = asyncio.Event()
    stream = run.watch(ready=ready)
    try:
        await stream.__anext__()
        assert ready.is_set()
    finally:
        await stream.aclose()
    assert graph.stream_closed
