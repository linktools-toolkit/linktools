#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-owned best-effort projection from durable event history into Metrics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from linktools.core import environ

from ..core import Page, CorrelationData, TaskStatus, canonical_sha256
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._event import TaskEvent, TaskEventType

_logger = environ.get_logger("ai.task.metrics")
_EVENT_PAGE_SIZE = 1000
_MAX_PROJECTION_EVENTS = 100_000
_PROJECTION_TIMEOUT_SECONDS = 5.0
_PROJECTED_CACHE_SIZE = 4096
_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


class _TaskMetricRepository(Protocol):
    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_event_seq: int,
        limit: int,
    ) -> Page[TaskEvent]: ...

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None: ...


class _TaskMetricAdmission(Protocol):
    correlation: CorrelationData


class _TaskMetricAdmissionRepository(Protocol):
    async def get(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> _TaskMetricAdmission | None: ...


@dataclass(slots=True)
class _Attempt:
    execution_id: str | None
    started_at: datetime
    attempt_index: int
    ready_at: datetime | None = None
    invalid: bool = False
    terminal: TaskEvent | None = None


@dataclass(slots=True)
class _Projection:
    admission: TaskEvent
    terminal: TaskEvent
    attempts: dict[tuple[str, int], _Attempt]
    correlation: CorrelationData


class _TaskMetricProjector:
    def __init__(
        self,
        repository: _TaskMetricRepository,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
        admissions: _TaskMetricAdmissionRepository | None = None,
    ) -> None:
        self._repository = repository
        self._recorder = recorder
        self._admissions = admissions
        self._source_namespace = source_namespace
        self._active: dict[tuple[str, str], asyncio.Event] = {}
        self._projected: dict[tuple[str, str], None] = {}
        self._accepting = True

    async def trigger(self, graph_id: str, *, tenant_id: str) -> None:
        key = (tenant_id, graph_id)
        while self._accepting and key not in self._projected:
            active = self._active.get(key)
            if active is not None:
                await active.wait()
                continue
            completed = asyncio.Event()
            self._active[key] = completed
            try:
                if await self._project(graph_id, tenant_id=tenant_id):
                    self._projected[key] = None
                    while len(self._projected) > _PROJECTED_CACHE_SIZE:
                        self._projected.pop(next(iter(self._projected)))
            except Exception as error:
                _logger.warning(
                    "task metric projection skipped: graph=%s error=%s",
                    graph_id,
                    type(error).__name__,
                )
            finally:
                self._active.pop(key)
                completed.set()
            return

    async def close(self) -> None:
        self._accepting = False
        for completed in tuple(self._active.values()):
            await completed.wait()

    async def _project(self, graph_id: str, *, tenant_id: str) -> bool:
        try:
            projection = await asyncio.wait_for(
                self._collect(graph_id, tenant_id=tenant_id),
                _PROJECTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "task metric fact collection timed out: graph=%s",
                graph_id,
            )
            return False
        if projection is None:
            return False
        retry_count = sum(
            1
            for attempt in projection.attempts.values()
            if not attempt.invalid
            and attempt.terminal is not None
            and attempt.attempt_index > 1
        )
        # Source I/O has its own timeout; graph size must not consume a shared
        # write deadline after durable facts have been collected.
        accepted = await self._record_graph(
            projection.admission,
            projection.terminal,
            retry_count=retry_count,
            tenant_id=tenant_id,
            correlation=projection.correlation,
        )
        for (node_id, fence), attempt in projection.attempts.items():
            if attempt.invalid or attempt.terminal is None:
                continue
            accepted = (
                await self._record_attempt(
                    graph_id,
                    node_id,
                    fence,
                    attempt,
                    tenant_id=tenant_id,
                    correlation=projection.correlation,
                )
                and accepted
            )
        return accepted

    async def _collect(
        self, graph_id: str, *, tenant_id: str,
    ) -> _Projection | None:
        terminal = await self._repository.latest_event(
            graph_id,
            tenant_id=tenant_id,
        )
        if (
            terminal is None
            or terminal.graph_id != graph_id
            or terminal.node_id is not None
            or terminal.status not in _TERMINAL
        ):
            return None
        correlation: CorrelationData = {}
        if self._admissions is not None:
            admitted = await self._admissions.get(graph_id, tenant_id=tenant_id)
            if admitted is None:
                raise ValueError("task metric admission is missing")
            correlation = dict(admitted.correlation)

        admission: TaskEvent | None = None
        attempts: dict[tuple[str, int], _Attempt] = {}
        ready_at: dict[str, datetime] = {}
        attempt_counts: dict[str, int] = {}
        cursor = 0
        event_count = 0
        while True:
            page = await self._repository.list_events(
                graph_id,
                tenant_id=tenant_id,
                after_event_seq=cursor,
                limit=_EVENT_PAGE_SIZE,
            )
            if not page.items:
                if page.next_cursor is not None:
                    raise ValueError("task event page cursor is invalid")
                break
            for event in page.items:
                if event.graph_id != graph_id or event.event_seq != cursor + 1:
                    raise ValueError("task event sequence is invalid")
                cursor = event.event_seq
                event_count += 1
                if event_count > _MAX_PROJECTION_EVENTS:
                    _logger.warning(
                        "task node metric projection limit exceeded: graph=%s events=%s",
                        graph_id,
                        event_count,
                    )
                    return None
                if event.event_seq == 1:
                    if (
                        event.event_type is not TaskEventType.GRAPH_ADMITTED
                        or event.node_id is not None
                    ):
                        raise ValueError("task event admission is invalid")
                    admission = event
                self._consume_node_event(
                    attempts,
                    ready_at,
                    attempt_counts,
                    event,
                )
            if page.next_cursor is None:
                break

        if admission is None:
            raise ValueError("task event admission is missing")
        return _Projection(admission, terminal, attempts, correlation)

    @staticmethod
    def _consume_node_event(
        attempts: dict[tuple[str, int], _Attempt],
        ready_at: dict[str, datetime],
        attempt_counts: dict[str, int],
        event: TaskEvent,
    ) -> None:
        if event.event_type is not TaskEventType.NODE_CHANGED or event.node_id is None:
            return
        if event.status is TaskStatus.READY:
            ready_at[event.node_id] = event.occurred_at
            return
        if event.fence < 1:
            return
        key = (event.node_id, event.fence)
        if event.status is TaskStatus.RUNNING:
            attempt = attempts.get(key)
            if attempt is None:
                attempt_index = attempt_counts.get(event.node_id, 0) + 1
                attempt_counts[event.node_id] = attempt_index
                attempts[key] = _Attempt(
                    event.execution_id,
                    event.occurred_at,
                    attempt_index,
                    ready_at.get(event.node_id),
                )
                return
            if (
                event.execution_id is not None
                and attempt.execution_id is not None
                and event.execution_id != attempt.execution_id
            ):
                attempt.invalid = True
            elif attempt.execution_id is None:
                attempt.execution_id = event.execution_id
            return
        if event.status not in _TERMINAL:
            return
        attempt = attempts.get(key)
        if attempt is None:
            return
        if (
            event.execution_id is not None
            and attempt.execution_id is not None
            and event.execution_id != attempt.execution_id
        ):
            attempt.invalid = True
            return
        if attempt.execution_id is None:
            attempt.execution_id = event.execution_id
        if attempt.terminal is not None:
            attempt.invalid = True
            return
        attempt.terminal = event

    async def _record_graph(
        self,
        admission: TaskEvent,
        terminal: TaskEvent,
        *,
        retry_count: int,
        tenant_id: str,
        correlation: CorrelationData,
    ) -> bool:
        measurements = [MetricMeasurement("retry_count", 1, retry_count)]
        latency = _duration_ns(admission.occurred_at, terminal.occurred_at)
        if latency is not None:
            measurements.insert(0, MetricMeasurement("latency_ns", 1, latency))
        return await self._safe_record(
            lambda: Observation(
                version=1,
                observation_id=canonical_sha256(
                    {
                        "contract": "linktools.task.graph.terminal.v1",
                        "source_namespace": self._source_namespace,
                        "tenant_id": tenant_id,
                        "graph_id": terminal.graph_id,
                    }
                ),
                kind="linktools.task.graph.terminal",
                occurred_at=terminal.occurred_at,
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=terminal.status.value,
                error_code=terminal.error_code,
                correlation=_task_correlation(correlation, graph_id=terminal.graph_id),
                dimensions={},
                measurements=tuple(measurements),
            )
        )

    async def _record_attempt(
        self,
        graph_id: str,
        node_id: str,
        fence: int,
        attempt: _Attempt,
        *,
        tenant_id: str,
        correlation: CorrelationData,
    ) -> bool:
        terminal = cast(TaskEvent, attempt.terminal)
        correlation = _task_correlation(
            correlation,
            graph_id=graph_id,
            node_id=node_id,
            fence=fence,
            execution_id=attempt.execution_id,
            attempt_index=attempt.attempt_index,
        )
        measurements = [
            MetricMeasurement(
                "retry_count",
                1,
                1 if attempt.attempt_index > 1 else 0,
            )
        ]
        latency = _duration_ns(attempt.started_at, terminal.occurred_at)
        if latency is not None:
            measurements.insert(0, MetricMeasurement("latency_ns", 1, latency))
        if attempt.ready_at is not None:
            queue_wait = _duration_ns(attempt.ready_at, attempt.started_at)
            if queue_wait is not None:
                measurements.append(
                    MetricMeasurement("queue_wait_ns", 1, queue_wait)
                )
        return await self._safe_record(
            lambda: Observation(
                version=1,
                observation_id=canonical_sha256(
                    {
                        "contract": "linktools.task.node.attempt.v1",
                        "source_namespace": self._source_namespace,
                        "tenant_id": tenant_id,
                        "graph_id": graph_id,
                        "node_id": node_id,
                        "fence": fence,
                    }
                ),
                kind="linktools.task.node.attempt",
                occurred_at=terminal.occurred_at,
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=terminal.status.value,
                error_code=terminal.error_code,
                correlation=correlation,
                dimensions={},
                measurements=tuple(measurements),
            )
        )

    async def _safe_record(self, factory: Callable[[], Observation]) -> bool:
        try:
            return await self._recorder.try_record(factory())
        except Exception as error:
            _logger.warning(
                "task metric observation rejected: error=%s", type(error).__name__,
            )
            return False


def _duration_ns(start: datetime, end: datetime) -> int | None:
    delta = end - start
    if delta.total_seconds() < 0:
        _logger.warning("task metric negative duration skipped")
        return None
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    ) * 1_000


def _task_correlation(
    correlation: CorrelationData,
    **system: str | int | None,
) -> dict[str, str | int]:
    correlation: dict[str, str | int] = dict(correlation)
    for key, value in system.items():
        if value is not None:
            correlation[f"linktools.{key}"] = value
    return correlation


__all__: list[str] = []
