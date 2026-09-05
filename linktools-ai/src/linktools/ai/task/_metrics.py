#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-owned best-effort projection from durable event history into Metrics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, cast

from linktools.core import environ

from ..core import Page, TaskStatus, canonical_sha256
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._event import TaskEvent, TaskEventType

_logger = environ.get_logger("ai.task.metrics")
_EVENT_PAGE_SIZE = 1000
_MAX_PROJECTION_EVENTS = 100_000
_PROJECTION_TIMEOUT_SECONDS = 5.0
_DRAIN_TIMEOUT_SECONDS = 5.0
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
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]: ...

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None: ...


@dataclass(slots=True)
class _Attempt:
    start: TaskEvent
    execution_id: str | None
    invalid: bool = False
    terminal: TaskEvent | None = None


class _TaskMetricProjector:
    def __init__(
        self,
        repository: _TaskMetricRepository,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
    ) -> None:
        self._repository = repository
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._accepting = True

    def trigger(self, graph_id: str, *, tenant_id: str) -> None:
        if not self._accepting:
            return
        key = (tenant_id, graph_id)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._project_bounded(graph_id, tenant_id=tenant_id),
            name=f"task-metric-project-{tenant_id}-{graph_id}",
        )
        self._tasks[key] = task

        def consume(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except BaseException:  # noqa: BLE001
                _logger.exception("task metric projection failed: graph=%s", graph_id)
            finally:
                if self._tasks.get(key) is done:
                    self._tasks.pop(key, None)

        task.add_done_callback(consume)

    async def close(self) -> None:
        self._accepting = False
        pending = tuple(task for task in self._tasks.values() if not task.done())
        if not pending:
            await asyncio.sleep(0)
            return
        gather = asyncio.gather(*pending, return_exceptions=True)
        try:
            await asyncio.wait_for(asyncio.shield(gather), _DRAIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

    async def _project_bounded(self, graph_id: str, *, tenant_id: str) -> None:
        try:
            await asyncio.wait_for(
                self._project(graph_id, tenant_id=tenant_id),
                _PROJECTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            _logger.warning("task metric projection timed out: graph=%s", graph_id)
        except Exception as error:
            _logger.warning(
                "task metric projection skipped: graph=%s error=%s",
                graph_id,
                type(error).__name__,
            )

    async def _project(self, graph_id: str, *, tenant_id: str) -> None:
        first_page = await self._repository.list_events(
            graph_id,
            tenant_id=tenant_id,
            after_sequence=0,
            limit=1,
        )
        if len(first_page.items) != 1:
            raise ValueError("task event admission is missing")
        admission = first_page.items[0]
        if (
            admission.sequence != 1
            or admission.event_type is not TaskEventType.GRAPH_ADMITTED
            or admission.graph_id != graph_id
            or admission.node_id is not None
        ):
            raise ValueError("task event admission is invalid")
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
            return
        self._record_graph(admission, terminal, tenant_id=tenant_id)

        attempts: dict[tuple[str, int], _Attempt] = {}
        cursor = 0
        event_count = 0
        while True:
            page = await self._repository.list_events(
                graph_id,
                tenant_id=tenant_id,
                after_sequence=cursor,
                limit=_EVENT_PAGE_SIZE,
            )
            if not page.items:
                if page.next_cursor is not None:
                    raise ValueError("task event page cursor is invalid")
                break
            for event in page.items:
                if event.graph_id != graph_id or event.sequence != cursor + 1:
                    raise ValueError("task event sequence is invalid")
                cursor = event.sequence
                event_count += 1
                if event_count > _MAX_PROJECTION_EVENTS:
                    _logger.warning(
                        "task node metric projection limit exceeded: graph=%s events=%s",
                        graph_id,
                        event_count,
                    )
                    return
                self._consume_node_event(attempts, event)
            if page.next_cursor is None:
                break

        for (node_id, fence), attempt in attempts.items():
            if attempt.invalid or attempt.terminal is None:
                continue
            self._record_attempt(
                graph_id,
                node_id,
                fence,
                attempt,
                tenant_id=tenant_id,
            )

    @staticmethod
    def _consume_node_event(
        attempts: dict[tuple[str, int], _Attempt],
        event: TaskEvent,
    ) -> None:
        if event.event_type is not TaskEventType.NODE_CHANGED or event.node_id is None:
            return
        if event.fence < 1:
            return
        key = (event.node_id, event.fence)
        if event.status is TaskStatus.RUNNING:
            attempt = attempts.get(key)
            if attempt is None:
                attempts[key] = _Attempt(event, event.execution_id)
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

    def _record_graph(
        self,
        admission: TaskEvent,
        terminal: TaskEvent,
        *,
        tenant_id: str,
    ) -> None:
        measurements: tuple[MetricMeasurement, ...] = ()
        latency = _latency_ns(admission, terminal)
        if latency is not None:
            measurements = (MetricMeasurement("latency_ns", 1, latency),)
        self._safe_record(
            Observation(
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
                error_code=None,
                correlation={"graph_id": terminal.graph_id},
                dimensions={},
                measurements=measurements,
            )
        )

    def _record_attempt(
        self,
        graph_id: str,
        node_id: str,
        fence: int,
        attempt: _Attempt,
        *,
        tenant_id: str,
    ) -> None:
        terminal = cast(TaskEvent, attempt.terminal)
        correlation: dict[str, str | int] = {
            "graph_id": graph_id,
            "node_id": node_id,
            "fence": fence,
        }
        if attempt.execution_id is not None:
            correlation["execution_id"] = attempt.execution_id
        measurements: tuple[MetricMeasurement, ...] = ()
        latency = _latency_ns(attempt.start, terminal)
        if latency is not None:
            measurements = (MetricMeasurement("latency_ns", 1, latency),)
        self._safe_record(
            Observation(
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
                measurements=measurements,
            )
        )

    def _safe_record(self, observation: Observation) -> None:
        try:
            self._recorder.try_record(observation)
        except Exception:
            _logger.exception("task metric observation rejected")


def _latency_ns(start: TaskEvent, terminal: TaskEvent) -> int | None:
    delta = terminal.occurred_at - start.occurred_at
    if delta.total_seconds() < 0:
        _logger.warning(
            "task metric negative latency skipped: graph=%s start=%s terminal=%s",
            terminal.graph_id,
            start.sequence,
            terminal.sequence,
        )
        return None
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    ) * 1_000


__all__: list[str] = []
