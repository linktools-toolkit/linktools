#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned best-effort Metrics buffering and observation helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from time import monotonic
from typing import Protocol

from linktools.core import environ

from ..core import JsonValue, TaskStatus, canonical_sha256
from ..errors import AIError
from ..observe import MetricMeasurement, MetricRecorder, Metrics, Observation
from ..storage import StoredPayload
from ..task import TaskGraphView, TaskLease, TaskNode, TaskNodeView, TaskResultRecord
from .state._contracts import (
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
)

_logger = environ.get_logger("ai.runtime.metrics")
_QUEUE_CAPACITY = 1024
_BATCH_SIZE = 64
_WRITE_TIMEOUT_SECONDS = 2.0
_CLOSE_DEADLINE_SECONDS = 5.0
_WARNING_INTERVAL_SECONDS = 30.0
_TASK_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)
_TASK_ATTEMPT_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)


class _RuntimeMetricBuffer(MetricRecorder):
    """Bound one Runtime's automatic observations without owning the Metrics store."""

    def __init__(self, metrics: Metrics) -> None:
        if not isinstance(metrics, Metrics):
            raise TypeError("metrics must be Metrics")
        self._metrics = metrics
        self._queue: asyncio.Queue[Observation | None] = asyncio.Queue(
            maxsize=_QUEUE_CAPACITY
        )
        self._accepting = True
        self._accepted = 0
        self._dropped = 0
        self._write_failures = 0
        self._last_warning_at = 0.0
        self._writer = asyncio.create_task(
            self._run(),
            name="linktools-runtime-metrics",
        )

    def try_record(self, observation: Observation) -> bool:
        if not self._accepting or not isinstance(observation, Observation):
            self._drop("invalid or closed metric observation")
            return False
        try:
            self._queue.put_nowait(observation)
        except asyncio.QueueFull:
            self._drop("runtime metric queue full")
            return False
        self._accepted += 1
        return True

    async def close(self) -> None:
        if not self._accepting and self._writer.done():
            self._consume_writer()
            return
        self._accepting = False
        started = monotonic()
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=_CLOSE_DEADLINE_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError):
            self._drop_remaining("runtime metric close deadline exceeded")
        remaining = max(0.0, _CLOSE_DEADLINE_SECONDS - (monotonic() - started))
        if not self._writer.done():
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                self._drop_remaining("runtime metric close queue remained full")
                self._queue.put_nowait(None)
            try:
                await asyncio.wait_for(asyncio.shield(self._writer), timeout=remaining)
            except (TimeoutError, asyncio.TimeoutError):
                self._writer.cancel()
                await self._consume_cancelled_writer()
        self._consume_writer()
        if self._dropped or self._write_failures:
            _logger.warning(
                "runtime metrics closed with loss: accepted=%s dropped=%s write_failures=%s",
                self._accepted,
                self._dropped,
                self._write_failures,
            )
        else:
            _logger.debug(
                "runtime metrics closed: accepted=%s",
                self._accepted,
            )

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            while len(batch) < _BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._write(tuple(batch))
                    for _ in batch:
                        self._queue.task_done()
                    return
                batch.append(item)
            await self._write(tuple(batch))
            for _ in batch:
                self._queue.task_done()

    async def _write(self, batch: tuple[Observation, ...]) -> None:
        try:
            await asyncio.wait_for(
                self._metrics.record_observations(batch),
                timeout=_WRITE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._write_failures += 1
            self._dropped += len(batch)
            self._warn("runtime metric batch write failed")

    def _drop(self, message: str) -> None:
        self._dropped += 1
        self._warn(message)

    def _drop_remaining(self, message: str) -> None:
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if item is not None:
                dropped += 1
        if dropped:
            self._dropped += dropped
            self._warn(message)

    def _warn(self, message: str) -> None:
        now = monotonic()
        if now - self._last_warning_at < _WARNING_INTERVAL_SECONDS:
            return
        self._last_warning_at = now
        _logger.warning(
            "%s: accepted=%s dropped=%s write_failures=%s",
            message,
            self._accepted,
            self._dropped,
            self._write_failures,
        )

    async def _consume_cancelled_writer(self) -> None:
        try:
            await self._writer
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._warn("runtime metric writer failed during cancellation")

    def _consume_writer(self) -> None:
        if not self._writer.done():
            return
        try:
            self._writer.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._warn("runtime metric writer failed")


class _ExecutionTerminalCommitter(Protocol):
    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
    ) -> ExecutionTerminalCommitResult: ...


class _MetricExecutionTerminalCommitter:
    """Project durable terminal truth into best-effort execution observations."""

    def __init__(
        self,
        delegate: _ExecutionTerminalCommitter,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._source_namespace = source_namespace

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None,
    ) -> ExecutionTerminalCommitResult:
        result = await self._delegate.commit_terminal_checkpoint(
            commit,
            session_id=session_id,
        )
        execution = result.execution
        usage = result.result.usage
        _try_record(
            self._recorder,
            lambda: _observation(
                observation_id=_stable_observation_id(
                    self._source_namespace,
                    execution.tenant_id,
                    execution.execution_id,
                    "terminal",
                ),
                kind="linktools.execution.terminal",
                source_namespace=self._source_namespace,
                tenant_id=execution.tenant_id,
                status=execution.status.value,
                error_code=execution.error_code,
                correlation={
                    "execution_id": execution.execution_id,
                    **(
                        {"session_id": session_id}
                        if session_id is not None
                        else {}
                    ),
                },
                dimensions={
                    "agent_id": execution.binding.agent_spec.id,
                    "mode": execution.mode,
                },
                measurements=(
                    _measurement("input_tokens", usage.input_tokens),
                    _measurement("output_tokens", usage.output_tokens),
                ),
                occurred_at=result.result.created_at,
            ),
        )
        return result


class _TaskRepository(Protocol):
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None: ...

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]: ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]: ...

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView: ...

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> object: ...

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> object: ...

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...


class _MetricTaskRepository:
    """Project Task durable terminal operations without changing repository semantics."""

    def __init__(
        self,
        delegate: _TaskRepository,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._task_types: dict[tuple[str, str, str], str] = {}

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self._delegate.reconcile_graph(graph_id, tenant_id=tenant_id)
        self._observe_graph(view, tenant_id=tenant_id)
        return view

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None:
        view = await self._delegate.get_graph(graph_id, tenant_id=tenant_id)
        if view is not None:
            self._observe_graph(view, tenant_id=tenant_id)
        return view

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]:
        states = await self._delegate.list_nodes(graph_id, tenant_id=tenant_id)
        for state in states:
            self._record_attempt_state(state, tenant_id=tenant_id)
        return states

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]:
        return await self._delegate.get_results(
            graph_id,
            node_ids,
            tenant_id=tenant_id,
        )

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        return await self._delegate.claim(
            graph_id,
            node_id,
            tenant_id=tenant_id,
            owner=owner,
            lease_seconds=lease_seconds,
        )

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease:
        return await self._delegate.renew(
            lease,
            tenant_id=tenant_id,
            lease_seconds=lease_seconds,
        )

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        return await self._delegate.bind_execution(
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> object:
        result = await self._delegate.complete(
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
            result_digest=result_digest,
            result_payload=result_payload,
        )
        self._record_attempt(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.SUCCEEDED,
            error_code=None,
            execution_id=execution_id,
        )
        return result

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> object:
        result = await self._delegate.fail(
            lease,
            tenant_id=tenant_id,
            error_code=error_code,
            error_digest=error_digest,
            execution_id=execution_id,
        )
        self._record_attempt(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.FAILED,
            error_code=error_code,
            execution_id=execution_id,
        )
        return result

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self._delegate.cancel_graph(graph_id, tenant_id=tenant_id)
        self._observe_graph(view, tenant_id=tenant_id)
        return view

    def _observe_graph(self, view: TaskGraphView, *, tenant_id: str) -> None:
        for node in view.nodes:
            task_type = _task_type(node)
            if task_type is not None:
                self._task_types[(tenant_id, view.graph_id, node.node_id)] = task_type
        if view.status not in _TASK_TERMINAL:
            return
        _try_record(
            self._recorder,
            lambda: _observation(
                observation_id=_stable_observation_id(
                    self._source_namespace,
                    tenant_id,
                    view.graph_id,
                    "terminal",
                ),
                kind="linktools.task.graph.terminal",
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=view.status.value,
                error_code=None,
                correlation={"graph_id": view.graph_id},
            ),
        )

    def _record_attempt_state(self, state: TaskNodeView, *, tenant_id: str) -> None:
        if state.status not in _TASK_ATTEMPT_TERMINAL or state.fence < 1:
            return
        self._record_attempt_values(
            graph_id=state.graph_id,
            node_id=state.node_id,
            fence=state.fence,
            tenant_id=tenant_id,
            status=state.status,
            error_code=state.error_code,
            execution_id=state.execution_id,
        )

    def _record_attempt(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        status: TaskStatus,
        error_code: str | None,
        execution_id: str | None,
    ) -> None:
        self._record_attempt_values(
            graph_id=lease.graph_id,
            node_id=lease.node_id,
            fence=lease.fence,
            tenant_id=tenant_id,
            status=status,
            error_code=error_code,
            execution_id=execution_id,
        )

    def _record_attempt_values(
        self,
        *,
        graph_id: str,
        node_id: str,
        fence: int,
        tenant_id: str,
        status: TaskStatus,
        error_code: str | None,
        execution_id: str | None,
    ) -> None:
        correlation: dict[str, str | int] = {
            "graph_id": graph_id,
            "node_id": node_id,
            "fence": fence,
        }
        if execution_id is not None:
            correlation["execution_id"] = execution_id
        task_type = self._task_types.get((tenant_id, graph_id, node_id))
        _try_record(
            self._recorder,
            lambda: _observation(
                observation_id=_stable_observation_id(
                    self._source_namespace,
                    tenant_id,
                    graph_id,
                    node_id,
                    str(fence),
                    "attempt",
                ),
                kind="linktools.task.node.attempt",
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=status.value,
                error_code=error_code,
                correlation=correlation,
                dimensions={} if task_type is None else {"task_type": task_type},
            ),
        )


def _task_type(node: TaskNode) -> str | None:
    value: JsonValue | None = node.input.get("type")
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


def _stable_observation_id(*parts: str) -> str:
    return canonical_sha256({"parts": list(parts)})


def _measurement(name: str, value: int | float) -> MetricMeasurement:
    return MetricMeasurement(name, 1, value)


def _observation(
    *,
    observation_id: str,
    kind: str,
    source_namespace: str,
    tenant_id: str,
    status: str | None,
    error_code: str | None,
    correlation: Mapping[str, str | int] | None = None,
    dimensions: Mapping[str, str] | None = None,
    measurements: tuple[MetricMeasurement, ...] = (),
    occurred_at: datetime | None = None,
) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind=kind,
        occurred_at=occurred_at or datetime.now(timezone.utc),
        source_namespace=source_namespace,
        tenant_id=tenant_id,
        status=status,
        error_code=error_code,
        correlation=dict(correlation or {}),
        dimensions=dict(dimensions or {}),
        measurements=measurements,
    )


def _try_record(
    recorder: MetricRecorder | None,
    factory: Callable[[], Observation],
) -> bool:
    if recorder is None:
        return False
    try:
        return recorder.try_record(factory())
    except (AIError, TypeError, ValueError):
        _logger.warning("runtime metric observation rejected", exc_info=False)
        return False


__all__: list[str] = []
