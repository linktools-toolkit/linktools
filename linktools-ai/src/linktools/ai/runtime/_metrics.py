#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned best-effort Metrics buffering and observation helpers."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from time import monotonic

from linktools.core import environ

from ..core import JsonValue, TaskStatus, canonical_sha256
from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Metrics, Observation
from ..storage import StoredPayload
from ..task import (
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
    TaskTerminalRecord,
)
from ._execution import _ExecutionTerminalCommitter
from .state._contracts import (
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    TaskRepository,
)

_logger = environ.get_logger("ai.runtime.metrics")
_QUEUE_CAPACITY = 1024
_BATCH_SIZE = 64
_WRITE_TIMEOUT_SECONDS = 2.0
_WRITE_ATTEMPTS = 2
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


class _RuntimeMetricBuffer(MetricRecorder):
    """Bound one Runtime's automatic observations without owning the Metrics store."""

    def __init__(self, metrics: Metrics) -> None:
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
        if not self._accepting:
            self._drop("runtime metric buffer closed")
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
            _logger.debug("runtime metrics closed: accepted=%s", self._accepted)

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
        for attempt in range(_WRITE_ATTEMPTS):
            try:
                await asyncio.wait_for(
                    self._metrics.record_observations(batch),
                    timeout=_WRITE_TIMEOUT_SECONDS,
                )
                return
            except asyncio.CancelledError:
                raise
            except (TimeoutError, asyncio.TimeoutError):
                if attempt + 1 < _WRITE_ATTEMPTS:
                    continue
            except AIError as error:
                if attempt + 1 < _WRITE_ATTEMPTS and (
                    error.retryable
                    or error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                ):
                    continue
            except Exception:
                pass
            break
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


class _MetricExecutionTerminalCommitter:
    """Project service-owned durable terminal checkpoints into Metrics."""

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
        _record_execution_terminal(
            self._recorder,
            source_namespace=self._source_namespace,
            result=result,
            session_id=session_id,
        )
        return result


class _MetricTaskRepository:
    """Project Task terminal mutations without observing ordinary reads."""

    def __init__(
        self,
        delegate: TaskRepository,
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
        await self._observe_graph(view, tenant_id=tenant_id)
        return view

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None:
        return await self._delegate.get_graph(graph_id, tenant_id=tenant_id)

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]:
        return await self._delegate.list_nodes(graph_id, tenant_id=tenant_id)

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
    ) -> TaskTerminalRecord:
        terminal = await self._delegate.complete(
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
            result_digest=result_digest,
            result_payload=result_payload,
        )
        self._record_attempt(lease, terminal, tenant_id=tenant_id)
        return terminal

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> TaskTerminalRecord:
        terminal = await self._delegate.fail(
            lease,
            tenant_id=tenant_id,
            error_code=error_code,
            error_digest=error_digest,
            execution_id=execution_id,
        )
        self._record_attempt(lease, terminal, tenant_id=tenant_id)
        return terminal

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self._delegate.cancel_graph(graph_id, tenant_id=tenant_id)
        await self._observe_graph(view, tenant_id=tenant_id)
        return view

    async def _observe_graph(self, view: TaskGraphView, *, tenant_id: str) -> None:
        graph_key = (tenant_id, view.graph_id)
        for node in view.nodes:
            task_type = _task_type(node)
            if task_type is not None:
                self._task_types[(*graph_key, node.node_id)] = task_type
        if view.status not in _TASK_TERMINAL:
            return
        try:
            event = await self._delegate.latest_event(
                view.graph_id,
                tenant_id=tenant_id,
            )
        except Exception as error:
            _logger.warning(
                "task graph metric projection skipped: graph=%s error=%s",
                view.graph_id,
                type(error).__name__,
            )
            event = None
        if (
            event is not None
            and event.graph_id == view.graph_id
            and event.status is view.status
            and event.node_id is None
        ):
            _try_record(
                self._recorder,
                lambda: _observation(
                    observation_id=_stable_observation_id(
                        self._source_namespace,
                        tenant_id,
                        view.graph_id,
                        str(event.sequence),
                        "terminal",
                    ),
                    kind="linktools.task.graph.terminal",
                    source_namespace=self._source_namespace,
                    tenant_id=tenant_id,
                    status=view.status.value,
                    error_code=None,
                    correlation={"graph_id": view.graph_id},
                    occurred_at=event.occurred_at,
                ),
            )
        for key in tuple(self._task_types):
            if key[:2] == graph_key:
                self._task_types.pop(key, None)

    def _record_attempt(
        self,
        lease: TaskLease,
        terminal: TaskTerminalRecord,
        *,
        tenant_id: str,
    ) -> None:
        correlation: dict[str, str | int] = {
            "graph_id": lease.graph_id,
            "node_id": terminal.node_id,
            "fence": terminal.fence,
        }
        if terminal.execution_id is not None:
            correlation["execution_id"] = terminal.execution_id
        task_type = self._task_types.get(
            (tenant_id, lease.graph_id, terminal.node_id)
        )
        _try_record(
            self._recorder,
            lambda: _observation(
                observation_id=_stable_observation_id(
                    self._source_namespace,
                    tenant_id,
                    lease.graph_id,
                    terminal.node_id,
                    str(terminal.fence),
                    "attempt",
                ),
                kind="linktools.task.node.attempt",
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=terminal.status.value,
                error_code=terminal.error_code,
                correlation=correlation,
                dimensions={} if task_type is None else {"task_type": task_type},
                occurred_at=terminal.completed_at,
            ),
        )


def _record_execution_terminal(
    recorder: MetricRecorder | None,
    *,
    source_namespace: str,
    result: ExecutionTerminalCommitResult,
    session_id: str | None,
) -> None:
    if recorder is None:
        return
    execution = result.execution
    usage = result.result.usage
    correlation: dict[str, str | int] = {"execution_id": execution.execution_id}
    if session_id is not None:
        correlation["session_id"] = session_id
    _try_record(
        recorder,
        lambda: _observation(
            observation_id=_stable_observation_id(
                source_namespace,
                execution.tenant_id,
                execution.execution_id,
                "terminal",
            ),
            kind="linktools.execution.terminal",
            source_namespace=source_namespace,
            tenant_id=execution.tenant_id,
            status=execution.status.value,
            error_code=execution.error_code,
            correlation=correlation,
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


def _record_storage_operation(
    recorder: MetricRecorder | None,
    *,
    source_namespace: str,
    tenant_id: str,
    domain: str,
    target: str,
    status: str,
    error_code: str | None = None,
    latency_ns: int | None = None,
) -> None:
    if recorder is None:
        return
    measurements = (
        ()
        if latency_ns is None
        else (_measurement("latency_ns", latency_ns),)
    )
    _try_record(
        recorder,
        lambda: _observation(
            observation_id=uuid.uuid4().hex,
            kind="linktools.storage.operation",
            source_namespace=source_namespace,
            tenant_id=tenant_id,
            status=status,
            error_code=error_code,
            dimensions={"domain": domain, "target": target},
            measurements=measurements,
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
        _logger.exception("runtime metric observation rejected")
        return False


__all__: list[str] = []
