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

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Metrics, Observation
from ._execution import _ExecutionTerminalCommitter
from .state._contracts import ExecutionTerminalCommit, ExecutionTerminalCommitResult

_logger = environ.get_logger("ai.runtime.metrics")
_QUEUE_CAPACITY = 1024
_BATCH_SIZE = 64
_WRITE_TIMEOUT_SECONDS = 2.0
_WRITE_ATTEMPTS = 2
_CLOSE_DEADLINE_SECONDS = 5.0
_WARNING_INTERVAL_SECONDS = 30.0


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
        if not isinstance(observation, Observation):
            self._drop("runtime metric observation invalid")
            return False
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
            self._drop_remaining("runtime metric close cleanup")
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
        self._drop_remaining("runtime metric close cleanup")
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
            stop_after_batch = False
            while len(batch) < _BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                await self._write(tuple(batch))
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop_after_batch:
                return

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
