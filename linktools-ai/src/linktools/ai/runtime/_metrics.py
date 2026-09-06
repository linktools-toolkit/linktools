#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned best-effort Metrics buffering and observation helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic, monotonic_ns

from linktools.core import environ

from ..core import CorrelationData, UsageMetrics, normalize_correlation
from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Metrics, Observation
from ._metric_id import _stable_observation_id
from .state._contracts import ExecutionTerminalCommitResult

_logger = environ.get_logger("ai.runtime.metrics")
_QUEUE_CAPACITY = 1024
_BATCH_SIZE = 128
_FLUSH_INTERVAL_SECONDS = 0.1
_WRITE_TIMEOUT_SECONDS = 5.0
_CLOSE_DEADLINE_SECONDS = 10.0
_WARNING_INTERVAL_SECONDS = 30.0
_FRAMEWORK_CORRELATION_KEYS = frozenset(
    {
        "execution_id",
        "session_id",
        "step_run_id",
        "tool_call_id",
        "parent_execution_id",
        "root_execution_id",
        "graph_id",
        "node_id",
        "fence",
        "attempt_index",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeMetricStatus:
    enabled: bool
    accepting: bool
    accepted: int
    persisted: int
    pending: int
    rejected: int
    lost: int
    write_failures: int
    queue_size: int = 0
    queue_capacity: int = 0
    high_watermark: int = 0
    last_failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeMetricFlushResult:
    completed: bool
    status: RuntimeMetricStatus


def _disabled_metric_status() -> RuntimeMetricStatus:
    return RuntimeMetricStatus(False, False, 0, 0, 0, 0, 0, 0)


def _metric_correlation(
    correlation: CorrelationData | Mapping[str, object] | None,
    **system: str | int | None,
) -> dict[str, str | int]:
    values: dict[str, str | int] = dict(normalize_correlation(correlation))
    for key, value in system.items():
        if value is not None:
            values[f"linktools.{key}"] = value
    return values


class _RuntimeMetricBuffer(MetricRecorder):
    """Bound one Runtime's automatic observations without owning the Metrics store."""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._queue: asyncio.Queue[Observation | None] = asyncio.Queue(
            maxsize=_QUEUE_CAPACITY
        )
        self._accepting = True
        self._accepted = 0
        self._persisted = 0
        self._rejected = 0
        self._lost = 0
        self._write_failures = 0
        self._high_watermark = 0
        self._last_failure_code: str | None = None
        self._last_warning_at = 0.0
        self._writer: asyncio.Task[None] | None = None
        self._resolution_event = asyncio.Event()
        self._execution_contexts: dict[str, CorrelationData] = {}
        self._runtime_dimensions: dict[str, str] = {}
        self._runtime_dimensions_configured = False

    def configure_runtime_dimensions(self, dimensions: Mapping[str, str]) -> None:
        normalized = {f"context.{key}": value for key, value in dimensions.items()}
        if self._runtime_dimensions_configured:
            if self._runtime_dimensions != normalized:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return
        if self._accepted or self._writer is not None or self._execution_contexts:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._runtime_dimensions = normalized
        self._runtime_dimensions_configured = True

    def bind_execution_context(
        self,
        execution_id: str,
        correlation: CorrelationData | Mapping[str, object],
    ) -> bool:
        try:
            normalized = normalize_correlation(correlation)
        except (TypeError, ValueError):
            self._set_failure("EXECUTION_CONTEXT_INVALID")
            self._warn("runtime metric execution correlation invalid")
            return False
        current = self._execution_contexts.get(execution_id)
        if current is not None and dict(current) != dict(normalized):
            self._set_failure("EXECUTION_CONTEXT_CONFLICT")
            self._warn("runtime metric execution correlation conflict")
            return False
        self._execution_contexts[execution_id] = normalized
        return True

    def release_execution_context(self, execution_id: str) -> None:
        self._execution_contexts.pop(execution_id, None)

    def status(self) -> RuntimeMetricStatus:
        pending = self._accepted - self._persisted - self._lost
        if pending < 0:
            raise RuntimeError("runtime metric counters are inconsistent")
        return RuntimeMetricStatus(
            True,
            self._accepting,
            self._accepted,
            self._persisted,
            pending,
            self._rejected,
            self._lost,
            self._write_failures,
            self._queue.qsize(),
            self._queue.maxsize,
            self._high_watermark,
            self._last_failure_code,
        )

    async def flush(self, *, timeout_seconds: float = 5.0) -> RuntimeMetricFlushResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        target = self._accepted
        if self._resolved >= target:
            return RuntimeMetricFlushResult(True, self.status())

        async def wait_resolved() -> None:
            while self._resolved < target:
                self._resolution_event.clear()
                if self._resolved >= target:
                    return
                await self._resolution_event.wait()

        try:
            await asyncio.wait_for(wait_resolved(), float(timeout_seconds))
        except asyncio.TimeoutError:
            return RuntimeMetricFlushResult(False, self.status())
        return RuntimeMetricFlushResult(True, self.status())

    @property
    def _resolved(self) -> int:
        return self._persisted + self._lost

    def try_record(self, observation: Observation) -> bool:
        if not isinstance(observation, Observation):
            self._reject("runtime metric observation invalid", "INVALID_OBSERVATION")
            return False
        if not self._accepting:
            self._reject("runtime metric buffer closed", "BUFFER_CLOSED")
            return False
        try:
            enriched = self._enrich_observation(observation)
        except (AIError, TypeError, ValueError):
            self._reject(
                "runtime metric observation enrichment failed",
                "ENRICHMENT_FAILED",
            )
            return False
        if not self._start_writer():
            return False
        try:
            self._queue.put_nowait(enriched)
        except asyncio.QueueFull:
            self._reject("runtime metric queue full", "QUEUE_FULL")
            return False
        self._accepted += 1
        self._high_watermark = max(self._high_watermark, self._queue.qsize())
        return True

    def _enrich_observation(self, observation: Observation) -> Observation:
        raw = dict(observation.correlation)
        execution_id = raw.get("linktools.execution_id", raw.get("execution_id"))
        context = (
            self._execution_contexts.get(execution_id)
            if isinstance(execution_id, str)
            else None
        )
        correlation: dict[str, str | int] = dict(context or {})
        for key, value in raw.items():
            normalized_key = (
                f"linktools.{key}" if key in _FRAMEWORK_CORRELATION_KEYS else key
            )
            existing = correlation.get(normalized_key)
            if existing is not None and existing != value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            correlation[normalized_key] = value

        raw_dimensions = dict(observation.dimensions)
        dimensions = dict(self._runtime_dimensions)
        for key, value in raw_dimensions.items():
            existing = dimensions.get(key)
            if existing is not None and existing != value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            dimensions[key] = value

        if correlation == raw and dimensions == raw_dimensions:
            return observation
        return replace(
            observation,
            correlation=correlation,
            dimensions=dimensions,
        )

    def _start_writer(self) -> bool:
        if self._writer is not None:
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._reject(
                "runtime metric writer requires a running event loop",
                "WRITER_UNAVAILABLE",
            )
            return False
        self._writer = loop.create_task(
            self._run(),
            name="linktools-runtime-metrics",
        )
        return True

    async def close(self) -> None:
        """Stop accepting, best-effort drain, and never fail Runtime shutdown."""
        try:
            await self._close_impl()
        except BaseException:  # noqa: BLE001
            self._set_failure("CLOSE_FAILED")
            _logger.exception("runtime metric close failed open")
            self._accepting = False
            writer = self._writer
            if writer is not None and not writer.done():
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
            self._lose_remaining("runtime metric close failed")
            self._execution_contexts.clear()
            self._log_close()

    async def _close_impl(self) -> None:
        self._accepting = False
        writer = self._writer
        if writer is None:
            self._execution_contexts.clear()
            self._log_close()
            return
        started = monotonic()
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=_CLOSE_DEADLINE_SECONDS,
            )
        except asyncio.TimeoutError:
            self._set_failure("CLOSE_TIMEOUT")
            self._lose_remaining("runtime metric close deadline exceeded")
        remaining = max(0.0, _CLOSE_DEADLINE_SECONDS - (monotonic() - started))
        if not writer.done():
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                self._set_failure("CLOSE_TIMEOUT")
                self._lose_remaining("runtime metric close queue remained full")
                self._queue.put_nowait(None)
            try:
                await asyncio.wait_for(asyncio.shield(writer), timeout=remaining)
            except asyncio.TimeoutError:
                self._set_failure("CLOSE_TIMEOUT")
                writer.cancel()
                await self._consume_cancelled_writer(writer)
        self._consume_writer(writer)
        self._lose_remaining("runtime metric close cleanup")
        self._execution_contexts.clear()
        self._log_close()

    def _log_close(self) -> None:
        if self._rejected or self._lost or self._write_failures:
            _logger.warning(
                "runtime metrics closed with loss: accepted=%s persisted=%s rejected=%s lost=%s write_failures=%s",
                self._accepted,
                self._persisted,
                self._rejected,
                self._lost,
                self._write_failures,
            )
        else:
            _logger.debug(
                "runtime metrics closed: accepted=%s persisted=%s",
                self._accepted,
                self._persisted,
            )

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            stop_after_batch = False
            deadline = asyncio.get_running_loop().time() + _FLUSH_INTERVAL_SECONDS
            while len(batch) < _BATCH_SIZE:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                await self._write(tuple(batch))
            except asyncio.CancelledError:
                self._lost += len(batch)
                self._resolution_event.set()
                raise
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop_after_batch:
                return

    async def _write(self, batch: tuple[Observation, ...]) -> None:
        try:
            await asyncio.wait_for(
                self._metrics.record_observations(batch),
                timeout=_WRITE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._write_failures += 1
            self._lost += len(batch)
            self._set_failure("WRITE_FAILED")
            self._resolution_event.set()
            self._warn("runtime metric batch write failed")
        else:
            self._persisted += len(batch)
            self._resolution_event.set()

    def _reject(self, message: str, code: str) -> None:
        self._rejected += 1
        self._set_failure(code)
        self._warn(message)

    def _set_failure(self, code: str) -> None:
        self._last_failure_code = code

    def _lose_remaining(self, message: str) -> None:
        lost = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if item is not None:
                lost += 1
        if lost:
            self._lost += lost
            self._resolution_event.set()
            self._warn(message)

    def _warn(self, message: str) -> None:
        now = monotonic()
        if now - self._last_warning_at < _WARNING_INTERVAL_SECONDS:
            return
        self._last_warning_at = now
        _logger.warning(
            "%s: accepted=%s persisted=%s rejected=%s lost=%s write_failures=%s",
            message,
            self._accepted,
            self._persisted,
            self._rejected,
            self._lost,
            self._write_failures,
        )

    async def _consume_cancelled_writer(self, writer: asyncio.Task[None]) -> None:
        try:
            await writer
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._set_failure("WRITER_FAILED")
            self._warn("runtime metric writer failed during cancellation")

    def _consume_writer(self, writer: asyncio.Task[None]) -> None:
        if not writer.done():
            return
        try:
            writer.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._set_failure("WRITER_FAILED")
            self._warn("runtime metric writer failed")


def _bind_metric_execution_context(
    recorder: MetricRecorder,
    execution_id: str,
    correlation: CorrelationData,
) -> None:
    if isinstance(recorder, _RuntimeMetricBuffer):
        recorder.bind_execution_context(execution_id, correlation)


def _release_metric_execution_context(
    recorder: MetricRecorder | None,
    execution_id: str,
) -> None:
    if not isinstance(recorder, _RuntimeMetricBuffer):
        return
    try:
        recorder.release_execution_context(execution_id)
    except Exception:
        _logger.exception("runtime metric execution context release failed")


def _execution_usage_measurements(
    usage: UsageMetrics,
) -> tuple[MetricMeasurement, ...]:
    return (
        _measurement("model_requests", usage.model_requests),
        _measurement("tool_calls", usage.tool_calls),
        _measurement("input_tokens", usage.input_tokens),
        _measurement("output_tokens", usage.output_tokens),
        _measurement("cache_read_tokens", usage.cache_read_tokens),
        _measurement("cache_write_tokens", usage.cache_write_tokens),
        _measurement("total_tokens", usage.total_tokens),
    )


def _execution_latency_ns(result: ExecutionTerminalCommitResult) -> int | None:
    delta = result.result.created_at - result.execution.created_at
    if delta.total_seconds() < 0:
        _logger.warning(
            "execution metric negative latency skipped: execution=%s",
            result.execution.execution_id,
        )
        return None
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    ) * 1_000


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
    latency = _execution_latency_ns(result)
    measurements = list(_execution_usage_measurements(result.result.usage))
    if latency is not None:
        measurements.insert(0, _measurement("latency_ns", latency))
    correlation = _metric_correlation(
        execution.correlation,
        execution_id=execution.execution_id,
        session_id=session_id,
        parent_execution_id=execution.parent_execution_id,
        root_execution_id=execution.root_execution_id,
    )
    _try_record(
        recorder,
        lambda: _observation(
            observation_id=_stable_observation_id(
                "linktools.execution.terminal.v1",
                source_namespace,
                execution.tenant_id,
                execution.execution_id,
            ),
            kind="linktools.execution.terminal",
            source_namespace=source_namespace,
            tenant_id=execution.tenant_id,
            status=execution.status.value,
            error_code=execution.error_code,
            correlation=correlation,
            dimensions={
                "agent_id": execution.binding.agent_spec.id,
                "lineage_kind": execution.lineage_kind.value,
            },
            measurements=tuple(measurements),
            occurred_at=result.result.created_at,
        ),
    )


def _record_storage_operation(
    recorder: MetricRecorder | None,
    *,
    observation_id: str | None,
    started_at_ns: int | None,
    source_namespace: str,
    tenant_id: str,
    execution_id: str,
    session_id: str | None,
    correlation: CorrelationData,
    status: str,
    error_code: str | None,
    domain: str,
    target: str,
) -> None:
    if recorder is None or observation_id is None or started_at_ns is None:
        return
    _try_record(
        recorder,
        lambda: _observation(
            observation_id=observation_id,
            kind="linktools.storage.operation",
            source_namespace=source_namespace,
            tenant_id=tenant_id,
            status=status,
            error_code=error_code,
            correlation=_metric_correlation(
                correlation,
                execution_id=execution_id,
                session_id=session_id,
            ),
            dimensions={"domain": domain, "target": target},
            measurements=(
                _measurement("latency_ns", monotonic_ns() - started_at_ns),
            ),
        ),
    )


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
    except Exception:
        _logger.exception("runtime metric observation rejected")
        return False


__all__ = ["RuntimeMetricFlushResult", "RuntimeMetricStatus"]
