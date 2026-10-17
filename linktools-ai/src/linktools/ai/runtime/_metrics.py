#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime observation enrichment without owning a write queue."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from time import monotonic, monotonic_ns

from linktools.core import environ

from ..core import CorrelationData, UsageMetrics, normalize_correlation
from ..errors import AIError, ErrorCode
from ..observe import (
    MetricRecorderStatus, MetricFlushResult, MetricMeasurement, MetricRecorder, Observation,
)
from ._metric_id import _stable_observation_id
from .state._contracts import ExecutionTerminalCommitResult

_logger = environ.get_logger("ai.runtime.metrics")
_WARNING_INTERVAL_SECONDS = 30.0
_FRAMEWORK_CORRELATION_KEYS = frozenset(
    {
        "execution_id",
        "session_id",
        "agent_run_id",
        "tool_call_id",
        "parent_execution_id",
        "root_execution_id",
        "parent_invocation_id",
        "graph_id",
        "node_id",
        "fence",
        "attempt_index",
    }
)


def _disabled_metric_status() -> MetricRecorderStatus:
    return MetricRecorderStatus(False, False, 0, 0, 0, 0, 0, 0)


def _metric_correlation(
    correlation: CorrelationData | Mapping[str, object] | None,
    **system: str | int | None,
) -> dict[str, str | int]:
    values: dict[str, str | int] = dict(normalize_correlation(correlation))
    for key, value in system.items():
        if value is not None:
            values[f"linktools.{key}"] = value
    return values


class _RuntimeMetricRecorder:
    """Enrich one Runtime's observations and delegate its recording lifetime."""

    def __init__(self, recorder: MetricRecorder) -> None:
        self._recorder = recorder
        self._execution_contexts: dict[str, CorrelationData] = {}
        self._runtime_dimensions: dict[str, str] = {}
        self._runtime_dimensions_configured = False
        self._recording_started = False
        self._rejected = 0
        self._last_failure_code: str | None = None
        self._failure_baseline: tuple[int, int, int, str | None] | None = None
        self._last_warning_at = 0.0

    def configure_runtime_dimensions(self, dimensions: Mapping[str, str]) -> None:
        normalized = {f"context.{key}": value for key, value in dimensions.items()}
        if self._runtime_dimensions_configured:
            if self._runtime_dimensions != normalized:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return
        if self._recording_started or self._execution_contexts:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._runtime_dimensions = normalized
        self._runtime_dimensions_configured = True

    def bind_execution_context(
        self, execution_id: str, correlation: CorrelationData | Mapping[str, object],
    ) -> bool:
        try:
            normalized = normalize_correlation(correlation)
        except (TypeError, ValueError):
            self._fail("EXECUTION_CONTEXT_INVALID")
            return False
        current = self._execution_contexts.get(execution_id)
        if current is not None and dict(current) != dict(normalized):
            self._fail("EXECUTION_CONTEXT_CONFLICT")
            return False
        self._execution_contexts[execution_id] = normalized
        return True

    def release_execution_context(self, execution_id: str) -> None:
        self._execution_contexts.pop(execution_id, None)

    def capture(self, observation: Observation) -> Observation | None:
        try:
            if not isinstance(observation, Observation):
                raise TypeError("observation must be Observation")
            return self._enrich_observation(observation)
        except (AIError, TypeError, ValueError):
            self._rejected += 1
            self._fail("ENRICHMENT_FAILED")
            return None

    async def try_record(self, observation: Observation) -> bool:
        captured = self.capture(observation)
        if captured is None:
            return False
        self._recording_started = True
        return await self._recorder.try_record(captured)

    def status(self) -> MetricRecorderStatus:
        status = self._recorder.status()
        if self._failure_baseline != (
            status.rejected, status.unconfirmed, status.write_failures, status.last_failure_code,
        ):
            self._last_failure_code = None
        return replace(
            status, rejected=status.rejected + self._rejected,
            last_failure_code=self._last_failure_code or status.last_failure_code,
        )

    async def flush(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        result = await self._recorder.flush(timeout_seconds=timeout_seconds)
        return MetricFlushResult(result.completed, self.status())

    async def close(self) -> None:
        try:
            await self._recorder.close()
        finally:
            self._execution_contexts.clear()

    def _fail(self, code: str) -> None:
        self._last_failure_code = code
        status = self._recorder.status()
        self._failure_baseline = (
            status.rejected, status.unconfirmed, status.write_failures, status.last_failure_code,
        )
        now = monotonic()
        if now - self._last_warning_at >= _WARNING_INTERVAL_SECONDS:
            self._last_warning_at = now
            _logger.warning("runtime metric enrichment failed: code=%s", code)

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


def _bind_metric_execution_context(
    recorder: MetricRecorder,
    execution_id: str,
    correlation: CorrelationData,
) -> None:
    if isinstance(recorder, _RuntimeMetricRecorder):
        recorder.bind_execution_context(execution_id, correlation)


def _release_metric_execution_context(
    recorder: MetricRecorder | None,
    execution_id: str,
) -> None:
    if not isinstance(recorder, _RuntimeMetricRecorder):
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


def _capture_execution_terminal(
    recorder: MetricRecorder | None,
    *,
    source_namespace: str,
    tenant_id: str,
    result: ExecutionTerminalCommitResult,
    session_id: str | None,
) -> Observation | None:
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
        parent_invocation_id=execution.parent_invocation_id,
    )
    return _capture_metric_observation(
        recorder,
        lambda: _observation(
            observation_id=_stable_observation_id(
                "linktools.execution.terminal.v1",
                source_namespace,
                tenant_id,
                execution.execution_id,
            ),
            kind="linktools.execution.terminal",
            source_namespace=source_namespace,
            tenant_id=tenant_id,
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


def _capture_storage_operation(
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
) -> Observation | None:
    if recorder is None or observation_id is None or started_at_ns is None:
        return
    return _capture_metric_observation(
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


async def _record_execution_terminal(
    recorder: MetricRecorder | None, *, source_namespace: str, tenant_id: str,
    result: ExecutionTerminalCommitResult, session_id: str | None,
) -> None:
    await _record_metric_observation(recorder, _capture_execution_terminal(
        recorder, source_namespace=source_namespace, tenant_id=tenant_id,
        result=result, session_id=session_id,
    ))


async def _record_storage_operation(
    recorder: MetricRecorder | None, *, observation_id: str | None,
    started_at_ns: int | None, source_namespace: str, tenant_id: str,
    execution_id: str, session_id: str | None, correlation: CorrelationData,
    status: str, error_code: str | None, domain: str, target: str,
) -> None:
    await _record_metric_observation(recorder, _capture_storage_operation(
        recorder, observation_id=observation_id, started_at_ns=started_at_ns,
        source_namespace=source_namespace, tenant_id=tenant_id,
        execution_id=execution_id, session_id=session_id, correlation=correlation,
        status=status, error_code=error_code, domain=domain, target=target,
    ))


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


def _capture_metric_observation(
    recorder: MetricRecorder | None, factory: Callable[[], Observation],
) -> Observation | None:
    if recorder is None:
        return None
    try:
        observation = factory()
        if isinstance(recorder, _RuntimeMetricRecorder):
            return recorder.capture(observation)
        return observation
    except Exception:
        _logger.exception("runtime metric observation capture failed")
        return None


async def _record_metric_observation(
    recorder: MetricRecorder | None, observation: Observation | None,
) -> None:
    if observation is not None:
        await _try_record(recorder, lambda: observation)


async def _try_record(
    recorder: MetricRecorder | None, factory: Callable[[], Observation],
) -> bool:
    if recorder is None:
        return False
    try:
        return await recorder.try_record(factory())
    except Exception:
        _logger.exception("runtime metric observation rejected")
        return False
