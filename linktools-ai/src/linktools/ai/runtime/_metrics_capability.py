#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit automatic Agent metrics backed by a shared Metrics dataset."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..capability import AgentContext
from ..errors import AIError, ErrorCode
from ..observe import MetricFlushResult, MetricRecorder, MetricRecorderStatus, Metrics
from ._budget import RunBudgetContext
from ._journal import ModelRequestJournal
from ._metric_capability import ModelInteractionRecorder, ModelObservationCapability, ModelRequestEventSink
from ._tool_metrics import ToolMetricContext
from ._metrics import RuntimeMetricRecorder

AppT = TypeVar("AppT")


class MetricsRecordingBinding:
    """Own one Runtime recording lifetime over a borrowed Metrics dataset.

    Agent, Task and Evaluation observations share this recording scope. Closing
    it drains only its queue and leaves the dataset, engine and cache reusable.
    """

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._recorder = RuntimeMetricRecorder(metrics.recorder())

    @property
    def recorder(self) -> RuntimeMetricRecorder:
        return self._recorder

    def bind(self, capability: MetricsCapability[AppT]) -> MetricsCapability[AppT]:
        """Keep registered configuration while isolating each Runtime's sink."""
        if capability.metrics is not self._metrics:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        bound = copy(capability)
        bound._recording = self
        return bound

    def configure_runtime_dimensions(self, dimensions: Mapping[str, str]) -> None:
        self._recorder.configure_runtime_dimensions(dimensions)

    def status(self) -> MetricRecorderStatus:
        return self._recorder.status()

    async def flush(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        return await self._recorder.flush(timeout_seconds=timeout_seconds)

    async def close(self) -> None:
        await self._recorder.close()


class MetricsCapability(AbstractCapability[AgentContext[AppT]]):
    """Collect automatic Agent model, tool, attempt and execution metrics.

    Register at Runtime scope with the shared Metrics dataset. Agent, Task and
    Evaluation observations share one independently owned recording lifetime.
    Explicit business recording stays available on the dataset independently.
    """

    id: str | None = "runtime.metrics"
    _recording: MetricsRecordingBinding | None = None

    def __init__(self, *, metrics: Metrics) -> None:
        if not isinstance(metrics, Metrics):
            raise TypeError("metrics must be Metrics")
        self._metrics = metrics

    @property
    def metrics(self) -> Metrics:
        """The borrowed dataset used by this capability's recording lifetime."""
        return self._metrics

    def open_recording(self) -> MetricsRecordingBinding:
        """Create one isolated recording lifetime for this Runtime binding."""
        return MetricsRecordingBinding(self._metrics)

    def _recorder(self) -> MetricRecorder:
        if self._recording is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._recording.recorder

    def bind_model_observer(
        self, *, source_namespace: str, tenant_id: str,
        execution_id: str, session_id: str | None, agent_run_id: str, agent_id: str,
        agent_run_seq: int = 1, journal: ModelRequestJournal | None = None,
        interaction_recorder: ModelInteractionRecorder | None = None,
        event_sink: ModelRequestEventSink | None = None, budget: RunBudgetContext | None = None,
    ) -> ModelObservationCapability:
        """Bind the metric sink while preserving model journal and budget duties."""
        return ModelObservationCapability(
            self._recorder(), source_namespace=source_namespace, tenant_id=tenant_id,
            execution_id=execution_id, session_id=session_id, agent_run_id=agent_run_id,
            agent_id=agent_id, agent_run_seq=agent_run_seq, journal=journal,
            interaction_recorder=interaction_recorder, event_sink=event_sink, budget=budget,
        )

    def bind_tool_observer(
        self, *, source_namespace: str, tenant_id: str,
        execution_id: str, session_id: str | None, agent_run_id: str, agent_id: str,
    ) -> ToolMetricContext:
        """Bind shared context for native and managed tool observations."""
        return ToolMetricContext(
            self._recorder(), source_namespace=source_namespace, tenant_id=tenant_id,
            execution_id=execution_id, session_id=session_id, agent_run_id=agent_run_id,
            agent_id=agent_id,
        )


__all__ = ["MetricsCapability", "MetricsRecordingBinding"]
