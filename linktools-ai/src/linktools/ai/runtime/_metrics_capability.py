#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit automatic Agent metrics backed by a shared Metrics dataset."""

from __future__ import annotations

from typing import TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..capability import AgentContext
from ..observe import MetricRecorder
from ._budget import RunBudgetContext
from ._journal import ModelRequestJournal
from ._metric_capability import ModelInteractionRecorder, ModelObservationCapability, ModelRequestEventSink
from ._tool_metrics import ToolMetricContext

AppT = TypeVar("AppT")


class MetricsCapability(AbstractCapability[AgentContext[AppT]]):
    """Collect automatic Agent model, tool, attempt and execution metrics.

    Register at Runtime scope and supply a Metrics dataset to Runtime. Task,
    Evaluation and business recording use that dataset independently. Runtime
    owns the recording lifetime; observers do not force turn-boundary flushes.
    """

    id: str | None = "runtime.metrics"

    def bind_model_observer(
        self, recorder: MetricRecorder, *, source_namespace: str, tenant_id: str,
        execution_id: str, session_id: str | None, agent_run_id: str, agent_id: str,
        agent_run_seq: int = 1, journal: ModelRequestJournal | None = None,
        interaction_recorder: ModelInteractionRecorder | None = None,
        event_sink: ModelRequestEventSink | None = None, budget: RunBudgetContext | None = None,
    ) -> ModelObservationCapability:
        """Bind the metric sink while preserving model journal and budget duties."""
        return ModelObservationCapability(
            recorder, source_namespace=source_namespace, tenant_id=tenant_id,
            execution_id=execution_id, session_id=session_id, agent_run_id=agent_run_id,
            agent_id=agent_id, agent_run_seq=agent_run_seq, journal=journal,
            interaction_recorder=interaction_recorder, event_sink=event_sink, budget=budget,
        )

    def bind_tool_observer(
        self, recorder: MetricRecorder, *, source_namespace: str, tenant_id: str,
        execution_id: str, session_id: str | None, agent_run_id: str, agent_id: str,
    ) -> ToolMetricContext:
        """Bind shared context for native and managed tool observations."""
        return ToolMetricContext(
            recorder, source_namespace=source_namespace, tenant_id=tenant_id,
            execution_id=execution_id, session_id=session_id, agent_run_id=agent_run_id,
            agent_id=agent_id,
        )


__all__ = ["MetricsCapability"]
