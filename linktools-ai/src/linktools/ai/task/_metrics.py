#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-owned projection from durable task events into Metrics."""

from collections.abc import Mapping

from linktools.core import environ

from ..core import TaskStatus, canonical_sha256
from ..observe import MetricRecorder, Observation
from ._event import TaskEvent, TaskEventType
from ._graph import TaskNode

_logger = environ.get_logger("ai.task.metrics")
_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


def record_task_event(
    recorder: MetricRecorder | None,
    *,
    source_namespace: str | None,
    tenant_id: str,
    event: TaskEvent,
    nodes: Mapping[str, TaskNode],
) -> None:
    """Best-effort projection of one durable TaskEvent."""
    if recorder is None or source_namespace is None or event.status not in _TERMINAL:
        return
    try:
        if event.event_type is TaskEventType.NODE_CHANGED:
            if event.node_id is None or event.fence < 1:
                return
            node = nodes.get(event.node_id)
            if node is None:
                raise ValueError("terminal task event references an unknown node")
            correlation: dict[str, str | int] = {
                "graph_id": event.graph_id,
                "node_id": event.node_id,
                "fence": event.fence,
            }
            if event.execution_id is not None:
                correlation["execution_id"] = event.execution_id
            task_type = node.input.get("type")
            dimensions = (
                {"task_type": task_type}
                if isinstance(task_type, str)
                and task_type
                and task_type == task_type.strip()
                else {}
            )
            observation = Observation(
                version=1,
                observation_id=_stable_observation_id(
                    source_namespace,
                    tenant_id,
                    event.graph_id,
                    event.node_id,
                    str(event.fence),
                    "attempt",
                ),
                kind="linktools.task.node.attempt",
                occurred_at=event.occurred_at,
                source_namespace=source_namespace,
                tenant_id=tenant_id,
                status=event.status.value,
                error_code=event.error_code,
                correlation=correlation,
                dimensions=dimensions,
                measurements=(),
            )
        elif event.node_id is None:
            observation = Observation(
                version=1,
                observation_id=_stable_observation_id(
                    source_namespace,
                    tenant_id,
                    event.graph_id,
                    str(event.sequence),
                    "terminal",
                ),
                kind="linktools.task.graph.terminal",
                occurred_at=event.occurred_at,
                source_namespace=source_namespace,
                tenant_id=tenant_id,
                status=event.status.value,
                error_code=None,
                correlation={"graph_id": event.graph_id},
                dimensions={},
                measurements=(),
            )
        else:
            return
        recorder.try_record(observation)
    except Exception as error:
        _logger.warning(
            "task metric projection skipped: graph=%s sequence=%s error=%s",
            event.graph_id,
            event.sequence,
            type(error).__name__,
        )


def _stable_observation_id(*parts: str) -> str:
    return canonical_sha256({"parts": list(parts)})


__all__: list[str] = []
