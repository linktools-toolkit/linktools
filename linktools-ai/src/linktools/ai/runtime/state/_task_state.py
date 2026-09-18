#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Task durable-state status helpers."""

from ...core import TaskStatus
from ...errors import AIError, ErrorCode
from ...task import TaskGraphView, TaskNodeView


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_canonical_graph_status(status: TaskStatus) -> None:
    if status in {TaskStatus.READY, TaskStatus.WAITING}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _isolated_graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:
    statuses = {node.status for node in nodes}
    if TaskStatus.RECOVERY_REQUIRED in statuses:
        return TaskStatus.RECOVERY_REQUIRED
    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.RUNNING in statuses or TaskStatus.WAITING in statuses:
        return TaskStatus.RUNNING
    if TaskStatus.PENDING in statuses or TaskStatus.READY in statuses:
        return TaskStatus.PENDING
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if TaskStatus.BLOCKED in statuses:
        return TaskStatus.BLOCKED
    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
        return TaskStatus.CANCELLED
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _effective_graph_status(
    graph: TaskGraphView,
    nodes: tuple[TaskNodeView, ...],
) -> TaskStatus:
    isolated = _isolated_graph_status(nodes)
    if isolated is TaskStatus.RECOVERY_REQUIRED:
        return isolated
    if graph.status is TaskStatus.CANCELLED:
        return TaskStatus.CANCELLED
    return isolated
