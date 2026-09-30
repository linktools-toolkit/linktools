#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Task durable-state status helpers."""

from ...core import TaskStatus
from ...errors import AIError, ErrorCode
from ...task import TaskGraphView, TaskNode, TaskNodeView


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_canonical_graph_status(status: TaskStatus) -> None:
    if status in {TaskStatus.READY, TaskStatus.WAITING}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _isolated_graph_status(
    nodes: tuple[TaskNodeView, ...],
    definitions: tuple[TaskNode, ...],
) -> TaskStatus:
    statuses = {node.status for node in nodes}
    failure_policies = {node.node_id: node.failure_policy for node in definitions}
    if set(failure_policies) != {node.node_id for node in nodes}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if TaskStatus.RECOVERY_REQUIRED in statuses:
        return TaskStatus.RECOVERY_REQUIRED
    if TaskStatus.RUNNING in statuses or TaskStatus.WAITING in statuses:
        return TaskStatus.RUNNING
    if TaskStatus.PENDING in statuses or TaskStatus.READY in statuses:
        return TaskStatus.PENDING
    if any(
        node.status is TaskStatus.FAILED
        and failure_policies[node.node_id] == "propagate"
        for node in nodes
    ):
        return TaskStatus.FAILED
    if any(
        node.status is TaskStatus.BLOCKED
        and failure_policies[node.node_id] == "propagate"
        for node in nodes
    ):
        return TaskStatus.BLOCKED
    if TaskStatus.CANCELLED in statuses:
        return TaskStatus.CANCELLED
    return TaskStatus.SUCCEEDED


def _effective_graph_status(
    graph: TaskGraphView,
    nodes: tuple[TaskNodeView, ...],
    definitions: tuple[TaskNode, ...] | None = None,
) -> TaskStatus:
    isolated = _isolated_graph_status(
        nodes,
        graph.nodes if definitions is None else definitions,
    )
    if isolated is TaskStatus.RECOVERY_REQUIRED:
        return isolated
    if graph.status is TaskStatus.CANCELLED:
        return TaskStatus.CANCELLED
    return isolated
