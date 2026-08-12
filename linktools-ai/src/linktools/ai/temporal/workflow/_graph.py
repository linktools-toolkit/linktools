#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic TaskGraph workflow boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ...core import validate_lease_owner, validate_tenant_id
from ...errors import AIError, ErrorCode
from ...task import SwarmLimits
from ._execution import (
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
)


@dataclass(frozen=True, slots=True)
class TaskWorkflowNode:
    task_id: str
    dependencies: "tuple[str, ...]"
    binding_digest: str
    budget_cost: int = 1
    owner: str = ""
    fence: int = 0
    operation_id: str = ""

    def __post_init__(self) -> None:
        if self.owner:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise AIError(ErrorCode.TASK_DAG_INVALID) from error
        if not self.task_id.strip() or not self.binding_digest.strip() or len(self.dependencies) > 256 or len(set(self.dependencies)) != len(self.dependencies) or any(not dependency.strip() for dependency in self.dependencies) or self.budget_cost < 1 or self.fence < 0 or (self.fence and not self.owner.strip()) or (self.operation_id and not self.owner.strip()):
            raise AIError(ErrorCode.TASK_DAG_INVALID)


@dataclass(frozen=True, slots=True)
class TaskWorkflowInput:
    graph_id: str
    tenant_id: str
    nodes: "tuple[TaskWorkflowNode, ...]"
    limits: SwarmLimits
    request_ref: str
    worker_build: str

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
        except AIError as error:
            raise ValueError("task workflow tenant is invalid") from error
        if not self.graph_id.strip() or not self.request_ref.strip() or not self.worker_build.strip():
            raise ValueError("task workflow input is incomplete")
        object.__setattr__(self, "nodes", tuple(self.nodes))


@dataclass(frozen=True, slots=True)
class TaskWorkflowResult:
    graph_id: str
    status: str
    completed_task_ids: "tuple[str, ...]"


class TaskActivity(Protocol):
    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult: ...


class _ChildWorkflowHandle(Protocol):
    def cancel(self) -> bool: ...

    async def result(self) -> ExecutionWorkflowResult: ...


class TaskWorkflow:
    def __init__(self, activity: 'TaskActivity | None' = None) -> None:
        self._activity = activity
        self._cancelled = False
        self._graph_id = ""
        self._active_children: list[_ChildWorkflowHandle] = []

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        self._graph_id = request.graph_id
        if len(set(node.task_id for node in request.nodes)) != len(request.nodes):
            raise ValueError("task workflow contains duplicate task ids")
        _validate_graph(request.nodes, request.limits)
        if self._cancelled:
            return TaskWorkflowResult(request.graph_id, "CANCELLED", ())
        if self._activity is not None:
            result = await self._activity.run(request)
        elif _temporal_workflow is not None and _TemporalRetryPolicy is not None:
            result = await _run_task_children(request, self._is_cancelled, self._active_children)
        else:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        if result.graph_id != request.graph_id:
            raise ValueError("task activity returned mismatched graph identity")
        return result

    def cancel(self, cancel_request_id: str) -> TaskWorkflowResult:
        if not cancel_request_id.strip():
            raise ValueError("cancel request id is required")
        self._cancelled = True
        for child in tuple(self._active_children):
            child.cancel()
        return TaskWorkflowResult(self._graph_id, "CANCELLED", ())

    def _is_cancelled(self) -> bool:
        return self._cancelled


async def _run_task_children(
    request: TaskWorkflowInput,
    is_cancelled: 'Callable[[], bool]',
    active_children: 'list[_ChildWorkflowHandle]',
) -> TaskWorkflowResult:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    nodes = request.nodes
    completed: set[str] = set()
    failed: set[str] = set()
    pending = {node.task_id: node for node in nodes}
    while pending:
        if is_cancelled():
            return TaskWorkflowResult(request.graph_id, "CANCELLED", tuple(sorted(completed)))
        blocked = {task_id for task_id, node in pending.items() if any(dependency in failed for dependency in node.dependencies)}
        failed.update(blocked)
        for task_id in blocked:
            pending.pop(task_id)
        ready = tuple(sorted((node for node in pending.values() if all(dependency in completed for dependency in node.dependencies)), key=lambda node: node.task_id))
        if not ready:
            if pending:
                raise AIError(ErrorCode.TASK_GRAPH_DEADLOCK)
            break
        limits = request.limits
        for offset in range(0, len(ready), limits.max_concurrency):
            batch = ready[offset:offset + limits.max_concurrency]
            if is_cancelled():
                return TaskWorkflowResult(request.graph_id, "CANCELLED", tuple(sorted(completed)))
            handles = tuple(_temporal_workflow.start_child_workflow(
                ExecutionWorkflow,
                ExecutionWorkflowInput(
                    execution_id=f"{request.graph_id}:{node.task_id}",
                    tenant_id=request.tenant_id or "task",
                    binding_digest=node.binding_digest,
                    bundle_digest=node.binding_digest,
                    request_ref=request.request_ref or f"task:{request.graph_id}:{node.task_id}",
                    worker_build=request.worker_build,
                    owner=node.owner or f"task-workflow:{request.graph_id}",
                    fence=node.fence or 1,
                    operation_id=node.operation_id or f"task:{request.graph_id}:{node.task_id}",
                ),
                id=f"{request.graph_id}:{node.task_id}",
                retry_policy=_TemporalRetryPolicy(maximum_attempts=1),
            ) for node in batch)
            active_children.extend(handles)
            results_list: list[ExecutionWorkflowResult] = []
            try:
                for handle in handles:
                    results_list.append(await handle.result())
                    if is_cancelled():
                        for remaining in handles:
                            remaining.cancel()
                        return TaskWorkflowResult(request.graph_id, "CANCELLED", tuple(sorted(completed)))
            finally:
                for handle in handles:
                    active_children.remove(handle)
            results = tuple(results_list)
            for node, result in zip(batch, results):
                if result.status == "SUCCEEDED":
                    completed.add(node.task_id)
                else:
                    failed.add(node.task_id)
                pending.pop(node.task_id)
    status = "SUCCEEDED" if len(completed) == len(nodes) else "FAILED"
    return TaskWorkflowResult(request.graph_id, status, tuple(sorted(completed)))


def _validate_graph(nodes: tuple[TaskWorkflowNode, ...], limits: SwarmLimits) -> None:
    if len(nodes) > limits.max_nodes or sum(node.budget_cost for node in nodes) > limits.max_budget:
        raise AIError(ErrorCode.TASK_DAG_INVALID)
    identifiers = {node.task_id for node in nodes}
    if len(identifiers) != len(nodes) or any(dependency not in identifiers for node in nodes for dependency in node.dependencies):
        raise AIError(ErrorCode.TASK_DEPENDENCY_UNKNOWN)
    remaining = {node.task_id: set(node.dependencies) for node in nodes}
    depth: dict[str, int] = {}
    while remaining:
        ready = tuple(sorted(task_id for task_id, dependencies in remaining.items() if not dependencies))
        if not ready:
            raise AIError(ErrorCode.TASK_GRAPH_CYCLE)
        for task_id in ready:
            node = next(node for node in nodes if node.task_id == task_id)
            depth[task_id] = 1 + max((depth[item] for item in node.dependencies), default=0)
            remaining.pop(task_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    if max(depth.values(), default=0) > limits.max_depth:
        raise AIError(ErrorCode.TASK_DAG_INVALID)


def _validate_child_results(
    request: TaskWorkflowInput,
    results: "tuple[ExecutionWorkflowResult, ...]",
) -> None:
    if len(results) != len(request.nodes):
        raise ValueError("task child workflow count does not match the graph")
    for result, node in zip(results, request.nodes):
        task_id = node.task_id
        if result.execution_id != f"{request.graph_id}:{task_id}":
            raise ValueError("task child workflow returned mismatched identity")
        if result.status not in {"SUCCEEDED", "FAILED", "CANCELLED", "WAITING"}:
            raise ValueError("task child workflow returned an invalid status")


if _temporal_workflow is not None:
    TaskWorkflow.run = _temporal_workflow.run(TaskWorkflow.run)
    TaskWorkflow.cancel = _temporal_workflow.update(name="cancel")(TaskWorkflow.cancel)
    TaskWorkflow = _temporal_workflow.defn(name="TaskWorkflow")(TaskWorkflow)


__all__ = ["TaskActivity", "TaskWorkflow", "TaskWorkflowInput", "TaskWorkflowNode", "TaskWorkflowResult"]
