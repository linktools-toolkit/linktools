#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic TaskGraph workflow boundary."""

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

from .execution import ExecutionWorkflow, ExecutionWorkflowInput, ExecutionWorkflowResult


@dataclass(frozen=True, slots=True)
class TaskWorkflowInput:
    graph_id: str
    binding_digest: str
    task_ids: "tuple[str, ...]"
    tenant_id: str = ""
    request_ref: str = ""
    worker_build: str = "linktools-ai"


@dataclass(frozen=True, slots=True)
class TaskWorkflowResult:
    graph_id: str
    status: str
    completed_task_ids: "tuple[str, ...]"


class TaskActivity(Protocol):
    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult: ...


class TaskWorkflow:
    def __init__(self, activity: 'TaskActivity | None' = None) -> None:
        self._activity = activity

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        if not request.graph_id or not request.binding_digest:
            raise ValueError("task workflow input is incomplete")
        if len(set(request.task_ids)) != len(request.task_ids):
            raise ValueError("task workflow contains duplicate task ids")
        if self._activity is not None:
            result = await self._activity.run(request)
        elif _temporal_workflow is not None and _TemporalRetryPolicy is not None:
            result = await _run_task_children(request)
        else:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        if result.graph_id != request.graph_id:
            raise ValueError("task activity returned mismatched graph identity")
        return result


async def _run_task_children(request: TaskWorkflowInput) -> TaskWorkflowResult:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    handles = tuple(
        _temporal_workflow.start_child_workflow(
            ExecutionWorkflow,
            ExecutionWorkflowInput(
                execution_id=f"{request.graph_id}:{task_id}",
                tenant_id=request.tenant_id or "task",
                binding_digest=request.binding_digest,
                bundle_digest=request.binding_digest,
                request_ref=request.request_ref or f"task:{request.graph_id}:{task_id}",
                worker_build=request.worker_build,
            ),
            id=f"{request.graph_id}:{task_id}",
            retry_policy=_TemporalRetryPolicy(maximum_attempts=3),
        )
        for task_id in request.task_ids
    )
    results = tuple(await handle.result() for handle in handles)
    _validate_child_results(request, results)
    completed = tuple(
        task_id
        for task_id, result in zip(request.task_ids, results)
        if result.status == "SUCCEEDED"
    )
    status = "SUCCEEDED" if len(completed) == len(request.task_ids) else "FAILED"
    return TaskWorkflowResult(request.graph_id, status, completed)


def _validate_child_results(
    request: TaskWorkflowInput,
    results: "tuple[ExecutionWorkflowResult, ...]",
) -> None:
    if len(results) != len(request.task_ids):
        raise ValueError("task child workflow count does not match the graph")
    for result, task_id in zip(results, request.task_ids):
        if result.execution_id != f"{request.graph_id}:{task_id}":
            raise ValueError("task child workflow returned mismatched identity")
        if result.status not in {"SUCCEEDED", "FAILED", "CANCELLED", "WAITING"}:
            raise ValueError("task child workflow returned an invalid status")


if _temporal_workflow is not None:
    TaskWorkflow.run = _temporal_workflow.run(TaskWorkflow.run)
    TaskWorkflow = _temporal_workflow.defn(name="TaskWorkflow")(TaskWorkflow)


__all__ = ["TaskActivity", "TaskWorkflow", "TaskWorkflowInput", "TaskWorkflowResult"]
