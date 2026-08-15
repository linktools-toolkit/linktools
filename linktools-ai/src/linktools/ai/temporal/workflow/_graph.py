#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic TaskGraph workflow boundary."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, cast

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ...core import (
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...task import (
    TaskDependencyResult,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskLease,
    TaskNode,
)
from ._execution import (
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
)


@dataclass(frozen=True, slots=True)
class TaskWorkflowNode:
    node_id: str
    dependencies: tuple[str, ...]
    input_json: str
    budget_cost: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_json, str)
            or not self.input_json
            or isinstance(self.dependencies, (str, bytes))
        ):
            raise ValueError("task workflow node input is required")
        try:
            dependencies = tuple(self.dependencies)
            value = json.loads(self.input_json)
            if not isinstance(value, dict) or canonical_json_bytes(value).decode("utf-8") != self.input_json:
                raise ValueError("task workflow node input is not canonical JSON")
            TaskNode(
                self.node_id,
                dependencies,
                input=cast(Mapping[str, JsonValue], value),
                budget_cost=self.budget_cost,
            )
            object.__setattr__(self, "dependencies", dependencies)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("task workflow node is invalid") from error


@dataclass(frozen=True, slots=True)
class TaskWorkflowInput:
    graph_id: str
    tenant_id: str
    nodes: tuple[TaskWorkflowNode, ...]
    limits: TaskGraphLimits
    request_ref: str
    worker_build: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.graph_id, self.request_ref, self.worker_build)
        ):
            raise ValueError("task workflow input is incomplete")
        try:
            validate_tenant_id(self.tenant_id)
            graph = TaskGraph(self.graph_id, tuple(_task_node(node) for node in self.nodes))
            graph.validate_limits(self.limits)
        except (AIError, AttributeError, TypeError, ValueError) as error:
            raise ValueError("task workflow graph is invalid") from error
        object.__setattr__(self, "nodes", tuple(self.nodes))

    @classmethod
    def from_request(
        cls,
        request: TaskGraphRequest,
        *,
        request_ref: str,
        worker_build: str,
    ) -> "TaskWorkflowInput":
        return cls(
            request.graph.graph_id,
            request.principal.tenant_id,
            tuple(
                TaskWorkflowNode(
                    node.node_id,
                    node.dependencies,
                    canonical_json_bytes(node.input).decode("utf-8"),
                    node.budget_cost,
                )
                for node in request.graph.nodes
            ),
            request.limits,
            request_ref,
            worker_build,
        )


@dataclass(frozen=True, slots=True)
class TaskWorkflowResult:
    graph_id: str
    status: str
    completed_node_ids: tuple[str, ...]


class TaskActivity(Protocol):
    async def prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> tuple[TaskLease, ExecutionWorkflowInput]: ...

    async def renew(self, lease: TaskLease) -> TaskLease: ...

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskDependencyResult | None: ...


class _ChildWorkflowHandle(Protocol):
    def cancel(self) -> bool: ...

    async def result(self) -> ExecutionWorkflowResult: ...


class TaskWorkflow:
    def __init__(self, activity: "TaskActivity | None" = None) -> None:
        self._activity = activity
        self._cancelled = False
        self._graph_id = ""
        self._active_children: list[_ChildWorkflowHandle] = []

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        self._graph_id = request.graph_id
        if self._cancelled:
            return TaskWorkflowResult(request.graph_id, "CANCELLED", ())
        completed_results: dict[str, TaskDependencyResult] = {}
        failed: set[str] = set()
        pending = {node.node_id: node for node in request.nodes}
        while pending:
            if self._cancelled:
                return TaskWorkflowResult(request.graph_id, "CANCELLED", tuple(sorted(completed_results)))
            blocked = {
                node_id
                for node_id, node in pending.items()
                if any(dependency in failed for dependency in node.dependencies)
            }
            failed.update(blocked)
            for node_id in blocked:
                pending.pop(node_id)
            ready = tuple(
                sorted(
                    (
                        node
                        for node in pending.values()
                        if all(dependency in completed_results for dependency in node.dependencies)
                    ),
                    key=lambda node: node.node_id,
                )
            )
            if not ready:
                if pending:
                    raise AIError(ErrorCode.TASK_GRAPH_DEADLOCK)
                break
            for offset in range(0, len(ready), request.limits.max_concurrency):
                batch = ready[offset : offset + request.limits.max_concurrency]
                results = await asyncio.gather(
                    *(
                        self._run_node(
                            request,
                            node,
                            {
                                dependency: completed_results[dependency]
                                for dependency in node.dependencies
                            },
                        )
                        for node in batch
                    )
                )
                for node, result in zip(batch, results):
                    pending.pop(node.node_id)
                    if result is None:
                        failed.add(node.node_id)
                    else:
                        completed_results[node.node_id] = result
        status = "SUCCEEDED" if not failed else "FAILED"
        return TaskWorkflowResult(request.graph_id, status, tuple(sorted(completed_results)))

    def cancel(self, idempotency_key: str) -> TaskWorkflowResult:
        if not idempotency_key.strip():
            raise ValueError("cancel request id is required")
        self._cancelled = True
        for child in tuple(self._active_children):
            child.cancel()
        return TaskWorkflowResult(self._graph_id, "CANCELLED", ())

    async def _run_node(
        self,
        request: TaskWorkflowInput,
        node: TaskWorkflowNode,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> TaskDependencyResult | None:
        lease, execution_input = await self._prepare(request, node.node_id, dependency_results)
        child = self._start_child(execution_input, request, node)
        self._active_children.append(child)
        try:
            try:
                lease, result = await self._wait_for_child(execution_input, lease, child)
            except asyncio.CancelledError:
                raise
            except Exception:
                result = ExecutionWorkflowResult(
                    execution_input.execution_id,
                    "FAILED",
                    None,
                    0,
                )
        finally:
            self._active_children.remove(child)
        return await self._settle(request, lease, result)

    async def _prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> tuple[TaskLease, ExecutionWorkflowInput]:
        if self._activity is not None:
            return await self._activity.prepare(request, node_id, dependency_results)
        return cast(
            tuple[TaskLease, ExecutionWorkflowInput],
            await _execute_task_activity(
                "task_node_prepare",
                request,
                node_id,
                dependency_results,
            ),
        )

    async def _settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskDependencyResult | None:
        if self._activity is not None:
            return await self._activity.settle(request, lease, result)
        return cast(
            "TaskDependencyResult | None",
            await _execute_task_activity("task_node_settle", request, lease, result),
        )

    async def _wait_for_child(
        self,
        execution_input: ExecutionWorkflowInput,
        lease: TaskLease,
        child: _ChildWorkflowHandle,
    ) -> tuple[TaskLease, ExecutionWorkflowResult]:
        child_task = asyncio.create_task(child.result())
        try:
            current_lease = lease
            while True:
                renew_task = asyncio.create_task(self._renew(current_lease))
                done, _ = await asyncio.wait(
                    (child_task, renew_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if child_task in done:
                    renew_task.cancel()
                    await asyncio.gather(renew_task, return_exceptions=True)
                    return current_lease, child_task.result()
                current_lease = renew_task.result()
                if self._cancelled:
                    child.cancel()
                    return current_lease, ExecutionWorkflowResult(
                        execution_input.execution_id,
                        "CANCELLED",
                        None,
                        0,
                    )
        except asyncio.CancelledError:
            child.cancel()
            raise

    async def _renew(self, lease: TaskLease) -> TaskLease:
        if self._activity is not None:
            return await self._activity.renew(lease)
        await asyncio.sleep(30)
        return cast(TaskLease, await _execute_task_activity("task_node_renew", lease))

    def _start_child(
        self,
        execution_input: ExecutionWorkflowInput,
        request: TaskWorkflowInput,
        node: TaskWorkflowNode,
    ) -> _ChildWorkflowHandle:
        if _temporal_workflow is None or _TemporalRetryPolicy is None:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        workflow_id = "task-node-" + canonical_sha256(
            {
                "tenant_id": request.tenant_id,
                "graph_id": request.graph_id,
                "node_id": node.node_id,
            }
        )
        return cast(
            _ChildWorkflowHandle,
            _temporal_workflow.start_child_workflow(
                ExecutionWorkflow,
                execution_input,
                id=workflow_id,
                retry_policy=_TemporalRetryPolicy(maximum_attempts=1),
            ),
        )


async def _execute_task_activity(name: str, *args: object) -> object:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    return await _temporal_workflow.execute_activity(
        name,
        args=list(args),
        start_to_close_timeout=timedelta(seconds=60),
        heartbeat_timeout=timedelta(seconds=15),
        retry_policy=_TemporalRetryPolicy(maximum_attempts=3),
    )


def _task_node(node: TaskWorkflowNode) -> TaskNode:
    return TaskNode(
        node.node_id,
        node.dependencies,
        input=cast(Mapping[str, JsonValue], json.loads(node.input_json)),
        budget_cost=node.budget_cost,
    )


if _temporal_workflow is not None:
    TaskWorkflow.run = _temporal_workflow.run(TaskWorkflow.run)
    TaskWorkflow.cancel = _temporal_workflow.update(name="cancel")(TaskWorkflow.cancel)
    TaskWorkflow = _temporal_workflow.defn(name="TaskWorkflow")(TaskWorkflow)


__all__ = [
    "TaskActivity",
    "TaskWorkflow",
    "TaskWorkflowInput",
    "TaskWorkflowNode",
    "TaskWorkflowResult",
]
