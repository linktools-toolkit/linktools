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
    from temporalio.common import (
        WorkflowIDReusePolicy as _TemporalWorkflowIDReusePolicy,
    )
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None
    _TemporalWorkflowIDReusePolicy = None

from ...core import (
    JsonValue,
    TaskStatus,
    canonical_json_bytes,
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
    TaskNodeView,
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
    input: str
    budget_cost: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input, str)
            or not self.input
            or isinstance(self.dependencies, (str, bytes))
        ):
            raise ValueError("task workflow node input is required")
        try:
            dependencies = tuple(self.dependencies)
            value = json.loads(self.input)
            if (
                not isinstance(value, dict)
                or canonical_json_bytes(value).decode("utf-8") != self.input
            ):
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
            graph = TaskGraph(
                self.graph_id,
                tuple(_task_node(node) for node in self.nodes),
            )
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
    ) -> "TaskNodeView | tuple[TaskLease, ExecutionWorkflowInput]": ...

    async def renew(self, lease: TaskLease) -> "TaskLease | TaskNodeView": ...

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskNodeView: ...


class _ChildWorkflowHandle(Protocol):
    def cancel(self) -> bool: ...

    async def result(self) -> ExecutionWorkflowResult: ...


class TaskWorkflow:
    def __init__(self, activity: "TaskActivity | None" = None) -> None:
        self._activity = activity
        self._cancelled = False
        self._graph_id = ""
        self._active_children: list[_ChildWorkflowHandle] = []
        self._active_node_tasks: set[asyncio.Task[TaskNodeView]] = set()

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        self._graph_id = request.graph_id
        if self._cancelled:
            return TaskWorkflowResult(request.graph_id, "CANCELLED", ())
        completed_results: dict[str, TaskDependencyResult] = {}
        failed: set[str] = set()
        pending = {node.node_id: node for node in request.nodes}
        while pending:
            if self._cancelled:
                return TaskWorkflowResult(
                    request.graph_id,
                    "CANCELLED",
                    tuple(sorted(completed_results)),
                )
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
                        if all(
                            dependency in completed_results
                            for dependency in node.dependencies
                        )
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
                batch_tasks = tuple(
                    asyncio.create_task(
                        self._run_node(
                            request,
                            node,
                            {
                                dependency: completed_results[dependency]
                                for dependency in node.dependencies
                            },
                        )
                    )
                    for node in batch
                )
                self._active_node_tasks.update(batch_tasks)
                try:
                    results = await asyncio.gather(*batch_tasks)
                except BaseException:
                    await self._cleanup_batch(batch_tasks)
                    if self._cancelled:
                        return TaskWorkflowResult(
                            request.graph_id,
                            "CANCELLED",
                            tuple(sorted(completed_results)),
                        )
                    raise
                finally:
                    self._active_node_tasks.difference_update(batch_tasks)
                for node, result in zip(batch, results):
                    pending.pop(node.node_id)
                    if result.status is TaskStatus.SUCCEEDED:
                        if result.execution_id is None or result.result_digest is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        completed_results[node.node_id] = TaskDependencyResult(
                            result.result_digest,
                            result.execution_id,
                        )
                    elif result.status in {
                        TaskStatus.FAILED,
                        TaskStatus.BLOCKED,
                    }:
                        failed.add(node.node_id)
                    elif result.status is TaskStatus.CANCELLED:
                        return TaskWorkflowResult(
                            request.graph_id,
                            "CANCELLED",
                            tuple(sorted(completed_results)),
                        )
                    else:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if self._cancelled:
                    return TaskWorkflowResult(
                        request.graph_id,
                        "CANCELLED",
                        tuple(sorted(completed_results)),
                    )
        if self._cancelled:
            return TaskWorkflowResult(
                request.graph_id,
                "CANCELLED",
                tuple(sorted(completed_results)),
            )
        status = "SUCCEEDED" if not failed else "FAILED"
        return TaskWorkflowResult(
            request.graph_id,
            status,
            tuple(sorted(completed_results)),
        )

    def cancel(self, idempotency_key: str) -> TaskWorkflowResult:
        if not idempotency_key.strip():
            raise ValueError("cancel request id is required")
        self._cancelled = True
        for child in tuple(self._active_children):
            child.cancel()
        for task in tuple(self._active_node_tasks):
            if not task.done():
                task.cancel()
        return TaskWorkflowResult(self._graph_id, "CANCELLED", ())

    async def _run_node(
        self,
        request: TaskWorkflowInput,
        node: TaskWorkflowNode,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> TaskNodeView:
        while True:
            prepared = await self._prepare(request, node.node_id, dependency_results)
            if isinstance(prepared, TaskNodeView):
                if prepared.status is TaskStatus.RUNNING:
                    await _wait_for_foreign_lease(prepared)
                    continue
                if prepared.status in {
                    TaskStatus.SUCCEEDED,
                    TaskStatus.FAILED,
                    TaskStatus.BLOCKED,
                    TaskStatus.CANCELLED,
                }:
                    return prepared
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            lease, execution_input = prepared
            child = self._start_child(execution_input, request, node)
            self._active_children.append(child)
            try:
                waited = await self._wait_for_child(
                    execution_input,
                    lease,
                    child,
                )
            finally:
                self._active_children.remove(child)
            if isinstance(waited, TaskNodeView):
                continue
            lease, result = waited
            return await self._settle(request, lease, result)

    async def _prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> "TaskNodeView | tuple[TaskLease, ExecutionWorkflowInput]":
        if self._activity is not None:
            return await self._activity.prepare(request, node_id, dependency_results)
        return cast(
            "TaskNodeView | tuple[TaskLease, ExecutionWorkflowInput]",
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
    ) -> TaskNodeView:
        if self._activity is not None:
            return await self._activity.settle(request, lease, result)
        return cast(
            "TaskNodeView",
            await _execute_task_activity("task_node_settle", request, lease, result),
        )

    async def _cleanup_batch(
        self,
        batch_tasks: tuple["asyncio.Task[TaskNodeView]", ...],
    ) -> None:
        for child in tuple(self._active_children):
            child.cancel()
        for task in batch_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*batch_tasks, return_exceptions=True)

    async def _wait_for_child(
        self,
        execution_input: ExecutionWorkflowInput,
        lease: TaskLease,
        child: _ChildWorkflowHandle,
    ) -> "TaskNodeView | tuple[TaskLease, ExecutionWorkflowResult]":
        child_task = asyncio.create_task(child.result())
        renew_task: asyncio.Task[TaskLease | TaskNodeView] | None = None
        try:
            current_lease = lease
            while True:
                renew_task = asyncio.create_task(self._renew(current_lease))
                done, _ = await asyncio.wait(
                    (child_task, renew_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if renew_task in done:
                    try:
                        current_lease = renew_task.result()
                    except BaseException as error:
                        if not isinstance(error, Exception):
                            raise
                        await _cancel_local_tasks(child_task, renew_task)
                        if self._cancelled:
                            raise asyncio.CancelledError from error
                        child.cancel()
                        raise
                    renew_task = None
                    if isinstance(current_lease, TaskNodeView):
                        child.cancel()
                        await _cancel_local_tasks(child_task)
                        return current_lease
                    if self._cancelled:
                        child.cancel()
                        await _cancel_local_tasks(child_task)
                        raise asyncio.CancelledError
                    if child_task not in done:
                        continue
                else:
                    renew_task.cancel()
                    await asyncio.gather(renew_task, return_exceptions=True)
                    renew_task = None
                if self._cancelled:
                    await _cancel_local_tasks(child_task)
                    raise asyncio.CancelledError
                return current_lease, _read_child_result(child_task, execution_input)
        except asyncio.CancelledError:
            child.cancel()
            await _cancel_local_tasks(child_task, renew_task)
            raise
        except BaseException:
            child.cancel()
            await _cancel_local_tasks(child_task, renew_task)
            raise

    async def _renew(self, lease: TaskLease) -> "TaskLease | TaskNodeView":
        if self._activity is not None:
            return await self._activity.renew(lease)
        if _temporal_workflow is None:
            raise RuntimeError(
                "Temporal SDK is required for production workflow execution"
            )
        await _temporal_workflow.sleep(30)
        return cast(
            "TaskLease | TaskNodeView",
            await _execute_task_activity("task_node_renew", lease),
        )

    def _start_child(
        self,
        execution_input: ExecutionWorkflowInput,
        request: TaskWorkflowInput,
        node: TaskWorkflowNode,
    ) -> _ChildWorkflowHandle:
        if (
            _temporal_workflow is None
            or _TemporalRetryPolicy is None
            or _TemporalWorkflowIDReusePolicy is None
        ):
            raise RuntimeError(
                "Temporal SDK is required for production workflow execution"
            )
        del request, node
        workflow_id = "task-node-" + execution_input.operation_id
        return cast(
            _ChildWorkflowHandle,
            _temporal_workflow.start_child_workflow(
                ExecutionWorkflow,
                execution_input,
                id=workflow_id,
                parent_close_policy=_temporal_workflow.ParentClosePolicy.TERMINATE,
                id_reuse_policy=_TemporalWorkflowIDReusePolicy.ALLOW_DUPLICATE,
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


async def _cancel_local_tasks(
    *tasks: "asyncio.Task[object] | None",
) -> None:
    pending = tuple(task for task in tasks if task is not None)
    for task in pending:
        if not task.done():
            task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _read_child_result(
    child_task: "asyncio.Task[ExecutionWorkflowResult]",
    execution_input: ExecutionWorkflowInput,
) -> ExecutionWorkflowResult:
    try:
        return child_task.result()
    except asyncio.CancelledError:
        return ExecutionWorkflowResult(
            execution_input.execution_id,
            "CANCELLED",
            None,
            0,
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        return ExecutionWorkflowResult(
            execution_input.execution_id,
            "FAILED",
            None,
            0,
        )


async def _wait_for_foreign_lease(node: TaskNodeView) -> None:
    if _temporal_workflow is None or node.lease_expires_at is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    delay = max(
        (node.lease_expires_at - _temporal_workflow.now()).total_seconds(),
        1.0,
    )
    await _temporal_workflow.sleep(delay)


def _task_node(node: TaskWorkflowNode) -> TaskNode:
    return TaskNode(
        node.node_id,
        node.dependencies,
        input=cast(Mapping[str, JsonValue], json.loads(node.input)),
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
