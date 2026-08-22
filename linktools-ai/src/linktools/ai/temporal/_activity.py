#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal activity adapters with stable registered names."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ..runtime import RuntimeTaskNodeRunner
from ..runtime.state import TaskState
from ..storage import ObjectStore
from ..task import TaskDependencyResult, TaskLease, TaskNodeView
from .workflow import (
    EvaluationWorkflowInput,
    EvaluationWorkflowResult,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
    ExecutionWorkflowState,
    SessionWorkflowInput,
    SessionWorkflowResult,
    TaskWorkflowInput,
)

try:
    from temporalio import activity as _temporal_activity
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_activity = None

_logger = environ.get_logger("ai.temporal.activity")


@dataclass(frozen=True, slots=True)
class ActivityOptions:
    start_to_close_seconds: int
    retry_max_attempts: int
    heartbeat_timeout_seconds: int


class ExecutionOperation(Protocol):
    async def load_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def reserve_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def run_agent(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def persist_deferred(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def load_resume_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def commit_result(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def settle_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def cancel_effect(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...


class ExecuteActivity:
    options = ActivityOptions(300, 3, 30)

    def __init__(self, operation: ExecutionOperation) -> None:
        self._operation = operation

    async def load_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.load_input(state)

    async def reserve_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.reserve_budget(state)

    async def run_agent(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.run_agent(state)

    async def persist_deferred(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.persist_deferred(state)

    async def load_resume_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.load_resume_input(state)

    async def commit_result(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.commit_result(state)

    async def settle_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.settle_budget(state)

    async def cancel_effect(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await self._operation.cancel_effect(state)


class EvaluationOperation(Protocol):
    async def execute(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult: ...


class EvaluationActivity:
    options = ActivityOptions(1800, 2, 60)

    def __init__(self, operation: EvaluationOperation) -> None:
        self._operation = operation

    async def run(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
        _logger.debug("executing evaluation activity: evaluation_id=%s", request.evaluation_id)
        return await self._operation.execute(request)


class SessionOperation(Protocol):
    async def execute(self, request: SessionWorkflowInput) -> SessionWorkflowResult: ...


class SessionActivity:
    options = ActivityOptions(60, 3, 15)

    def __init__(self, operation: SessionOperation) -> None:
        self._operation = operation

    async def run(self, request: SessionWorkflowInput) -> SessionWorkflowResult:
        _logger.debug(
            "executing session activity: session_id=%s operation_id=%s",
            request.session_id,
            request.operation_id,
        )
        return await self._operation.execute(request)


class TaskOperation(Protocol):
    async def prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
        *,
        workflow_run_id: str,
    ) -> "TaskNodeView | tuple[TaskLease, ExecutionWorkflowInput]": ...

    async def renew(self, lease: TaskLease) -> "TaskLease | TaskNodeView": ...

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskNodeView: ...


class TaskActivity:
    options = ActivityOptions(60, 3, 15)

    def __init__(self, operation: TaskOperation) -> None:
        self._operation = operation

    @classmethod
    def from_runtime(
        cls,
        *,
        task_state: TaskState,
        runner: RuntimeTaskNodeRunner,
        request_store: ObjectStore,
        namespace: str,
    ) -> "TaskActivity":
        from ._task_operation import _RuntimeTaskOperation

        return cls(
            _RuntimeTaskOperation(
                task_state=task_state,
                runner=runner,
                request_store=request_store,
                namespace=namespace,
            )
        )

    async def prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> "TaskNodeView | tuple[TaskLease, ExecutionWorkflowInput]":
        _logger.debug("preparing task node: graph_id=%s node=%s", request.graph_id, node_id)
        return await self._operation.prepare(
            request,
            node_id,
            dependency_results,
            workflow_run_id=_workflow_run_id(),
        )

    async def renew(self, lease: TaskLease) -> "TaskLease | TaskNodeView":
        _logger.debug(
            "renewing task node lease: graph_id=%s node=%s fence=%s",
            lease.graph_id,
            lease.node_id,
            lease.fence,
        )
        return await self._operation.renew(lease)

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskNodeView:
        _logger.debug(
            "settling task node: graph_id=%s node=%s status=%s",
            request.graph_id,
            lease.node_id,
            result.status,
        )
        return await self._operation.settle(request, lease, result)


def _workflow_run_id() -> str:
    if _temporal_activity is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    try:
        workflow_run_id = _temporal_activity.info().workflow_run_id
    except (AttributeError, RuntimeError) as error:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
    if not isinstance(workflow_run_id, str) or not workflow_run_id.strip():
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return workflow_run_id


if _temporal_activity is not None:
    ExecuteActivity.load_input = _temporal_activity.defn(name="load_input")(ExecuteActivity.load_input)
    ExecuteActivity.reserve_budget = _temporal_activity.defn(name="reserve_budget")(ExecuteActivity.reserve_budget)
    ExecuteActivity.run_agent = _temporal_activity.defn(name="run_agent")(ExecuteActivity.run_agent)
    ExecuteActivity.persist_deferred = _temporal_activity.defn(name="persist_deferred")(ExecuteActivity.persist_deferred)
    ExecuteActivity.load_resume_input = _temporal_activity.defn(name="load_resume_input")(ExecuteActivity.load_resume_input)
    ExecuteActivity.commit_result = _temporal_activity.defn(name="commit_result")(ExecuteActivity.commit_result)
    ExecuteActivity.settle_budget = _temporal_activity.defn(name="settle_budget")(ExecuteActivity.settle_budget)
    ExecuteActivity.cancel_effect = _temporal_activity.defn(name="cancel_effect")(ExecuteActivity.cancel_effect)
    EvaluationActivity.run = _temporal_activity.defn(name="evaluation")(EvaluationActivity.run)
    SessionActivity.run = _temporal_activity.defn(name="session_mutation")(SessionActivity.run)
    TaskActivity.prepare = _temporal_activity.defn(name="task_node_prepare")(TaskActivity.prepare)
    TaskActivity.renew = _temporal_activity.defn(name="task_node_renew")(TaskActivity.renew)
    TaskActivity.settle = _temporal_activity.defn(name="task_node_settle")(TaskActivity.settle)


__all__ = [
    "ActivityOptions",
    "EvaluationActivity",
    "EvaluationOperation",
    "ExecuteActivity",
    "ExecutionOperation",
    "SessionActivity",
    "SessionOperation",
    "TaskActivity",
    "TaskOperation",
]
