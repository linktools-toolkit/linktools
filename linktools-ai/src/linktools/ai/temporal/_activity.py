#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal activity adapters with stable registered names."""

from dataclasses import dataclass
from typing import Protocol, cast

from linktools.core import environ

from .workflow import (
    EvaluationWorkflowInput,
    EvaluationWorkflowResult,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
    ExecutionWorkflowState,
    SessionWorkflowInput,
    SessionWorkflowResult,
    TaskWorkflowInput,
    TaskWorkflowResult,
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
    async def execute(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult: ...


class ExecutionStageOperation(Protocol):
    async def load_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def fix_bundle_route(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def fix_binding(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def load_prompt(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def reserve_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def run_agent(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def process_deferred(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def commit_result(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...
    async def settle_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState: ...


class ExecuteActivity:
    options = ActivityOptions(start_to_close_seconds=300, retry_max_attempts=3, heartbeat_timeout_seconds=30)

    def __init__(self, operation: ExecutionOperation) -> None:
        self._operation = operation

    async def run(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult:
        _logger.info("executing durable activity: execution_id=%s", request.execution_id)
        result = await self._operation.execute(request)
        _logger.info("durable activity completed: execution_id=%s status=%s", request.execution_id, result.status)
        return result

    async def load_input(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).load_input(state)

    async def fix_bundle_route(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).fix_bundle_route(state)

    async def fix_binding(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).fix_binding(state)

    async def load_prompt(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).load_prompt(state)

    async def reserve_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).reserve_budget(state)

    async def run_agent(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).run_agent(state)

    async def process_deferred(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).process_deferred(state)

    async def commit_result(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).commit_result(state)

    async def settle_budget(self, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
        return await cast(ExecutionStageOperation, self._operation).settle_budget(state)


class EvaluationOperation(Protocol):
    async def execute(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult: ...


class EvaluationActivity:
    options = ActivityOptions(start_to_close_seconds=1800, retry_max_attempts=2, heartbeat_timeout_seconds=60)

    def __init__(self, operation: EvaluationOperation) -> None:
        self._operation = operation

    async def run(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
        _logger.info("executing evaluation activity: evaluation_id=%s", request.evaluation_id)
        return await self._operation.execute(request)


class SessionOperation(Protocol):
    async def execute(self, request: SessionWorkflowInput) -> SessionWorkflowResult: ...


class SessionActivity:
    options = ActivityOptions(start_to_close_seconds=60, retry_max_attempts=3, heartbeat_timeout_seconds=15)

    def __init__(self, operation: SessionOperation) -> None:
        self._operation = operation

    async def run(self, request: SessionWorkflowInput) -> SessionWorkflowResult:
        _logger.info("executing session activity: session_id=%s mutation_id=%s", request.session_id, request.mutation_id)
        return await self._operation.execute(request)


class TaskOperation(Protocol):
    async def execute(self, request: TaskWorkflowInput) -> TaskWorkflowResult: ...


class TaskActivity:
    options = ActivityOptions(start_to_close_seconds=900, retry_max_attempts=3, heartbeat_timeout_seconds=30)

    def __init__(self, operation: TaskOperation) -> None:
        self._operation = operation

    async def run(self, request: TaskWorkflowInput) -> TaskWorkflowResult:
        _logger.info("executing task activity: graph_id=%s", request.graph_id)
        return await self._operation.execute(request)


if _temporal_activity is not None:
    ExecuteActivity.run = _temporal_activity.defn(name="execute")(ExecuteActivity.run)
    ExecuteActivity.load_input = _temporal_activity.defn(name="load_input")(ExecuteActivity.load_input)
    ExecuteActivity.fix_bundle_route = _temporal_activity.defn(name="fix_bundle_route")(ExecuteActivity.fix_bundle_route)
    ExecuteActivity.fix_binding = _temporal_activity.defn(name="fix_binding")(ExecuteActivity.fix_binding)
    ExecuteActivity.load_prompt = _temporal_activity.defn(name="load_prompt")(ExecuteActivity.load_prompt)
    ExecuteActivity.reserve_budget = _temporal_activity.defn(name="reserve_budget")(ExecuteActivity.reserve_budget)
    ExecuteActivity.run_agent = _temporal_activity.defn(name="run_agent")(ExecuteActivity.run_agent)
    ExecuteActivity.process_deferred = _temporal_activity.defn(name="process_deferred")(ExecuteActivity.process_deferred)
    ExecuteActivity.commit_result = _temporal_activity.defn(name="commit_result")(ExecuteActivity.commit_result)
    ExecuteActivity.settle_budget = _temporal_activity.defn(name="settle_budget")(ExecuteActivity.settle_budget)
    EvaluationActivity.run = _temporal_activity.defn(name="evaluation")(EvaluationActivity.run)
    SessionActivity.run = _temporal_activity.defn(name="session_mutation")(SessionActivity.run)
    TaskActivity.run = _temporal_activity.defn(name="task_graph")(TaskActivity.run)


__all__ = [
    "ActivityOptions", "EvaluationActivity", "EvaluationOperation", "ExecuteActivity",
    "ExecutionOperation", "ExecutionStageOperation", "SessionActivity", "SessionOperation",
    "TaskActivity", "TaskOperation",
]
