#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution Activities with narrow, independently retryable stage ports."""

from dataclasses import dataclass
from typing import Protocol, cast

from linktools.core import environ

from ..workflow.execution import ExecutionWorkflowInput, ExecutionWorkflowResult, ExecutionWorkflowState

try:
    from temporalio import activity as _temporal_activity
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_activity = None

_logger = environ.get_logger("ai.temporal.activity.execution")


@dataclass(frozen=True, slots=True)
class ActivityOptions:
    start_to_close_seconds: int
    retry_max_attempts: int
    heartbeat_timeout_seconds: int


class ExecutionOperation(Protocol):
    async def execute(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult: ...


class ExecutionStageOperation(Protocol):
    async def load_input(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def fix_bundle_route(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def fix_binding(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def load_prompt(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def reserve_budget(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def run_agent(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def process_deferred(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def append_event(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def commit_result(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def settle_budget(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...
    async def append_terminal_event(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState": ...


class ExecuteActivity:
    options = ActivityOptions(start_to_close_seconds=300, retry_max_attempts=3, heartbeat_timeout_seconds=30)

    def __init__(self, operation: ExecutionOperation) -> None:
        self._operation = operation

    async def run(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult:
        _logger.info("executing durable activity: execution_id=%s", request.execution_id)
        result = await self._operation.execute(request)
        _logger.info("durable activity completed: execution_id=%s status=%s", request.execution_id, result.status)
        return result

    async def load_input(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).load_input(state)

    async def fix_bundle_route(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).fix_bundle_route(state)

    async def fix_binding(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).fix_binding(state)

    async def load_prompt(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).load_prompt(state)

    async def reserve_budget(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).reserve_budget(state)

    async def run_agent(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).run_agent(state)

    async def process_deferred(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).process_deferred(state)

    async def append_event(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).append_event(state)

    async def commit_result(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).commit_result(state)

    async def settle_budget(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).settle_budget(state)

    async def append_terminal_event(self, state: "ExecutionWorkflowState") -> "ExecutionWorkflowState":
        return await cast(ExecutionStageOperation, self._operation).append_terminal_event(state)


if _temporal_activity is not None:
    ExecuteActivity.run = _temporal_activity.defn(name="execute")(ExecuteActivity.run)
    ExecuteActivity.load_input = _temporal_activity.defn(name="load_input")(ExecuteActivity.load_input)
    ExecuteActivity.fix_bundle_route = _temporal_activity.defn(name="fix_bundle_route")(ExecuteActivity.fix_bundle_route)
    ExecuteActivity.fix_binding = _temporal_activity.defn(name="fix_binding")(ExecuteActivity.fix_binding)
    ExecuteActivity.load_prompt = _temporal_activity.defn(name="load_prompt")(ExecuteActivity.load_prompt)
    ExecuteActivity.reserve_budget = _temporal_activity.defn(name="reserve_budget")(ExecuteActivity.reserve_budget)
    ExecuteActivity.run_agent = _temporal_activity.defn(name="run_agent")(ExecuteActivity.run_agent)
    ExecuteActivity.process_deferred = _temporal_activity.defn(name="process_deferred")(ExecuteActivity.process_deferred)
    ExecuteActivity.append_event = _temporal_activity.defn(name="append_event")(ExecuteActivity.append_event)
    ExecuteActivity.commit_result = _temporal_activity.defn(name="commit_result")(ExecuteActivity.commit_result)
    ExecuteActivity.settle_budget = _temporal_activity.defn(name="settle_budget")(ExecuteActivity.settle_budget)
    ExecuteActivity.append_terminal_event = _temporal_activity.defn(name="append_terminal_event")(ExecuteActivity.append_terminal_event)


__all__ = ["ActivityOptions", "ExecuteActivity", "ExecutionOperation", "ExecutionStageOperation"]
