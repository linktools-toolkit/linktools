#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed task service used by the runtime composition root."""

import asyncio
from collections.abc import Callable, Mapping

from linktools.core import environ

from ..agent import AgentDefinition
from ..core import (
    ExecutionStatus,
    Principal,
    canonical_sha256,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..task import (
    CancelGraphRequest,
    DefaultTaskService,
    TaskDependencyResult,
    TaskGraphHandle,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
    TaskNodeRunResult,
)
from ._services import (
    CancelExecutionRequest,
    ExecutionRequest,
    ExecutionService,
    WorkflowGateway,
)

_logger = environ.get_logger("ai.runtime.planner")


class RuntimeTaskNodeRunner:
    """Run task nodes through the canonical ExecutionService boundary."""

    def __init__(
        self,
        execution: ExecutionService,
        definitions: "Mapping[str, AgentDefinition]",
        *,
        prompt_factory: "Callable[[TaskNode, Mapping[str, TaskDependencyResult]], str] | None" = None,
    ) -> None:
        self._execution = execution
        self._definitions = definitions
        self._prompt_factory = prompt_factory or _default_task_prompt

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> "TaskNodeRunResult":
        if node.binding_digest is None or node.binding_digest not in self._definitions:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        prompt = self._prompt_factory(node, dependency_results)
        if not isinstance(prompt, str) or not prompt.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        request = ExecutionRequest(
            prompt=prompt,
            principal=principal,
            idempotency_key=canonical_sha256({"graph_id": graph_id, "node_id": node.node_id, "definition": node.binding_digest, "principal": principal_identity_payload(principal)}),
            memory_scope=None,
        )
        launch_task = asyncio.create_task(self._execution.run(node.binding_digest, request))
        try:
            handle = await asyncio.shield(launch_task)
        except asyncio.CancelledError as cancellation:
            try:
                handle = await asyncio.shield(launch_task)
            except BaseException as error:
                _logger.warning("task execution launch failed during cancellation: graph=%s task=%s error=%s", graph_id, node.node_id, type(error).__name__)
                raise cancellation
            await _cancel_execution(self._execution, handle.execution_id, principal, graph_id, node.node_id)
            raise cancellation
        wait_task = asyncio.create_task(self._execution.wait(handle.execution_id, principal=principal))
        try:
            result = await asyncio.shield(wait_task)
        except asyncio.CancelledError as cancellation:
            await _cancel_execution(self._execution, handle.execution_id, principal, graph_id, node.node_id)
            if not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            raise cancellation
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.EXECUTION_FAILED)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNodeRunResult(canonical_sha256(result.output), result.execution_id)


class WorkflowTaskGraphLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, request: TaskGraphRequest) -> TaskGraphHandle:
        return await self._gateway.start_task_graph(request.graph.graph_id, request)

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        return await self._gateway.cancel_task_graph(graph_id, request.idempotency_key)


def _default_task_prompt(node: TaskNode, dependency_results: Mapping[str, TaskDependencyResult]) -> str:
    del dependency_results
    return node.node_id


async def _cancel_execution(
    execution: ExecutionService,
    execution_id: str,
    principal: Principal,
    graph_id: str,
    node_id: str,
) -> None:
    request = CancelExecutionRequest(
        principal,
        canonical_sha256({"task_graph": graph_id, "node_id": node_id, "execution_id": execution_id}),
    )
    cleanup = asyncio.create_task(execution.cancel(execution_id, request))
    try:
        await asyncio.shield(cleanup)
    except BaseException:
        _logger.warning("task execution cancellation cleanup failed: graph=%s task=%s execution=%s", graph_id, node_id, execution_id)


__all__ = ["DefaultTaskService", "RuntimeTaskNodeRunner", "WorkflowTaskGraphLauncher"]
