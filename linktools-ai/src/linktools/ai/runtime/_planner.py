#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of Agent-backed generic TaskNodes."""

import asyncio
import json
from collections.abc import Mapping

from linktools.core import environ

from ..agent import AgentCompiler, AgentDefinition
from ..core import (
    ExecutionStatus,
    Principal,
    canonical_sha256,
    principal_identity_payload,
    validate_agent_id,
    validate_user_prompt,
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
from .service_api import (
    CancelExecutionRequest,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    WorkflowGateway,
)

_logger = environ.get_logger("ai.runtime.planner")
_AGENT_TASK_FIELDS = frozenset({"type", "version", "agent_id", "binding_digest", "user_prompt"})


class RuntimeTaskNodeRunner:
    """Prepare and execute an admitted Agent-backed TaskNode."""

    def __init__(
        self,
        execution: ExecutionService,
        definitions: dict[str, AgentDefinition],
        compiler: AgentCompiler,
    ) -> None:
        self._execution = execution
        self._definitions = definitions
        self._compiler = compiler

    async def prepare(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> "tuple[str, ExecutionRequest]":
        payload = node.input
        if set(payload) != _AGENT_TASK_FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if payload["type"] != "linktools.ai.agent" or payload["version"] != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        agent_id = payload["agent_id"]
        binding_digest = payload["binding_digest"]
        base_user_prompt = payload["user_prompt"]
        if (
            not isinstance(agent_id, str)
            or not isinstance(binding_digest, str)
            or not isinstance(base_user_prompt, str)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        validate_agent_id(agent_id)
        validate_user_prompt(base_user_prompt)
        definition = self._definitions.get(binding_digest)
        if definition is not None:
            if definition.digest != binding_digest or definition.spec.id != agent_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            try:
                definition = await self._compiler.compile(agent_id=agent_id)
            except AIError as error:
                if error.code is ErrorCode.AGENT_NOT_FOUND:
                    raise AIError(
                        ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                        safe_details={
                            "agent_id": agent_id,
                            "binding_digest": binding_digest,
                        },
                    ) from error
                raise
            if definition.digest != binding_digest:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            self._definitions[binding_digest] = definition
            _logger.info("task Agent definition rehydrated: agent=%s digest=%s", agent_id, binding_digest)
        if set(dependency_results) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        dependency_payload: dict[str, object] = {}
        for dependency_id in sorted(node.dependencies):
            dependency = dependency_results[dependency_id]
            if dependency.execution_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await self._execution.result(dependency.execution_id, principal=principal)
            _validate_dependency_result(result, dependency.result_digest)
            dependency_payload[dependency_id] = result.output
        if dependency_payload:
            effective_user_prompt = (
                base_user_prompt
                + "\n\nUpstream task results (JSON, keyed by task id):\n"
                + _canonical_json(dependency_payload)
            )
        else:
            effective_user_prompt = base_user_prompt
        validate_user_prompt(effective_user_prompt)
        idempotency_key = canonical_sha256(
            {
                "version": 2,
                "graph_id": graph_id,
                "node_id": node.node_id,
                "agent_id": agent_id,
                "binding_digest": binding_digest,
                "input": node.input,
                "dependencies": [
                    {
                        "node_id": dependency_id,
                        "result_digest": dependency_results[dependency_id].result_digest,
                    }
                    for dependency_id in sorted(node.dependencies)
                ],
                "principal": principal_identity_payload(principal),
            }
        )
        request = ExecutionRequest(
            user_prompt=effective_user_prompt,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=None,
        )
        return binding_digest, request

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> TaskNodeRunResult:
        result = await self._execution.result(execution_id, principal=principal)
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.EXECUTION_FAILED)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNodeRunResult(canonical_sha256(result.output), result.execution_id)

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> TaskNodeRunResult:
        binding_digest, request = await self.prepare(
            node,
            graph_id=graph_id,
            principal=principal,
            dependency_results=dependency_results,
        )
        launch_task = asyncio.create_task(self._execution.run(binding_digest, request))
        try:
            handle = await asyncio.shield(launch_task)
        except asyncio.CancelledError as cancellation:
            try:
                handle = await asyncio.shield(launch_task)
            except BaseException as error:
                _logger.warning(
                    "task execution launch failed during cancellation: graph=%s task=%s error=%s",
                    graph_id,
                    node.node_id,
                    type(error).__name__,
                )
                raise cancellation
            await _cancel_execution(self._execution, handle.execution_id, principal, graph_id, node.node_id)
            raise cancellation
        wait_task = asyncio.create_task(self._execution.wait(handle.execution_id, principal=principal))
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError as cancellation:
            await _cancel_execution(self._execution, handle.execution_id, principal, graph_id, node.node_id)
            if not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            raise cancellation
        return await self.result(handle.execution_id, principal=principal)


class WorkflowTaskGraphLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, request: TaskGraphRequest) -> TaskGraphHandle:
        workflow_id = "task-" + canonical_sha256(
            {
                "tenant_id": request.principal.tenant_id,
                "graph_id": request.graph.graph_id,
            }
        )
        return await self._gateway.start_task_graph(workflow_id, request)

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        workflow_id = "task-" + canonical_sha256(
            {
                "tenant_id": request.principal.tenant_id,
                "graph_id": graph_id,
            }
        )
        return await self._gateway.cancel_task_graph(workflow_id, request.idempotency_key)


def _validate_dependency_result(result: ExecutionResult, expected_digest: str) -> None:
    if result.status is not ExecutionStatus.SUCCEEDED or result.output is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if canonical_sha256(result.output) != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
        _logger.warning(
            "task execution cancellation cleanup failed: graph=%s task=%s execution=%s",
            graph_id,
            node_id,
            execution_id,
        )


__all__ = ["DefaultTaskService", "RuntimeTaskNodeRunner", "WorkflowTaskGraphLauncher"]
