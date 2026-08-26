#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of Agent-backed generic TaskNodes."""

import asyncio
import json
from collections.abc import Mapping
from typing import cast

from linktools.core import environ

from ..agent import (
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    user_prompt_transport,
)
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
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    WorkflowGateway,
)

_logger = environ.get_logger("ai.runtime.planner")
_AGENT_TASK_FIELDS = frozenset(
    {"type", "version", "binding", "user_prompt", "planning", "thinking"}
)


class RuntimeTaskNodeRunner:
    """Prepare and execute an admitted Agent-backed TaskNode."""

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
    ) -> None:
        self._execution = execution
        self._catalog = catalog
        self._compiler = compiler
        self._detached_tasks: set[asyncio.Task[object]] = set()

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return tuple(task for task in self._detached_tasks if not task.done())

    async def prepare(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> "tuple[str, ExecutionRequest]":
        payload = node.input
        if not _AGENT_TASK_FIELDS.issubset(payload):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if payload["type"] != "linktools.ai.agent" or payload["version"] != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        base_user_prompt = payload["user_prompt"]
        user_prompt_codec = payload.get("user_prompt_codec", "text")
        planning = payload["planning"]
        thinking = payload["thinking"]
        if (
            not isinstance(base_user_prompt, str)
            or not isinstance(user_prompt_codec, str)
            or not isinstance(planning, bool)
            or not isinstance(thinking, bool)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            base_user_prompt = user_prompt_transport(
                base_user_prompt,
                user_prompt_codec,
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        try:
            snapshot = AgentBindingSnapshot.from_payload(payload["binding"])
            binding = self._catalog.register_binding(
                self._compiler.restore(snapshot)
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={
                    "binding_digest": (
                        snapshot.binding_digest
                        if "snapshot" in locals()
                        else None
                    )
                },
            ) from error
        if binding.snapshot != snapshot or binding.digest != snapshot.binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        validate_agent_id(binding.definition.spec.id)
        validate_user_prompt(base_user_prompt)
        if set(dependency_results) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        dependency_payload: dict[str, object] = {}
        for dependency_id in sorted(node.dependencies):
            dependency = dependency_results[dependency_id]
            if dependency.execution_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await self._execution.result(
                dependency.execution_id,
                principal=principal,
            )
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
                "version": 1,
                "graph_id": graph_id,
                "node_id": node.node_id,
                "binding_digest": binding.digest,
                "input": node.input,
                "dependencies": [
                    {
                        "node_id": dependency_id,
                        "result_digest": dependency_results[
                            dependency_id
                        ].result_digest,
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
            planning=planning,
            thinking=thinking,
        )
        return binding.digest, request

    async def terminal_result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        return await self._execution.result(
            execution_id,
            principal=principal,
        )

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> TaskNodeRunResult:
        result = await self.terminal_result(
            execution_id,
            principal=principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNodeRunResult(
            canonical_sha256(result.output),
            result.execution_id,
        )

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
        launch_task = asyncio.create_task(
            self._execution.run(binding_digest, request),
            name=f"task-execution-launch-{graph_id}-{node.node_id}",
        )
        try:
            handle = await asyncio.shield(launch_task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._cancel_after_launch(
                    launch_task,
                    principal,
                    graph_id,
                    node.node_id,
                ),
                name=f"task-execution-launch-cleanup-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                "task execution launch cleanup",
            )
            raise
        wait_task = asyncio.create_task(
            self._execution.wait(
                handle.execution_id,
                principal=principal,
            ),
            name=f"task-execution-wait-{graph_id}-{node.node_id}",
        )
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            if not wait_task.done():
                wait_task.cancel()
                self._detach(
                    cast("asyncio.Task[object]", wait_task),
                    "task execution wait cleanup",
                )
            else:
                self._consume_done(
                    cast("asyncio.Task[object]", wait_task),
                    "task execution wait cleanup",
                )
            cleanup = asyncio.create_task(
                _cancel_execution(
                    self._execution,
                    handle.execution_id,
                    principal,
                    graph_id,
                    node.node_id,
                ),
                name=f"task-execution-cancel-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", cleanup),
                "task execution cancellation",
            )
            raise
        return await self.result(
            handle.execution_id,
            principal=principal,
        )

    async def _cancel_after_launch(
        self,
        launch_task: "asyncio.Task[ExecutionHandle]",
        principal: Principal,
        graph_id: str,
        node_id: str,
    ) -> None:
        try:
            handle = await launch_task
        except asyncio.CancelledError:
            return
        except BaseException as error:  # noqa: BLE001
            _logger.warning(
                "task execution launch failed during cancellation: "
                "graph=%s task=%s error=%s",
                graph_id,
                node_id,
                type(error).__name__,
            )
            return
        execution_id = handle.execution_id
        if not execution_id:
            _logger.error(
                "task execution launch returned invalid handle during cancellation: "
                "graph=%s task=%s",
                graph_id,
                node_id,
            )
            return
        try:
            await _cancel_execution(
                self._execution,
                execution_id,
                principal,
                graph_id,
                node_id,
            )
        except BaseException as error:  # noqa: BLE001
            _logger.warning(
                "task execution cancellation requires recovery: "
                "graph=%s task=%s error=%s",
                graph_id,
                node_id,
                type(error).__name__,
            )

    def _detach(
        self,
        task: "asyncio.Task[object]",
        label: str,
    ) -> None:
        if task.done():
            self._consume_done(task, label)
            return
        if task in self._detached_tasks:
            return
        self._detached_tasks.add(task)

        def consume(done: "asyncio.Task[object]") -> None:
            try:
                self._consume_done(done, label)
            finally:
                self._detached_tasks.discard(done)

        task.add_done_callback(consume)

    @staticmethod
    def _consume_done(
        task: "asyncio.Task[object]",
        label: str,
    ) -> None:
        if not task.done():
            return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)


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

    async def cancel(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        workflow_id = "task-" + canonical_sha256(
            {
                "tenant_id": request.principal.tenant_id,
                "graph_id": graph_id,
            }
        )
        return await self._gateway.cancel_task_graph(
            workflow_id,
            request.idempotency_key,
        )


def _execution_failure(result: ExecutionResult) -> AIError:
    if result.status not in {
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    }:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if result.error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(result.error_code)
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AIError(code, safe_details=result.safe_error_details)


def _validate_dependency_result(
    result: ExecutionResult,
    expected_digest: str,
) -> None:
    if result.status is not ExecutionStatus.SUCCEEDED or result.output is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if canonical_sha256(result.output) != expected_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


async def _cancel_execution(
    execution: ExecutionService,
    execution_id: str,
    principal: Principal,
    graph_id: str,
    node_id: str,
) -> None:
    request = CancelExecutionRequest(
        principal,
        canonical_sha256(
            {
                "task_graph": graph_id,
                "node_id": node_id,
                "execution_id": execution_id,
            }
        ),
    )
    try:
        result = await execution.cancel(execution_id, request)
    except asyncio.CancelledError:
        raise
    except BaseException as error:  # noqa: BLE001
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if result.cancelled:
        return
    try:
        current = await execution.inspect(
            execution_id,
            principal=principal,
        )
    except asyncio.CancelledError:
        raise
    except BaseException as error:  # noqa: BLE001
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if current.status not in {
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    }:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)


__all__ = [
    "DefaultTaskService",
    "RuntimeTaskNodeRunner",
    "WorkflowTaskGraphLauncher",
]
