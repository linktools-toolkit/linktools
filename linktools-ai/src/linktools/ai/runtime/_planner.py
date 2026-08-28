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
)
from ..core import (
    ExecutionStatus,
    Principal,
    canonical_sha256,
    normalize_execution_mode,
    normalize_thinking,
    principal_identity_payload,
    validate_agent_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..task import DefaultTaskService, TaskDependencyResult, TaskNode, TaskNodeRunResult
from ._input import user_prompt_transport
from .service_api import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
)

_logger = environ.get_logger("ai.runtime.planner")
_AGENT_TASK_FIELDS = frozenset(
    {
        "type",
        "version",
        "binding",
        "user_prompt",
        "user_prompt_codec",
        "mode",
        "planning",
        "thinking",
    }
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
        self._active_execution_ids: dict[tuple[str, str, str], str] = {}
        self._active_launch_tasks: dict[
            tuple[str, str, str], asyncio.Task[ExecutionHandle]
        ] = {}

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
        if set(payload) != _AGENT_TASK_FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if payload["type"] != "linktools.ai.agent" or payload["version"] != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        base_user_prompt = payload["user_prompt"]
        user_prompt_codec = payload["user_prompt_codec"]
        mode = payload["mode"]
        planning = payload["planning"]
        thinking = payload["thinking"]
        if (
            not isinstance(base_user_prompt, str)
            or not isinstance(user_prompt_codec, str)
            or not isinstance(planning, bool)
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
            mode = normalize_execution_mode(mode)
            thinking = normalize_thinking(thinking)
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if mode != "run":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            snapshot = AgentBindingSnapshot.from_payload(payload["binding"])
            binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        except AIError as error:
            if error.code in {
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            }:
                raise
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={
                    "binding_digest": (
                        snapshot.binding_digest if "snapshot" in locals() else None
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
            user_prompt=str(effective_user_prompt),
            user_prompt_codec=effective_user_prompt.codec,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=None,
            mode=mode,
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
            node, graph_id=graph_id, principal=principal, dependency_results=dependency_results
        )
        key = (principal.tenant_id, graph_id, node.node_id)
        launch_task = asyncio.create_task(
            self._execution.run(binding_digest, request),
            name=f"task-execution-launch-{graph_id}-{node.node_id}",
        )
        self._active_launch_tasks[key] = launch_task
        try:
            handle = await asyncio.shield(launch_task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._observe_after_launch(launch_task, key, graph_id, node.node_id),
                name=f"task-execution-launch-handoff-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                "task execution launch handoff",
            )
            raise
        except BaseException:
            if self._active_launch_tasks.get(key) is launch_task:
                self._active_launch_tasks.pop(key, None)
            raise
        if self._active_launch_tasks.get(key) is launch_task:
            self._active_launch_tasks.pop(key, None)
        self._active_execution_ids[key] = handle.execution_id
        wait_task = asyncio.create_task(
            self._execution.wait(handle.execution_id, principal=principal),
            name=f"task-execution-wait-{graph_id}-{node.node_id}",
        )
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            if not wait_task.done():
                wait_task.cancel()
                self._detach(
                    cast("asyncio.Task[object]", wait_task),
                    "task execution wait handoff",
                )
            else:
                self._consume_done(
                    cast("asyncio.Task[object]", wait_task),
                    "task execution wait handoff",
                )
            raise
        try:
            return await self.result(handle.execution_id, principal=principal)
        finally:
            self._active_execution_ids.pop(key, None)

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> None:
        key = (principal.tenant_id, graph_id, node.node_id)
        execution_id = self._active_execution_ids.get(key)
        binding_digest: str | None = None
        request: ExecutionRequest | None = None

        if execution_id is None:
            launch_task = self._active_launch_tasks.get(key)
            if launch_task is not None:
                try:
                    handle = await asyncio.shield(launch_task)
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    handle = None
                finally:
                    if launch_task.done() and self._active_launch_tasks.get(key) is launch_task:
                        self._active_launch_tasks.pop(key, None)
                if handle is not None and handle.execution_id:
                    execution_id = handle.execution_id
                    self._active_execution_ids[key] = execution_id

        if execution_id is None:
            binding_digest, request = await self.prepare(
                node,
                graph_id=graph_id,
                principal=principal,
                dependency_results=dependency_results,
            )
            try:
                handle = await self._execution.resolve_existing(
                    binding_digest,
                    request,
                )
            except asyncio.CancelledError:
                raise
            except AIError as error:
                if error.code is ErrorCode.EXECUTION_START_UNKNOWN:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
                raise
            except BaseException as error:  # noqa: BLE001
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            if handle is None:
                self._active_execution_ids.pop(key, None)
                return
            if not handle.execution_id:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            execution_id = handle.execution_id
            self._active_execution_ids[key] = execution_id

        try:
            await _cancel_execution(
                self._execution,
                execution_id,
                principal,
                graph_id,
                node.node_id,
            )
        finally:
            self._active_execution_ids.pop(key, None)

    async def _observe_after_launch(
        self,
        launch_task: "asyncio.Task[ExecutionHandle]",
        key: tuple[str, str, str],
        graph_id: str,
        node_id: str,
    ) -> None:
        try:
            handle = await launch_task
        except asyncio.CancelledError:
            return
        except BaseException as error:  # noqa: BLE001
            _logger.warning(
                "task execution launch failed during ownership handoff: graph=%s task=%s error=%s",
                graph_id,
                node_id,
                type(error).__name__,
            )
            return
        finally:
            if self._active_launch_tasks.get(key) is launch_task:
                self._active_launch_tasks.pop(key, None)
        if not handle.execution_id:
            _logger.error(
                "task execution launch returned invalid handle during ownership handoff: graph=%s task=%s",
                graph_id,
                node_id,
            )
            return
        self._active_execution_ids[key] = handle.execution_id

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
]
