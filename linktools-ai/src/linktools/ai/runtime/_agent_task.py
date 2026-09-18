#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime adapter for Agent-backed TaskGraph nodes."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

from linktools.core import environ

from ..agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler
from ..core import (
    CorrelationData,
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Principal,
    ThinkingValue,
    canonical_sha256,
    normalize_execution_mode,
    normalize_thinking,
    principal_identity_payload,
    validate_agent_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..task import TaskDependency, TaskNode, TaskNodeRunControl, TaskNodeRunError
from ._input import decode_user_content_payload
from .service_api import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ResumeSessionRequest,
    SessionService,
)

_logger = environ.get_logger("ai.runtime.planner")
_AGENT_TASK_TYPE = "linktools.ai.agent"
_AGENT_TASK_VERSION = 1
_AGENT_BODY_FIELDS = frozenset(
    {
        "binding",
        "user_prompt",
        "mode",
        "planning",
        "thinking",
        "files",
        "session_id",
        "memory_scope",
    }
)


class _AgentTaskNodeHandler:
    type = _AGENT_TASK_TYPE
    version = _AGENT_TASK_VERSION

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        session: SessionService | None = None,
        release_dependency_hold: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self._execution = execution
        self._session = session
        self._catalog = catalog
        self._compiler = compiler
        self._release_dependency_hold = (
            _noop_async_callback
            if release_dependency_hold is None
            else release_dependency_hold
        )
        self._detached_tasks: set[asyncio.Task[object]] = set()
        self._cancelled_tasks: set[asyncio.Task[object]] = set()
        self._active_launch_tasks: dict[
            tuple[str, str, str], asyncio.Task[ExecutionHandle]
        ] = {}
        self._background_failures: dict[tuple[str, str, str], AIError] = {}

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        active = tuple(
            cast("asyncio.Task[object]", task)
            for task in self._active_launch_tasks.values()
            if not task.done()
        )
        detached = tuple(task for task in self._detached_tasks if not task.done())
        return (*active, *detached)

    @property
    def pending_cancelled_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return tuple(task for task in self._cancelled_tasks if not task.done())

    @property
    def background_failure(self) -> AIError | None:
        if not self._background_failures:
            return None
        failure = next(iter(self._background_failures.values()))
        return AIError(
            failure.code,
            category=failure.category,
            retryable=failure.retryable,
            operation_id=failure.operation_id,
            safe_details=dict(failure.safe_details),
            diagnostics=failure.diagnostics,
        )

    def normalize(self, input: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if set(input) != _AGENT_BODY_FIELDS:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        raw_user_prompt = input.get("user_prompt")
        mode = input.get("mode")
        planning = input.get("planning")
        thinking = input.get("thinking")
        if not isinstance(raw_user_prompt, Mapping) or not isinstance(planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        files = input.get("files")
        session_id = input.get("session_id")
        memory_scope = input.get("memory_scope")
        if (
            not isinstance(files, list)
            or any(not isinstance(value, str) or not value for value in files)
            or (session_id is not None and not isinstance(session_id, str))
            or (memory_scope is not None and not isinstance(memory_scope, str))
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            kind = raw_user_prompt.get("kind")
            if kind == "text":
                if set(raw_user_prompt) != {"kind", "text"}:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                base_user_prompt: str | tuple[object, ...] = cast(
                    str, raw_user_prompt["text"]
                )
            elif kind == "pydantic-user-content-v1":
                if set(raw_user_prompt) != {"kind", "value"}:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                value = raw_user_prompt.get("value")
                if not isinstance(value, Mapping):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                base_user_prompt = decode_user_content_payload(value)
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            resolved_mode = normalize_execution_mode(mode)
            resolved_thinking = normalize_thinking(thinking)
            snapshot = AgentBindingSnapshot.from_payload(input.get("binding"))
            binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if resolved_mode != "run":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if binding.snapshot != snapshot or binding.digest != snapshot.binding_digest:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_agent_id(binding.definition.spec.id)
        if isinstance(base_user_prompt, str):
            validate_user_prompt(base_user_prompt)
        return {
            "binding": binding.snapshot.to_payload(),
            "user_prompt": raw_user_prompt,
            "mode": "run",
            "planning": planning,
            "thinking": resolved_thinking,
            "files": list(files),
            "session_id": session_id,
            "memory_scope": memory_scope,
        }

    def validate_recovery(
        self,
        input: Mapping[str, JsonValue],
        *,
        graph_id: str,
        node_id: str,
    ) -> Mapping[str, JsonValue]:
        try:
            return self.normalize(input)
        except AIError as error:
            cause = error.__cause__
            if isinstance(cause, AIError) and cause.code in {
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            }:
                raise cause
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "task_type": self.type,
                    "task_version": self.version,
                },
            ) from error

    async def run_node(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        correlation: CorrelationData,
        dependencies: Mapping[str, TaskDependency],
        control: TaskNodeRunControl,
    ) -> tuple[JsonValue, str]:
        prepared = self._prepare_request(
            node,
            graph_id=graph_id,
            principal=principal,
            correlation=correlation,
            dependencies=dependencies,
        )
        binding_digest, request = prepared[:2]
        agent_id = prepared[2] if len(prepared) > 2 else ""
        session_id = prepared[3] if len(prepared) > 3 else None
        key = (principal.tenant_id, graph_id, node.node_id)
        hold_id = f"task:{graph_id}:{node.node_id}"
        if session_id is None or self._session is None:
            launch = self._execution.start(
                binding_digest,
                request,
                dependency_hold_id=hold_id,
            )
        else:
            launch = self._session.resume(
                agent_id,
                binding_digest,
                session_id,
                ResumeSessionRequest(
                    request.principal,
                    request.user_prompt,
                    request.idempotency_key,
                    request.memory_scope,
                    request.mode,
                    request.planning,
                    request.thinking,
                    request.correlation,
                    request.files,
                ),
            )
        launch_task = asyncio.create_task(
            launch,
            name=f"task-execution-launch-{graph_id}-{node.node_id}",
        )
        self._active_launch_tasks[key] = launch_task
        try:
            handle = await asyncio.shield(launch_task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._handoff_after_launch(
                    launch_task,
                    key,
                    control,
                ),
                name=f"task-execution-handoff-after-launch-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                (
                    "task execution handoff after launch "
                    f"graph={graph_id} task={node.node_id}"
                ),
            )
            raise
        finally:
            if launch_task.done() and self._active_launch_tasks.get(key) is launch_task:
                self._active_launch_tasks.pop(key, None)
        if not handle.execution_id:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        await self._handoff_execution(
            control,
            handle.execution_id,
            key=key,
        )
        wait_task = asyncio.create_task(
            self._execution.wait(handle.execution_id, principal=principal),
            name=f"task-execution-wait-{graph_id}-{node.node_id}",
        )
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            if not wait_task.done():
                wait_task.cancel()
                self._detach_cancelled(
                    cast("asyncio.Task[object]", wait_task),
                    f"task execution wait cleanup graph={graph_id} task={node.node_id}",
                )
            else:
                self._consume_done(
                    cast("asyncio.Task[object]", wait_task),
                    f"task execution wait cleanup graph={graph_id} task={node.node_id}",
                )
            raise
        result = await self._execution.result(handle.execution_id, principal=principal)
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result.output, result.execution_id

    async def read_result(
        self,
        execution_id: str,
        *,
        principal: Principal,
        expected_digest: str,
    ) -> JsonValue:
        result = await self._execution.result(execution_id, principal=principal)
        _validate_dependency_result(result, expected_digest)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result.output

    async def cancel_node(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        correlation: CorrelationData,
        dependencies: Mapping[str, TaskDependency],
        durable_execution_id: str | None,
    ) -> None:
        key = (principal.tenant_id, graph_id, node.node_id)
        execution_id = durable_execution_id
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
                    if (
                        launch_task.done()
                        and self._active_launch_tasks.get(key) is launch_task
                    ):
                        self._active_launch_tasks.pop(key, None)
                if handle is not None and handle.execution_id:
                    execution_id = handle.execution_id
        if execution_id is None:
            binding_digest, request, _, _ = self._prepare_request(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
                dependencies=dependencies,
            )
            try:
                handle = await self._execution.resolve_existing(binding_digest, request)
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                raise self._record_background_failure(
                    key,
                    error,
                    phase="task_execution_resolve_cancel",
                ) from error
            if handle is None:
                self._background_failures.pop(key, None)
                return
            if not handle.execution_id:
                raise self._record_background_failure(
                    key,
                    AIError(ErrorCode.EXECUTION_START_UNKNOWN),
                    phase="task_execution_resolve_cancel",
                )
            execution_id = handle.execution_id
        await _cancel_execution(
            self._execution,
            execution_id,
            principal,
            graph_id,
            node.node_id,
        )
        self._background_failures.pop(key, None)

    def _prepare_request(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        correlation: CorrelationData,
        dependencies: Mapping[str, TaskDependency],
    ) -> tuple[str, ExecutionRequest, str, str | None]:
        payload = node.input
        if payload.get("type") != self.type or payload.get("version") != self.version:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        body = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "version"}
        }
        normalized = self.validate_recovery(
            body,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if normalized != body:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        snapshot = AgentBindingSnapshot.from_payload(normalized["binding"])
        binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        raw_user_prompt = cast(Mapping[str, JsonValue], normalized["user_prompt"])
        if raw_user_prompt.get("kind") == "text":
            base_user_prompt: str | tuple[object, ...] = cast(
                str, raw_user_prompt["text"]
            )
        else:
            value = raw_user_prompt.get("value")
            if not isinstance(value, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            base_user_prompt = decode_user_content_payload(value)
        dependency_payload = {
            dependency_id: dependencies[dependency_id].output
            for dependency_id in sorted(node.dependencies)
        }
        if dependency_payload:
            dependency_text = (
                "\n\nUpstream task results (JSON, keyed by task id):\n"
                + _canonical_json(dependency_payload)
            )
            effective_user_prompt = (
                base_user_prompt + dependency_text
                if isinstance(base_user_prompt, str)
                else (*base_user_prompt, dependency_text)
            )
        else:
            effective_user_prompt = base_user_prompt
        if isinstance(effective_user_prompt, str):
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
                        "result_digest": dependencies[dependency_id].result_digest,
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
            memory_scope=cast("str | None", normalized["memory_scope"]),
            mode=cast(ExecutionMode, normalized["mode"]),
            planning=cast(bool, normalized["planning"]),
            thinking=cast(ThinkingValue, normalized["thinking"]),
            correlation=correlation,
            files=tuple(cast(list[str], normalized["files"])),
        )
        return (
            binding.digest,
            request,
            binding.definition.spec.id,
            cast("str | None", normalized["session_id"]),
        )

    async def _handoff_execution(
        self,
        control: TaskNodeRunControl,
        execution_id: str,
        *,
        key: tuple[str, str, str],
    ) -> None:
        task = asyncio.create_task(
            control.handoff_execution(execution_id),
            name=f"task-execution-handoff-{key[1]}-{key[2]}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._settle_detached_handoff(
                    task,
                    execution_id,
                    key,
                ),
                name=f"task-execution-handoff-settle-{key[1]}-{key[2]}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                f"task execution handoff graph={key[1]} task={key[2]}",
            )
            raise
        except BaseException:
            await self._release_dependency_hold(
                execution_id,
                tenant_id=key[0],
                hold_id=f"task:{key[1]}:{key[2]}",
            )
            raise

    async def _settle_detached_handoff(
        self,
        task: asyncio.Task[None],
        execution_id: str,
        key: tuple[str, str, str],
    ) -> None:
        handoff_succeeded = False
        try:
            await task
            handoff_succeeded = True
            await self._release_dependency_hold(
                execution_id,
                tenant_id=key[0],
                hold_id=f"task:{key[1]}:{key[2]}",
            )
        except asyncio.CancelledError:
            if not handoff_succeeded:
                await self._release_dependency_hold(
                    execution_id,
                    tenant_id=key[0],
                    hold_id=f"task:{key[1]}:{key[2]}",
                )
            raise
        except AIError as error:
            if not handoff_succeeded:
                await self._release_dependency_hold(
                    execution_id,
                    tenant_id=key[0],
                    hold_id=f"task:{key[1]}:{key[2]}",
                )
            if error.code in {
                ErrorCode.TASK_FENCE_STALE,
                ErrorCode.TASK_OWNER_CONFLICT,
                ErrorCode.TASK_NOT_READY,
            }:
                return
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_handoff",
            ) from error
        except BaseException as error:  # noqa: BLE001
            if not handoff_succeeded:
                await self._release_dependency_hold(
                    execution_id,
                    tenant_id=key[0],
                    hold_id=f"task:{key[1]}:{key[2]}",
                )
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_handoff",
            ) from error

    async def _handoff_after_launch(
        self,
        launch_task: asyncio.Task[ExecutionHandle],
        key: tuple[str, str, str],
        control: TaskNodeRunControl,
    ) -> None:
        try:
            handle = await launch_task
            if not handle.execution_id:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            await self._handoff_execution(
                control,
                handle.execution_id,
                key=key,
            )
            await self._release_dependency_hold(
                handle.execution_id,
                tenant_id=key[0],
                hold_id=f"task:{key[1]}:{key[2]}",
            )
        except asyncio.CancelledError:
            raise
        except AIError as error:
            if error.code in {
                ErrorCode.TASK_FENCE_STALE,
                ErrorCode.TASK_OWNER_CONFLICT,
                ErrorCode.TASK_NOT_READY,
            }:
                return
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_handoff_after_launch",
            ) from error
        except BaseException as error:  # noqa: BLE001
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_handoff_after_launch",
            ) from error
        finally:
            if self._active_launch_tasks.get(key) is launch_task:
                self._active_launch_tasks.pop(key, None)

    def _record_background_failure(
        self,
        key: tuple[str, str, str],
        error: BaseException,
        *,
        phase: str,
    ) -> AIError:
        details = dict(error.safe_details) if isinstance(error, AIError) else {}
        details.setdefault("phase", phase)
        details.setdefault("graph_id", key[1])
        details.setdefault("node_id", key[2])
        if isinstance(error, AIError):
            failure = AIError(
                error.code,
                category=error.category,
                retryable=error.retryable,
                operation_id=error.operation_id,
                safe_details=details,
                diagnostics=error.diagnostics,
            )
        else:
            failure = AIError(ErrorCode.INTERNAL_ERROR, safe_details=details)
        self._background_failures[key] = failure
        return failure

    def _detach(self, task: asyncio.Task[object], label: str) -> None:
        if task.done():
            self._consume_done(task, label)
            return
        self._detached_tasks.add(task)

        def consume(done: asyncio.Task[object]) -> None:
            try:
                self._consume_done(done, label)
            finally:
                self._detached_tasks.discard(done)

        task.add_done_callback(consume)

    def _detach_cancelled(self, task: asyncio.Task[object], label: str) -> None:
        if task.done():
            self._consume_done(task, label)
            return
        self._cancelled_tasks.add(task)

        def consume(done: asyncio.Task[object]) -> None:
            try:
                self._consume_done(done, label)
            finally:
                self._cancelled_tasks.discard(done)

        task.add_done_callback(consume)

    @staticmethod
    def _consume_done(task: asyncio.Task[object], label: str) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)


async def _noop_async_callback(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _execution_failure(result: ExecutionResult) -> TaskNodeRunError:
    if result.status not in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if result.error_code is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(result.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return TaskNodeRunError(
        code,
        result.execution_id,
        safe_details=result.safe_error_details,
    )


def _validate_dependency_result(result: ExecutionResult, expected_digest: str) -> None:
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
        current = await execution.inspect(execution_id, principal=principal)
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
