#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of generic TaskNodes."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai.messages import UserContent

from ..agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler, AgentDefinition
from ..capability import TaskExpander, TaskExpansionContext
from ..core import (
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Principal,
    CorrelationData,
    TaskStatus,
    ThinkingValue,
    canonical_json_bytes,
    canonical_sha256,
    normalize_execution_mode,
    normalize_json_value,
    normalize_thinking,
    principal_identity_payload,
    validate_agent_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ..task import (
    TaskDependency,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphSnapshot,
    TaskNode,
    TaskExpanderRef,
    TaskNodeContext,
    TaskNodeHandler,
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    TaskResultRecord,
)
from ._input import (
    CanonicalUserInput,
    decode_user_content_payload,
    task_prompt_draft,
    validate_user_input,
)
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .service_api import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
)
from .state import RuntimeDomain

_logger = environ.get_logger("ai.runtime.planner")
AppT = TypeVar("AppT")
_AGENT_TASK_TYPE = "linktools.ai.agent"
_AGENT_TASK_VERSION = 1
_TASK_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_EXPANDER_ID_PREFIX = "linktools.ai."
_AGENT_BODY_FIELDS = frozenset(
    {
        "binding",
        "user_prompt",
        "mode",
        "planning",
        "thinking",
    }
)


class _TaskStateReader(Protocol):
    async def snapshot_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot | None: ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]: ...


class _AgentTaskNodeHandler:
    type = _AGENT_TASK_TYPE
    version = _AGENT_TASK_VERSION

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        release_dependency_hold: Callable[..., Awaitable[None]] | None = None,
        request_terminal_handoff: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self._execution = execution
        self._catalog = catalog
        self._compiler = compiler
        self._release_dependency_hold = (
            _noop_async_callback
            if release_dependency_hold is None
            else release_dependency_hold
        )
        self._request_terminal_handoff = (
            _noop_async_callback
            if request_terminal_handoff is None
            else request_terminal_handoff
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
        if (
            not isinstance(raw_user_prompt, Mapping)
            or not isinstance(planning, bool)
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
        binding_digest, request = self._prepare_request(
            node,
            graph_id=graph_id,
            principal=principal,
            correlation=correlation,
            dependencies=dependencies,
        )
        key = (principal.tenant_id, graph_id, node.node_id)
        hold_id = f"task:{graph_id}:{node.node_id}"
        launch_task = asyncio.create_task(
            self._execution.start(
                binding_digest,
                request,
                dependency_hold_id=hold_id,
            ),
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
                    principal,
                ),
                name=f"task-execution-handoff-after-launch-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                f"task execution handoff after launch graph={graph_id} task={node.node_id}",
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
            principal=principal,
        )
        wait_task = asyncio.create_task(
            self._execution.wait(handle.execution_id, principal=principal),
            name=f"task-execution-wait-{graph_id}-{node.node_id}",
        )
        try:
            await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._finish_handoff_wait(
                    wait_task,
                    handle.execution_id,
                    key,
                    principal,
                ),
                name=f"task-execution-handoff-wait-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                f"task execution handoff wait graph={graph_id} task={node.node_id}",
            )
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
            binding_digest, request = self._prepare_request(
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
    ) -> tuple[str, ExecutionRequest]:
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
        return binding.digest, ExecutionRequest(
            user_prompt=effective_user_prompt,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=None,
            mode=cast(ExecutionMode, normalized["mode"]),
            planning=cast(bool, normalized["planning"]),
            thinking=cast(ThinkingValue, normalized["thinking"]),
            correlation=correlation,
        )

    async def _handoff_execution(
        self,
        control: TaskNodeRunControl,
        execution_id: str,
        *,
        key: tuple[str, str, str],
        principal: Principal,
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
                    principal,
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
        principal: Principal,
    ) -> None:
        handoff_succeeded = False
        try:
            await task
            handoff_succeeded = True
            wait_task = asyncio.create_task(
                self._execution.wait(execution_id, principal=principal),
                name=f"task-execution-detached-wait-{key[1]}-{key[2]}",
            )
            await self._finish_handoff_wait(
                wait_task,
                execution_id,
                key,
                principal,
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
        principal: Principal,
    ) -> None:
        try:
            handle = await launch_task
            if not handle.execution_id:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            await self._handoff_execution(
                control,
                handle.execution_id,
                key=key,
                principal=principal,
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

    async def _finish_handoff_wait(
        self,
        wait_task: asyncio.Task[ExecutionResult],
        execution_id: str,
        key: tuple[str, str, str],
        principal: Principal,
    ) -> None:
        try:
            await wait_task
            await self._execution.result(
                execution_id,
                principal=principal,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_handoff_wait",
            ) from error
        else:
            try:
                await self._finish_handoff(execution_id, key)
            except BaseException as error:  # noqa: BLE001
                raise self._record_background_failure(
                    key,
                    error,
                    phase="task_execution_handoff_release",
                ) from error

    async def _finish_handoff(
        self,
        execution_id: str,
        key: tuple[str, str, str],
    ) -> None:
        await self._release_dependency_hold(
            execution_id,
            tenant_id=key[0],
            hold_id=f"task:{key[1]}:{key[2]}",
        )
        await self._request_terminal_handoff(execution_id, tenant_id=key[0])

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


class _TaskExpansionContext:
    __slots__ = (
        "_build_agent_task",
        "_generated_agent_tasks",
        "_graph_id",
        "_output",
        "_principal",
        "_source_node",
    )

    def __init__(
        self,
        principal: Principal,
        graph_id: str,
        source_node: TaskNode,
        output: JsonValue,
        *,
        build_agent_task: Callable[..., TaskNode],
    ) -> None:
        self._principal = principal
        self._graph_id = graph_id
        self._source_node = source_node
        self._output = output
        self._build_agent_task = build_agent_task
        self._generated_agent_tasks: dict[str, TaskNode] = {}

    def agent_task(
        self,
        agent_id: str,
        node_id: str,
        user_prompt: str | Sequence[UserContent],
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
    ) -> TaskNode:
        node = self._build_agent_task(
            agent_id,
            node_id,
            validate_user_input(user_prompt),
            dependencies=dependencies,
            budget_cost=budget_cost,
            output=output,
            planning=planning,
            thinking=thinking,
            expander=expander,
        )
        self._generated_agent_tasks[node.node_id] = node
        return node

    @property
    def principal(self) -> Principal:
        return self._principal

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def source_node(self) -> TaskNode:
        return self._source_node

    @property
    def output(self) -> JsonValue:
        return self._output

    def _generated_agent_task(self, node_id: str) -> TaskNode | None:
        return self._generated_agent_tasks.get(node_id)


class RuntimeTaskNodeRunner(Generic[AppT]):
    """Interpret admitted TaskNodes using the frozen Runtime handler map."""

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        app: AppT,
        task_state: _TaskStateReader,
        task_objects: ObjectStore,
        object_key_factory: RuntimeObjectKeyFactory,
        payload_policy: PayloadPolicy,
        handlers: Sequence[TaskNodeHandler[AppT]] = (),
        expanders: Sequence[TaskExpander] = (),
        release_dependency_hold: Callable[..., Awaitable[None]] | None = None,
        request_terminal_handoff: Callable[..., Awaitable[None]] | None = None,
        task_durable: bool = False,
        execution_durable: bool = True,
        recovery_durable: bool = True,
    ) -> None:
        self._app = app
        self._catalog = catalog
        self._compiler = compiler
        self._execution = execution
        self._task_state = task_state
        self._task_objects = task_objects
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy
        self._agent = _AgentTaskNodeHandler(
            execution,
            catalog,
            compiler,
            release_dependency_hold=release_dependency_hold,
            request_terminal_handoff=request_terminal_handoff,
        )
        values: dict[tuple[str, int], TaskNodeHandler[AppT]] = {}
        for handler in handlers:
            key = _external_handler_identity(handler)
            if key in values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            values[key] = handler
        self._handlers = MappingProxyType(values)
        expander_values: dict[tuple[str, int], TaskExpander] = {}
        for expander in expanders:
            key = _external_expander_identity(expander)
            if key in expander_values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            expander_values[key] = expander
        self._expanders = MappingProxyType(expander_values)
        self._task_durable = task_durable
        self._execution_durable = execution_durable
        self._recovery_durable = recovery_durable
        self._materialization_tasks: set[asyncio.Task[StoredPayload]] = set()
        self._background_failure: AIError | None = None

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        materializations = tuple(
            cast("asyncio.Task[object]", task)
            for task in self._materialization_tasks
            if not task.done()
        )
        return (*self._agent.pending_background_tasks, *materializations)

    @property
    def pending_cancelled_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return self._agent.pending_cancelled_tasks

    @property
    def background_failure(self) -> AIError | None:
        if self._background_failure is not None:
            return AIError(
                self._background_failure.code,
                category=self._background_failure.category,
                retryable=self._background_failure.retryable,
                operation_id=self._background_failure.operation_id,
                safe_details=dict(self._background_failure.safe_details),
                diagnostics=self._background_failure.diagnostics,
            )
        return self._agent.background_failure

    def admit_node(self, node: TaskNode) -> TaskNode:
        task_type, task_version, body = _parse_node(node, request=True)
        handler = self._handler(task_type, task_version, request=True)
        if node.expander is not None:
            self._resolve_expander(node.expander, request=True)
        try:
            normalized = handler.normalize(body)
            canonical_body = _normalize_handler_body(normalized)
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        return TaskNode(
            node.node_id,
            node.dependencies,
            input={
                "type": task_type,
                "version": task_version,
                **canonical_body,
            },
            budget_cost=node.budget_cost,
            expander=node.expander,
        )

    def validate_request(self, graph: TaskGraph) -> None:
        for node in graph.nodes:
            canonical = self.admit_node(node)
            if canonical.input != node.input:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            self._validate_durability(node, graph_id=graph.graph_id, request=True)

    def validate_recovery(self, snapshot: TaskGraphSnapshot) -> None:
        for node, state in zip(snapshot.nodes, snapshot.node_states, strict=True):
            if state.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            }:
                continue
            self._validate_durability(
                node,
                graph_id=snapshot.graph_id,
                request=False,
            )
            task_type, task_version, body = _parse_node(node, request=False)
            try:
                handler = self._handler(task_type, task_version, request=False)
            except AIError as error:
                if error.code is not ErrorCode.CAPABILITY_REQUIRED_MISSING:
                    raise
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "task",
                        "task_type": task_type,
                        "task_version": task_version,
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                    },
                ) from error
            if node.expander is not None:
                self._resolve_expander(node.expander, request=False)
            if handler is self._agent:
                canonical_body = self._agent.validate_recovery(
                    body,
                    graph_id=snapshot.graph_id,
                    node_id=node.node_id,
                )
            else:
                try:
                    canonical_body = _normalize_handler_body(handler.normalize(body))
                except (AIError, TypeError, ValueError) as error:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        safe_details={
                            "graph_id": snapshot.graph_id,
                            "node_id": node.node_id,
                            "task_type": task_type,
                            "task_version": task_version,
                        },
                    ) from error
            canonical = TaskNode(
                node.node_id,
                node.dependencies,
                input={
                    "type": task_type,
                    "version": task_version,
                    **canonical_body,
                },
                budget_cost=node.budget_cost,
                expander=node.expander,
            )
            if canonical.input != node.input:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                        "task_type": task_type,
                        "task_version": task_version,
                    },
                )

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
        )
        execution_id: str | None = None
        if handler is self._agent:
            output, execution_id = await self._agent.run_node(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
                dependencies=dependencies,
                control=control,
            )
        else:
            idempotency_key = _custom_idempotency_key(
                graph_id,
                node,
                principal,
                dependencies,
            )
            task_context = TaskNodeContext(
                self._app,
                principal,
                graph_id,
                node.node_id,
                body,
                dependencies,
                idempotency_key,
                correlation,
            )
            try:
                output = normalize_json_value(await handler.run(task_context))
            except asyncio.CancelledError:
                raise
            except AIError:
                raise
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        return await self._complete_output(
            node,
            output,
            execution_id=execution_id,
            principal=principal,
            graph_id=graph_id,
        )

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        node = invocation.node
        task_type, task_version, _body = _parse_node(node, request=False)
        if (task_type, task_version) != (self._agent.type, self._agent.version):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = await self._execution.inspect(
            execution_id,
            principal=invocation.principal,
        )
        if view.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if view.status is ExecutionStatus.RECOVERY_REQUIRED:
            await self._execution.recover(
                execution_id,
                principal=invocation.principal,
            )
        result = await self._execution.wait(
            execution_id,
            principal=invocation.principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._complete_output(
            node,
            result.output,
            execution_id=execution_id,
            principal=invocation.principal,
            graph_id=invocation.graph_id,
        )

    def build_agent_task(
        self,
        agent_digest: str,
        node_id: str,
        user_prompt: CanonicalUserInput,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
    ) -> TaskNode:
        definition = self._root_definition(agent_digest)
        if planning is not None and not isinstance(planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        resolved_planning = (
            definition.spec.planning if planning is None else planning
        )
        resolved_thinking = (
            definition.spec.thinking
            if thinking is None
            else normalize_thinking(thinking)
        )
        binding = self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": self._agent.type,
                "version": self._agent.version,
                "binding": binding.snapshot.to_payload(),
                "user_prompt": task_prompt_draft(user_prompt),
                "mode": "run",
                "planning": resolved_planning,
                "thinking": resolved_thinking,
            },
            budget_cost=budget_cost,
            expander=expander,
        )

    def _build_agent_task_by_id(
        self,
        agent_id: str,
        node_id: str,
        user_prompt: CanonicalUserInput,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
    ) -> TaskNode:
        validate_agent_id(agent_id)
        try:
            definition = self._catalog.root_definition(agent_id)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={"kind": "agent", "agent_id": agent_id},
            ) from error
        return self.build_agent_task(
            definition.digest,
            node_id,
            user_prompt,
            dependencies=dependencies,
            budget_cost=budget_cost,
            output=output,
            planning=planning,
            thinking=thinking,
            expander=expander,
        )

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
        )
        if handler is self._agent:
            snapshot = await self._task_state.snapshot_graph(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            state = next(
                (
                    value
                    for value in snapshot.node_states
                    if value.node_id == node.node_id
                ),
                None,
            )
            if state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._agent.cancel_node(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
                dependencies=dependencies,
                durable_execution_id=state.execution_id,
            )
            return
        task_context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            body,
            dependencies,
            _custom_idempotency_key(graph_id, node, principal, dependencies),
            correlation,
        )
        await handler.cancel(task_context)

    async def get_result_record(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
    ) -> TaskResultRecord | None:
        records = await self._task_state.get_results(
            graph_id,
            (node_id,),
            tenant_id=tenant_id,
        )
        return records.get(node_id)

    async def read_result_record(self, record: TaskResultRecord) -> JsonValue:
        output = await self._read_payload(record.payload)
        if canonical_sha256(output) != record.result_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return output

    async def _complete_output(
        self,
        node: TaskNode,
        output: JsonValue,
        *,
        execution_id: str | None,
        principal: Principal,
        graph_id: str,
    ) -> TaskNodeRunResult:
        normalized = normalize_json_value(output)
        digest = canonical_sha256(normalized)
        payload = await self._materialize_result(
            normalized,
            tenant_id=principal.tenant_id,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if payload.digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expanded_nodes = self._expand_nodes(
            node,
            normalized,
            principal=principal,
            graph_id=graph_id,
        )
        return TaskNodeRunResult(
            digest,
            execution_id,
            payload,
            expanded_nodes,
        )

    def _expand_nodes(
        self,
        source_node: TaskNode,
        output: JsonValue,
        *,
        principal: Principal,
        graph_id: str,
    ) -> tuple[TaskNode, ...]:
        if source_node.expander is None:
            return ()
        expander = self._resolve_expander(source_node.expander, request=False)
        context_impl = _TaskExpansionContext(
            principal,
            graph_id,
            source_node,
            output,
            build_agent_task=self._build_agent_task_by_id,
        )
        context: TaskExpansionContext = context_impl
        try:
            expanded = expander.expand(context)
        except AIError as error:
            if error.code is ErrorCode.CAPABILITY_REQUIRED_MISSING:
                raise
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_failed",
            ) from error
        except (TypeError, ValueError) as error:
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_output_invalid",
            ) from error
        if not isinstance(expanded, Sequence) or isinstance(
            expanded,
            (str, bytes, bytearray),
        ):
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_output_invalid",
            )
        raw_nodes = tuple(expanded)
        raw_ids = tuple(node.node_id for node in raw_nodes if isinstance(node, TaskNode))
        if len(raw_ids) != len(raw_nodes) or len(set(raw_ids)) != len(raw_ids):
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="duplicate_node_id",
            )
        nodes: list[TaskNode] = []
        for raw_node in raw_nodes:
            task_type = raw_node.input.get("type")
            if task_type == self._agent.type:
                generated = context_impl._generated_agent_task(raw_node.node_id)
                if generated != raw_node:
                    raise _expansion_error(
                        graph_id,
                        source_node.node_id,
                        reason="agent_task_builder_required",
                        conflict=raw_node.node_id,
                    )
            try:
                admitted = self.admit_node(raw_node)
                self._validate_durability(
                    admitted,
                    graph_id=graph_id,
                    request=False,
                )
                nodes.append(admitted)
            except AIError as error:
                if error.code is ErrorCode.CAPABILITY_REQUIRED_MISSING:
                    raise
                raise _expansion_error(
                    graph_id,
                    source_node.node_id,
                    reason="node_admission_invalid",
                    conflict=raw_node.node_id,
                ) from error
        result = tuple(sorted(nodes, key=lambda item: item.node_id))
        _logger.info(
            "task graph expansion produced nodes: graph=%s source=%s nodes=%s",
            graph_id,
            source_node.node_id,
            tuple(node.node_id for node in result),
        )
        return result

    def _resolve_expander(
        self,
        reference: TaskExpanderRef,
        *,
        request: bool,
    ) -> TaskExpander:
        expander = self._expanders.get((reference.id, reference.version))
        if expander is not None:
            return expander
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={
                "kind": "task_expander",
                "expander_id": reference.id,
                "expander_version": reference.version,
            },
        )

    def _validate_durability(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        request: bool,
    ) -> None:
        task_type = node.input.get("type")
        if not self._task_durable or task_type != self._agent.type:
            return
        if self._execution_durable and self._recovery_durable:
            return
        raise AIError(
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            safe_details={
                "phase": "task_durability_validation",
                "graph_id": graph_id,
                "node_id": node.node_id,
                "request": request,
            },
        )

    def _root_definition(self, agent_digest: str) -> AgentDefinition:
        for agent_id in self._catalog.root_ids:
            definition = self._catalog.root_definition(agent_id)
            if definition.digest == agent_digest:
                return definition
        raise AIError(
            ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"kind": "agent", "agent_digest": agent_digest},
        )

    async def _dependencies(
        self,
        node: TaskNode,
        *,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> dict[str, TaskDependency]:
        if set(dependency_results) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values: dict[str, TaskDependency] = {}
        for dependency_id in sorted(node.dependencies):
            dependency = dependency_results[dependency_id]
            if dependency.result_payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            output = await self._read_payload(dependency.result_payload)
            values[dependency_id] = TaskDependency(
                dependency_id,
                output,
                dependency.result_digest,
                dependency.execution_id,
            )
        return values

    async def _materialize_result(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
        graph_id: str,
        node_id: str,
    ) -> StoredPayload:
        task = asyncio.create_task(
            self._materialize_result_inner(output, tenant_id=tenant_id),
            name=f"task-result-materialize-{graph_id}-{node_id}",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._consume_materialization(task, graph_id=graph_id, node_id=node_id)
            else:
                self._materialization_tasks.add(task)

                def consume(done: asyncio.Task[StoredPayload]) -> None:
                    try:
                        self._consume_materialization(
                            done,
                            graph_id=graph_id,
                            node_id=node_id,
                        )
                    finally:
                        self._materialization_tasks.discard(done)

                task.add_done_callback(consume)
            raise

    def _consume_materialization(
        self,
        task: asyncio.Task[StoredPayload],
        *,
        graph_id: str,
        node_id: str,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as error:  # noqa: BLE001
            if self._background_failure is not None:
                return
            details = dict(error.safe_details) if isinstance(error, AIError) else {}
            details.setdefault("phase", "task_result_materialize")
            details.setdefault("graph_id", graph_id)
            details.setdefault("node_id", node_id)
            self._background_failure = AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details=details,
            )

    async def _materialize_result_inner(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
    ) -> StoredPayload:
        inline = StoredPayload.inline_json(output)
        if payload_fits_inline(inline, self._payload_policy):
            return inline
        data = canonical_json_bytes(output)
        reference = await put_runtime_object(
            self._task_objects,
            self._object_key_factory,
            RuntimeDomain.TASK,
            tenant_id,
            data,
        )
        return StoredPayload.object(reference)

    async def _read_payload(self, payload: StoredPayload) -> JsonValue:
        try:
            if payload.kind == "inline":
                value = payload.decode()
            else:
                if payload.ref is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                data = await read_runtime_object(self._task_objects, payload.ref)
                value = json.loads(data.decode("utf-8"))
            normalized = normalize_json_value(value)
        except AIError:
            raise
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if canonical_sha256(normalized) != payload.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return normalized

    def _handler(
        self,
        task_type: str,
        task_version: int,
        *,
        request: bool,
    ) -> TaskNodeHandler[AppT] | _AgentTaskNodeHandler:
        if task_type == self._agent.type and task_version == self._agent.version:
            return self._agent
        handler = self._handlers.get((task_type, task_version))
        if handler is not None:
            return handler
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"task_type": task_type, "task_version": task_version},
        )


def _parse_node(
    node: TaskNode,
    *,
    request: bool,
) -> tuple[str, int, dict[str, JsonValue]]:
    payload = node.input
    task_type = payload.get("type")
    task_version = payload.get("version")
    if (
        not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    body = {
        key: value for key, value in payload.items() if key not in {"type", "version"}
    }
    return task_type, task_version, body


def _external_handler_identity(handler: TaskNodeHandler[object]) -> tuple[str, int]:
    task_type = handler.type
    task_version = handler.version
    if (
        not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or task_type.startswith("linktools.ai.")
        or not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return task_type, task_version


def _external_expander_identity(expander: TaskExpander) -> tuple[str, int]:
    try:
        reference = TaskExpanderRef(expander.id, expander.version)
    except (TypeError, ValueError, AttributeError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if reference.id.startswith(_RESERVED_EXPANDER_ID_PREFIX):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return reference.id, reference.version


def _expansion_error(
    graph_id: str,
    source_node_id: str,
    *,
    reason: str,
    conflict: str | None = None,
) -> AIError:
    details: dict[str, JsonValue] = {
        "phase": "task_graph_expansion",
        "reason": reason,
        "graph_id": graph_id,
        "source_node_id": source_node_id,
    }
    if conflict is not None:
        details["conflict"] = conflict
    return AIError(ErrorCode.TASK_DAG_INVALID, safe_details=details)


async def _noop_async_callback(*args: object, **kwargs: object) -> None:
    return None


def _normalize_handler_body(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("task handler normalize must return a mapping")
    normalized = normalize_json_value(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("task handler normalize must return a mapping")
    if "type" in normalized or "version" in normalized:
        raise ValueError("task handler normalize returned reserved fields")
    return normalized


def _custom_idempotency_key(
    graph_id: str,
    node: TaskNode,
    principal: Principal,
    dependencies: Mapping[str, TaskDependency],
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "graph_id": graph_id,
            "node_id": node.node_id,
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


__all__ = ["RuntimeTaskNodeRunner"]
