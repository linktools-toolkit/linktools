#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of generic TaskNodes."""

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ

from ..agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler
from ..core import (
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Principal,
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
    DefaultTaskService,
    TaskDependency,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphSnapshot,
    TaskNode,
    TaskNodeContext,
    TaskNodeHandler,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    TaskResultRecord,
)
from ._input import user_prompt_transport
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
_AGENT_BODY_FIELDS = frozenset(
    {
        "binding",
        "user_prompt",
        "user_prompt_codec",
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
    ) -> None:
        self._execution = execution
        self._catalog = catalog
        self._compiler = compiler
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
        base_user_prompt = input.get("user_prompt")
        user_prompt_codec = input.get("user_prompt_codec")
        mode = input.get("mode")
        planning = input.get("planning")
        thinking = input.get("thinking")
        if (
            not isinstance(base_user_prompt, str)
            or not isinstance(user_prompt_codec, str)
            or not isinstance(planning, bool)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            transport = user_prompt_transport(base_user_prompt, user_prompt_codec)
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
        validate_user_prompt(str(transport))
        return {
            "binding": binding.snapshot.to_payload(),
            "user_prompt": str(transport),
            "user_prompt_codec": transport.codec,
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
        dependencies: Mapping[str, TaskDependency],
        control: TaskNodeRunControl,
    ) -> tuple[JsonValue, str]:
        binding_digest, request = self._prepare_request(
            node,
            graph_id=graph_id,
            principal=principal,
            dependencies=dependencies,
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
                self._bind_after_launch(
                    launch_task,
                    key,
                    control,
                ),
                name=f"task-execution-bind-after-launch-{graph_id}-{node.node_id}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                f"task execution bind after launch graph={graph_id} task={node.node_id}",
            )
            raise
        finally:
            if launch_task.done() and self._active_launch_tasks.get(key) is launch_task:
                self._active_launch_tasks.pop(key, None)
        if not handle.execution_id:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        await self._bind_execution(
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
                    if launch_task.done() and self._active_launch_tasks.get(key) is launch_task:
                        self._active_launch_tasks.pop(key, None)
                if handle is not None and handle.execution_id:
                    execution_id = handle.execution_id
        if execution_id is None:
            binding_digest, request = self._prepare_request(
                node,
                graph_id=graph_id,
                principal=principal,
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
        dependencies: Mapping[str, TaskDependency],
    ) -> tuple[str, ExecutionRequest]:
        payload = node.input
        if payload.get("type") != self.type or payload.get("version") != self.version:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        body = {key: value for key, value in payload.items() if key not in {"type", "version"}}
        normalized = self.validate_recovery(
            body,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if normalized != body:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        snapshot = AgentBindingSnapshot.from_payload(normalized["binding"])
        binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        base_user_prompt = user_prompt_transport(
            cast(str, normalized["user_prompt"]),
            cast(str, normalized["user_prompt_codec"]),
        )
        dependency_payload = {
            dependency_id: dependencies[dependency_id].output
            for dependency_id in sorted(node.dependencies)
        }
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
                        "result_digest": dependencies[dependency_id].result_digest,
                    }
                    for dependency_id in sorted(node.dependencies)
                ],
                "principal": principal_identity_payload(principal),
            }
        )
        return binding.digest, ExecutionRequest(
            user_prompt=str(effective_user_prompt),
            user_prompt_codec=effective_user_prompt.codec,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=None,
            mode=cast(ExecutionMode, normalized["mode"]),
            planning=cast(bool, normalized["planning"]),
            thinking=cast(ThinkingValue, normalized["thinking"]),
        )

    async def _bind_execution(
        self,
        control: TaskNodeRunControl,
        execution_id: str,
        *,
        key: tuple[str, str, str],
    ) -> None:
        task = asyncio.create_task(
            control.bind_execution(execution_id),
            name=f"task-execution-bind-{key[1]}-{key[2]}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continuation = asyncio.create_task(
                self._settle_detached_bind(task, key),
                name=f"task-execution-bind-settle-{key[1]}-{key[2]}",
            )
            self._detach(
                cast("asyncio.Task[object]", continuation),
                f"task execution bind graph={key[1]} task={key[2]}",
            )
            raise

    async def _settle_detached_bind(
        self,
        task: asyncio.Task[None],
        key: tuple[str, str, str],
    ) -> None:
        try:
            await task
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
                phase="task_execution_bind",
            ) from error
        except BaseException as error:  # noqa: BLE001
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_bind",
            ) from error

    async def _bind_after_launch(
        self,
        launch_task: asyncio.Task[ExecutionHandle],
        key: tuple[str, str, str],
        control: TaskNodeRunControl,
    ) -> None:
        try:
            handle = await launch_task
            if not handle.execution_id:
                raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
            await control.bind_execution(handle.execution_id)
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
                phase="task_execution_bind_after_launch",
            ) from error
        except BaseException as error:  # noqa: BLE001
            raise self._record_background_failure(
                key,
                error,
                phase="task_execution_bind_after_launch",
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
    ) -> None:
        self._app = app
        self._task_state = task_state
        self._task_objects = task_objects
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy
        self._agent = _AgentTaskNodeHandler(execution, catalog, compiler)
        values: dict[tuple[str, int], TaskNodeHandler[AppT]] = {}
        for handler in handlers:
            key = _external_handler_identity(handler)
            if key in values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            values[key] = handler
        self._handlers = MappingProxyType(values)
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
        )

    def validate_request(self, graph: TaskGraph) -> None:
        for node in graph.nodes:
            canonical = self.admit_node(node)
            if canonical.input != node.input:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    def validate_recovery(self, graph: TaskGraph) -> None:
        for node in graph.nodes:
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
                        "graph_id": graph.graph_id,
                        "node_id": node.node_id,
                    },
                ) from error
            if handler is self._agent:
                canonical_body = self._agent.validate_recovery(
                    body,
                    graph_id=graph.graph_id,
                    node_id=node.node_id,
                )
            else:
                try:
                    canonical_body = _normalize_handler_body(handler.normalize(body))
                except (AIError, TypeError, ValueError) as error:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        safe_details={
                            "graph_id": graph.graph_id,
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
            )
            if canonical.input != node.input:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph.graph_id,
                        "node_id": node.node_id,
                        "task_type": task_type,
                        "task_version": task_version,
                    },
                )

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            principal=principal,
            dependency_results=dependency_results,
        )
        execution_id: str | None = None
        if handler is self._agent:
            output, execution_id = await self._agent.run_node(
                node,
                graph_id=graph_id,
                principal=principal,
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
            context = TaskNodeContext(
                self._app,
                principal,
                graph_id,
                node.node_id,
                body,
                dependencies,
                idempotency_key,
            )
            try:
                output = normalize_json_value(await handler.run(context))
            except asyncio.CancelledError:
                raise
            except AIError:
                raise
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        digest = canonical_sha256(output)
        payload = await self._materialize_result(
            output,
            tenant_id=principal.tenant_id,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if payload.digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNodeRunResult(digest, execution_id, payload)

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> None:
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            principal=principal,
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
                (value for value in snapshot.node_states if value.node_id == node.node_id),
                None,
            )
            if state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._agent.cancel_node(
                node,
                graph_id=graph_id,
                principal=principal,
                dependencies=dependencies,
                durable_execution_id=state.execution_id,
            )
            return
        context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            body,
            dependencies,
            _custom_idempotency_key(graph_id, node, principal, dependencies),
        )
        await handler.cancel(context)

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

    async def read_legacy_agent_result(
        self,
        execution_id: str,
        *,
        principal: Principal,
        expected_digest: str,
    ) -> JsonValue:
        return await self._agent.read_result(
            execution_id,
            principal=principal,
            expected_digest=expected_digest,
        )

    async def _dependencies(
        self,
        node: TaskNode,
        *,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> dict[str, TaskDependency]:
        if set(dependency_results) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values: dict[str, TaskDependency] = {}
        for dependency_id in sorted(node.dependencies):
            dependency = dependency_results[dependency_id]
            if dependency.result_payload is not None:
                output = await self._read_payload(dependency.result_payload)
            elif dependency.execution_id is not None:
                output = await self.read_legacy_agent_result(
                    dependency.execution_id,
                    principal=principal,
                    expected_digest=dependency.result_digest,
                )
            else:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
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
            ErrorCode.REQUEST_FIELD_INVALID if request else ErrorCode.CAPABILITY_REQUIRED_MISSING,
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
            ErrorCode.REQUEST_FIELD_INVALID if request else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    body = {key: value for key, value in payload.items() if key not in {"type", "version"}}
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


__all__ = ["DefaultTaskService", "RuntimeTaskNodeRunner"]
