#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime adapter for Agent-backed TaskGraph nodes."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Generic, TypeVar
from typing import cast

from linktools.core import environ

from ..agent import AgentBindingContract, AgentCatalog, AgentCompiler
from ..core import (
    CorrelationData,
    ImmutableJsonMapping,
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Principal,
    TaskStatus,
    ThinkingValue,
    canonical_sha256,
    normalize_json_value,
    normalize_execution_mode,
    normalize_thinking,
    principal_identity_payload,
    validate_agent_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..task import (
    TaskDependency,
    TaskDependencyState,
    TaskNode,
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    TaskResultRef,
)
from ._input import (
    ExecutionInputMaterializer,
    decode_task_prompt_draft,
    decode_user_content_payload,
)
from .state._codec import decode_domain
from ._input import task_prompt_draft, validate_user_input
from .state._contracts import StoredUserInput, TaskPreparedInputRecord
from .service_api import (
    CancelExecutionRequest,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ResumeSessionRequest,
    SessionService,
)
from ._agent_task_input import (
    AgentTaskInput,
    AgentTaskInputBuilder,
    AgentTaskInputContext,
    _AgentTaskContextError,
)
from ._input_contract import CanonicalUserInput

_logger = environ.get_logger("ai.runtime.planner")
AppT = TypeVar("AppT")


class RuntimeAgentTaskRunner(Generic[AppT]):
    """TaskNodeRunner adapter that keeps Agent calls on Runtime execution APIs."""

    def __init__(
        self,
        *,
        id: str,
        revision: int,
        input_mode: str,
        planning_default: bool,
        thinking_default: ThinkingValue,
        binding_contract: Mapping[str, JsonValue],
        build_input: AgentTaskInputBuilder | None,
        start_execution: Callable[..., Awaitable[object]],
        get_execution: Callable[[str, Principal], Awaitable[object]],
        result_reader: Callable[[TaskNodeInvocation, str], Awaitable[JsonValue]],
        result_ref_reader: Callable[[TaskNodeInvocation, str], Awaitable[TaskResultRef]],
        get_prepared_input: Callable[
            [TaskNodeInvocation], Awaitable[TaskPreparedInputRecord | None]
        ],
        publish_prepared_input: Callable[..., Awaitable[TaskPreparedInputRecord]],
        store_prepared_prompt: Callable[..., Awaitable[StoredUserInput]],
        restore_prepared_prompt: Callable[
            [StoredUserInput], Awaitable[CanonicalUserInput]
        ],
    ) -> None:
        if input_mode not in {"literal", "projected"}:
            raise ValueError("Agent Task input mode is invalid")
        if (input_mode == "projected") != (build_input is not None):
            raise ValueError("Agent Task input callback does not match input mode")
        self.id = id
        self.revision = revision
        self.input_mode = input_mode
        self._planning_default = planning_default
        self._thinking_default = thinking_default
        self._binding_contract = ImmutableJsonMapping(binding_contract)
        self._build_input = build_input
        self._start_execution = start_execution
        self._get_execution = get_execution
        self._result_reader = result_reader
        self._result_ref_reader = result_ref_reader
        self._get_prepared_input = get_prepared_input
        self._publish_prepared_input = publish_prepared_input
        self._store_prepared_prompt = store_prepared_prompt
        self._restore_prepared_prompt = restore_prepared_prompt

    def normalize(self, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        task_input = AgentTaskInput.from_mapping(value)
        normalized = dict(task_input)
        normalized["planning"] = (
            self._planning_default
            if task_input.planning is None
            else task_input.planning
        )
        normalized["thinking"] = (
            self._thinking_default
            if task_input.thinking is None
            else normalize_thinking(task_input.thinking)
        )
        return normalized

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        task_input = AgentTaskInput.from_mapping(invocation.node.input)
        files = task_input.files
        request_identity: str | None = None
        if self.input_mode == "literal":
            if task_input.parameters:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if task_input.stored_prompt is None:
                prompt = task_input.prompt
            else:
                if not isinstance(task_input.stored_prompt, StoredUserInput):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                prompt = await self._restore_prepared_prompt(task_input.stored_prompt)
                files = ()
        else:
            callback = self._build_input
            if callback is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            input_identity = _agent_task_input_identity(
                invocation,
                task_input,
                task_id=self.id,
                task_revision=self.revision,
                binding_contract=self._binding_contract,
            )
            prepared = await self._get_prepared_input(invocation)
            if prepared is not None and prepared.input_identity != input_identity:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if invocation.execution_id is not None and prepared is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if prepared is not None:
                prompt = await self._restore_prepared_prompt(
                    prepared.stored_user_input
                )
                files = ()
                request_identity = prepared.request_identity
                expected_digest = _prepared_agent_input_digest(
                    prompt, prepared.stored_user_input
                )
                expected_identity = _agent_task_request_identity(
                    invocation,
                    task_input,
                    prompt,
                    expected_digest,
                    input_identity,
                    self.id,
                    self.revision,
                    self._binding_contract,
                )
                if (
                    expected_digest != prepared.final_input_digest
                    or expected_identity != prepared.request_identity
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                context = AgentTaskInputContext(
                    invocation,
                    task_input,
                    lambda name: self._result_reader(invocation, name),
                    lambda name: self._result_ref_reader(invocation, name),
                )
                try:
                    projected = await callback(context)
                    prompt = validate_user_input(projected)
                except asyncio.CancelledError:
                    raise
                except _AgentTaskContextError as error:
                    cause = error.__cause__
                    if isinstance(cause, AIError):
                        raise cause
                    raise
                except AIError as error:
                    raise AIError(
                        ErrorCode.TASK_INPUT_PROJECTION_FAILED,
                        safe_details={"cause_code": error.code.value},
                    ) from error
                except Exception as error:  # noqa: BLE001
                    raise AIError(
                        ErrorCode.TASK_INPUT_PROJECTION_FAILED,
                        safe_details={"cause_type": type(error).__name__},
                    ) from error
                stored = await self._store_prepared_prompt(
                    prompt,
                    files=task_input.files,
                    tenant_id=invocation.principal.tenant_id,
                )
                prompt = await self._restore_prepared_prompt(stored)
                final_input_digest = _prepared_agent_input_digest(prompt, stored)
                request_identity = _agent_task_request_identity(
                    invocation,
                    task_input,
                    prompt,
                    final_input_digest,
                    input_identity,
                    self.id,
                    self.revision,
                    self._binding_contract,
                )
                prepared = await self._publish_prepared_input(
                    invocation,
                    input_identity=input_identity,
                    source_refs=context.source_refs,
                    stored_user_input=stored,
                    final_input_digest=final_input_digest,
                    request_identity=request_identity,
                )
                if (
                    prepared.input_identity != input_identity
                    or prepared.final_input_digest != final_input_digest
                    or prepared.request_identity != request_identity
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                prompt = await self._restore_prepared_prompt(
                    prepared.stored_user_input
                )
                files = ()

        planning = (
            self._planning_default
            if task_input.planning is None
            else task_input.planning
        )
        thinking = (
            self._thinking_default
            if task_input.thinking is None
            else normalize_thinking(task_input.thinking)
        )
        try:
            if request_identity is None:
                request_identity = canonical_sha256(
                    {
                        "version": 1,
                        "graph_id": invocation.graph_id,
                        "node_id": invocation.node.node_id,
                        "task_id": self.id,
                        "task_revision": self.revision,
                        "agent_binding_contract": dict(self._binding_contract),
                        "principal": principal_identity_payload(invocation.principal),
                        "prompt": task_prompt_draft(prompt),
                        "parameters": dict(task_input.parameters),
                        "files": list(files),
                        "session_id": task_input.session_id,
                        "memory_scope": task_input.memory_scope,
                        "planning": planning,
                        "thinking": thinking,
                        "output_contract": _node_output_contract(invocation),
                    }
                )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if request_identity is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = (
            await self._get_execution(
                invocation.execution_id,
                invocation.principal,
            )
            if invocation.execution_id is not None
            else await self._start_execution(
                invocation,
                prompt,
                files=files,
                session_id=task_input.session_id,
                memory_scope=task_input.memory_scope,
                planning=planning,
                thinking=thinking,
                idempotency_key=request_identity,
            )
        )
        execution_id = getattr(execution, "execution_id", None)
        if not isinstance(execution_id, str) or not execution_id:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        current = control.execution_id
        if current is None:
            await control.bind_execution(execution_id)
        elif current != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await control.handoff_execution(execution_id)
        result = await execution.wait()
        return _agent_task_result(result, execution_id)

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        execution = await self._get_execution(execution_id, invocation.principal)
        result = await execution.wait()
        return _agent_task_result(result, execution_id)

    async def supply_input(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        value: JsonValue,
    ) -> TaskNodeRunResult:
        del invocation, execution_id, value
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def resolve_effect(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        resolution: object,
    ) -> TaskNodeRunResult | None:
        del invocation, execution_id, resolution
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        execution_id = invocation.execution_id
        if execution_id is None:
            return
        execution = await self._get_execution(execution_id, invocation.principal)
        await execution.cancel()


def _node_output_contract(invocation: TaskNodeInvocation) -> dict[str, JsonValue] | None:
    value = invocation.node.output_contract
    return None if value is None else dict(value)


def _agent_task_input_identity(
    invocation: TaskNodeInvocation,
    task_input: AgentTaskInput,
    *,
    task_id: str,
    task_revision: int,
    binding_contract: Mapping[str, JsonValue],
) -> str:
    try:
        return canonical_sha256(
            {
                "version": 1,
                "graph_id": invocation.graph_id,
                "node_id": invocation.node.node_id,
                "task_id": task_id,
                "task_revision": task_revision,
                "binding_contract": dict(binding_contract),
                "principal": principal_identity_payload(invocation.principal),
                "input": dict(task_input),
                "output_contract": _node_output_contract(invocation),
            }
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _prepared_agent_input_digest(
    prompt: CanonicalUserInput,
    stored: StoredUserInput,
) -> str:
    try:
        return canonical_sha256(
            {
                "prompt": task_prompt_draft(prompt),
                "stored_user_input": stored.digest,
            }
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _agent_task_request_identity(
    invocation: TaskNodeInvocation,
    task_input: AgentTaskInput,
    prompt: CanonicalUserInput,
    final_input_digest: str,
    input_identity: str,
    task_id: str,
    task_revision: int,
    binding_contract: Mapping[str, JsonValue],
) -> str:
    try:
        return canonical_sha256(
            {
                "version": 1,
                "graph_id": invocation.graph_id,
                "node_id": invocation.node.node_id,
                "task_id": task_id,
                "task_revision": task_revision,
                "agent_binding_contract": dict(binding_contract),
                "principal": principal_identity_payload(invocation.principal),
                "input_identity": input_identity,
                "prompt": task_prompt_draft(prompt),
                "files": list(task_input.files),
                "final_input_digest": final_input_digest,
                "session_id": task_input.session_id,
                "memory_scope": task_input.memory_scope,
                "planning": task_input.planning,
                "thinking": task_input.thinking,
                "output_contract": _node_output_contract(invocation),
            }
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _agent_task_result(result: object, execution_id: str) -> TaskNodeRunResult:
    status = getattr(result, "status", None)
    output = getattr(result, "output", None)
    if status is not ExecutionStatus.SUCCEEDED:
        raw_code = getattr(result, "error_code", None)
        try:
            code = ErrorCode(raw_code) if isinstance(raw_code, str) else ErrorCode.TASK_NODE_FAILED
        except ValueError:
            code = ErrorCode.TASK_NODE_FAILED
        details = getattr(result, "safe_error_details", {})
        raise TaskNodeRunError(code, execution_id, safe_details=details)
    try:
        output = normalize_json_value(output)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return TaskNodeRunResult(canonical_sha256(output), execution_id)
_AGENT_TASK_ID = "linktools.ai.agent"
_AGENT_TASK_REVISION = 1
_AGENT_BODY_FIELDS = frozenset(
    {
        "binding_contract",
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
    id = _AGENT_TASK_ID
    revision = _AGENT_TASK_REVISION
    effect_policy = "none"
    output_type = None
    reconcile = None

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        session: SessionService | None = None,
        release_dependency_hold: Callable[..., Awaitable[None]] | None = None,
        input_materializer: ExecutionInputMaterializer | None = None,
    ) -> None:
        self._execution = execution
        self._input_materializer = input_materializer
        self._session = session
        del catalog
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
            elif kind == "task-user-content-v1":
                base_user_prompt = decode_task_prompt_draft(raw_user_prompt)
            elif kind == "stored-user-content-v1":
                if set(raw_user_prompt) != {"kind", "intent", "value"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                decode_domain(raw_user_prompt["value"], StoredUserInput)
                base_user_prompt = ()
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            resolved_mode = normalize_execution_mode(mode)
            resolved_thinking = normalize_thinking(thinking)
            binding_contract = AgentBindingContract.from_payload(
                input.get("binding_contract")
            )
            binding = self._compiler.restore(binding_contract)
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if resolved_mode != "run":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            binding.binding_contract != binding_contract
            or binding.binding_digest != binding_contract.binding_digest
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_agent_id(binding.compiled_agent.spec.id)
        if isinstance(base_user_prompt, str):
            validate_user_prompt(base_user_prompt)
        return {
            "binding_contract": binding.binding_contract.to_payload(),
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
                ErrorCode.AGENT_BINDING_UNAVAILABLE,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            }:
                raise cause
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "task_id": self.id,
                    "task_revision": self.revision,
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
        dependency_reader: Callable[[TaskDependency], Awaitable[JsonValue]],
        control: TaskNodeRunControl,
        dependency_states: Mapping[str, TaskDependencyState] | None = None,
    ) -> tuple[JsonValue, str]:
        prepared = await self._prepare_request(
            node,
            graph_id=graph_id,
            principal=principal,
            correlation=correlation,
        )
        binding_digest, request = prepared[:2]
        agent_id = prepared[2] if len(prepared) > 2 else ""
        session_id = prepared[3] if len(prepared) > 3 else None
        binding_contract = prepared[4] if len(prepared) > 4 else None
        key = (principal.tenant_id, graph_id, node.node_id)
        hold_id = f"task:{graph_id}:{node.node_id}"
        if session_id is None or self._session is None:
            launch = self._execution.start(
                binding_digest,
                request,
                dependency_hold_id=hold_id,
                binding_contract=binding_contract,
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
                binding_contract=binding_contract,
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
        dependency_reader: Callable[[TaskDependency], Awaitable[JsonValue]],
        durable_execution_id: str | None,
        dependency_states: Mapping[str, TaskDependencyState] | None = None,
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
            (
                binding_digest,
                request,
                _,
                _,
                binding_contract,
            ) = await self._prepare_request(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
            )
            try:
                handle = await self._execution.resolve_existing(
                    binding_digest,
                    request,
                    binding_contract=binding_contract,
                )
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

    async def _prepare_request(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        correlation: CorrelationData,
    ) -> tuple[
        str,
        ExecutionRequest,
        str,
        str | None,
        AgentBindingContract,
    ]:
        payload = node.input
        if (
            payload.get("task_id") != self.id
            or payload.get("task_revision") != self.revision
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        body = {
            key: value
            for key, value in payload.items()
            if key not in {"task_id", "task_revision"}
        }
        normalized = self.validate_recovery(
            body,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if normalized != body:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding_contract = AgentBindingContract.from_payload(
            normalized["binding_contract"]
        )
        binding = self._compiler.restore(binding_contract)
        raw_user_prompt = cast(Mapping[str, JsonValue], normalized["user_prompt"])
        if raw_user_prompt.get("kind") == "text":
            base_user_prompt: str | tuple[object, ...] = cast(
                str, raw_user_prompt["text"]
            )
        elif raw_user_prompt.get("kind") == "pydantic-user-content-v1":
            value = raw_user_prompt.get("value")
            if not isinstance(value, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            base_user_prompt = decode_user_content_payload(value)
        elif raw_user_prompt.get("kind") == "stored-user-content-v1":
            if self._input_materializer is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stored = decode_domain(raw_user_prompt["value"], StoredUserInput)
            base_user_prompt = await self._input_materializer.restore(stored)
        else:
            base_user_prompt = decode_task_prompt_draft(raw_user_prompt)
        effective_user_prompt = base_user_prompt
        if isinstance(effective_user_prompt, str):
            validate_user_prompt(effective_user_prompt)
        idempotency_key = canonical_sha256(
            {
                "version": 1,
                "graph_id": graph_id,
                "node_id": node.node_id,
                "binding_digest": binding.binding_digest,
                "input": node.input,
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
            binding.binding_digest,
            request,
            binding.compiled_agent.spec.id,
            cast("str | None", normalized["session_id"]),
            binding.binding_contract,
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


def _dependency_identity_payload(
    node: TaskNode,
    dependencies: Mapping[str, TaskDependency],
    dependency_states: Mapping[str, TaskDependencyState],
) -> list[dict[str, JsonValue]]:
    if node.dependency_policy == "all_succeeded":
        return [
            {
                "node_id": dependency_id,
                "result_digest": dependencies[dependency_id].result_digest,
            }
            for dependency_id in sorted(dependencies)
        ]
    if node.dependency_policy != "all_terminal":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(dependency_states) != set(node.dependencies):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    node_dependencies = set(node.dependencies)
    result: list[dict[str, JsonValue]] = []
    for dependency_id in sorted(node_dependencies | set(dependencies)):
        if dependency_id not in node_dependencies:
            result.append(
                {
                    "node_id": dependency_id,
                    "result_digest": dependencies[dependency_id].result_digest,
                }
            )
            continue
        state = dependency_states[dependency_id]
        if state.status is TaskStatus.SUCCEEDED:
            dependency = dependencies.get(dependency_id)
            if (
                dependency is None
                or dependency.result_digest != state.result_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif dependency_id in dependencies:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result.append(
            {
                "node_id": dependency_id,
                **state.to_payload(),
            }
        )
    return result


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
