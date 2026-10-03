#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime adapter for Agent-backed TaskGraph nodes."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Generic, TypeVar

from ..agent import AgentBindingContract
from ..core import (
    ImmutableJsonMapping,
    ExecutionStatus,
    JsonValue,
    Principal,
    ThinkingValue,
    canonical_sha256,
    normalize_json_value,
    normalize_thinking,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..task import (
    Task,
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunResult,
    TaskResultRef,
)
from ._input import decode_task_prompt_draft, task_prompt_draft
from .state._contracts import StoredUserInput, TaskPreparedInputRecord
from ._agent_task_input import (
    AgentTaskInput,
    AgentTaskInputBuilder,
    AgentTaskInputContext,
    _AgentTaskContextError,
)
from ._input_contract import CanonicalUserInput, validate_user_input

AppT = TypeVar("AppT")


class RuntimeAgentTaskRunner(Generic[AppT]):
    """TaskNodeRunner adapter that keeps Agent calls on Runtime execution APIs."""

    def __init__(
        self,
        *,
        id: str,
        revision: int,
        runtime_owner: object,
        agent_id: str,
        agent_revision: int,
        input_mode: str,
        planning_default: bool,
        thinking_default: ThinkingValue,
        binding_contract: Mapping[str, JsonValue],
        build_input: AgentTaskInputBuilder | None,
        start_execution: Callable[..., Awaitable[object]],
        get_execution: Callable[[str, Principal], Awaitable[object]],
        acquire_execution_hold: Callable[[str, Principal, str], Awaitable[None]],
        release_execution_hold: Callable[[str, Principal, str], Awaitable[None]],
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
        self._runtime_owner = runtime_owner
        self._agent_id = agent_id
        self._agent_revision = agent_revision
        self.input_mode = input_mode
        self._planning_default = planning_default
        self._thinking_default = thinking_default
        self._binding_contract = ImmutableJsonMapping(binding_contract)
        self._binding_digest = AgentBindingContract.from_payload(
            self._binding_contract
        ).binding_digest
        self._build_input = build_input
        self._start_execution = start_execution
        self._get_execution = get_execution
        self._acquire_execution_hold = acquire_execution_hold
        self._release_execution_hold = release_execution_hold
        self._result_reader = result_reader
        self._result_ref_reader = result_ref_reader
        self._get_prepared_input = get_prepared_input
        self._publish_prepared_input = publish_prepared_input
        self._store_prepared_prompt = store_prepared_prompt
        self._restore_prepared_prompt = restore_prepared_prompt

    def validate_binding(self, runtime_owner: object, task: Task[AppT]) -> None:
        if runtime_owner is not self._runtime_owner:
            raise AIError(ErrorCode.RUNTIME_SERVICE_MISMATCH)
        expected_contract: dict[str, JsonValue] = {
            "version": 1,
            "type": "agent",
            "effect_policy": "none",
            "output_contract": {"kind": "json"},
            "reconcile": False,
            "config": {
                "agent_id": self._agent_id,
                "agent_revision": self._agent_revision,
                "binding_contract": dict(self._binding_contract),
                "input_mode": self.input_mode,
            },
        }
        if (
            task.runner is not self
            or task.id != self.id
            or task.revision != self.revision
            or dict(task.contract) != expected_contract
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    def normalize(self, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        task_input = AgentTaskInput.from_authoring(value)
        if self.input_mode == "literal" and task_input.stored_prompt is None:
            validate_user_input(task_input.prompt)
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

    def normalize_durable(
        self,
        value: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        task_input = AgentTaskInput.from_mapping(value)
        if self.input_mode == "literal" and task_input.stored_prompt is None:
            validate_user_input(task_input.prompt)
        return dict(task_input)

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        task_input = AgentTaskInput.from_mapping(invocation.node.input)
        files = task_input.files
        request_identity: str | None = None
        if self.input_mode == "literal" or invocation.node.input.get("capture_fixed_input") is True:
            if task_input.parameters:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if task_input.stored_prompt is None:
                prompt = task_input.prompt
            else:
                if not isinstance(task_input.stored_prompt, StoredUserInput):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                prompt = await self._restore_prepared_prompt(task_input.stored_prompt)
                files = ()
            prompt = _append_captured_files(prompt, task_input)
        else:
            callback = self._build_input
            if callback is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            input_identity = _agent_task_input_identity(
                invocation,
                task_input,
                task_id=self.id,
                task_revision=self.revision,
                binding_digest=self._binding_digest,
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
                    self._binding_digest,
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
                prompt = _append_captured_files(prompt, task_input)
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
                    self._binding_digest,
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
                        "agent_binding_digest": self._binding_digest,
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
        execution, execution_id = await self._establish_execution(
            invocation,
            control,
            prompt,
            files=files,
            session_id=task_input.session_id,
            memory_scope=task_input.memory_scope,
            planning=planning,
            thinking=thinking,
            idempotency_key=request_identity,
        )
        result = await execution.wait()
        return _agent_task_result(result, execution_id)

    async def _establish_execution(
        self,
        invocation: TaskNodeInvocation,
        control: TaskNodeRunControl,
        prompt: CanonicalUserInput,
        *,
        files: tuple[str, ...],
        session_id: str | None,
        memory_scope: str | None,
        planning: bool,
        thinking: ThinkingValue,
        idempotency_key: str,
    ) -> tuple[object, str]:
        return await self._start_and_handoff(
            invocation,
            control,
            prompt,
            files=files,
            session_id=session_id,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            idempotency_key=idempotency_key,
        )

    async def _start_and_handoff(
        self,
        invocation: TaskNodeInvocation,
        control: TaskNodeRunControl,
        prompt: CanonicalUserInput,
        *,
        files: tuple[str, ...],
        session_id: str | None,
        memory_scope: str | None,
        planning: bool,
        thinking: ThinkingValue,
        idempotency_key: str,
    ) -> tuple[object, str]:
        hold_id = f"task:{invocation.graph_id}:{invocation.node.node_id}"
        if invocation.execution_id is None:
            execution = await self._start_execution(
                invocation,
                prompt,
                files=files,
                session_id=session_id,
                memory_scope=memory_scope,
                planning=planning,
                thinking=thinking,
                idempotency_key=idempotency_key,
                dependency_hold_id=hold_id,
            )
        else:
            execution = await self._get_execution(
                invocation.execution_id,
                invocation.principal,
            )
        execution_id = getattr(execution, "execution_id", None)
        if not isinstance(execution_id, str) or not execution_id:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        if invocation.execution_id is not None:
            await self._acquire_execution_hold(
                execution_id,
                invocation.principal,
                hold_id,
            )
        current = control.execution_id
        if current is None:
            await control.bind_execution(execution_id)
        elif current != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await control.handoff_execution(execution_id)
        await self._release_execution_hold(
            execution_id,
            invocation.principal,
            hold_id,
        )
        return execution, execution_id

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


def _append_captured_files(prompt: CanonicalUserInput, task_input: AgentTaskInput) -> CanonicalUserInput:
    payload = task_input.get("capture_files")
    if payload is None:
        return prompt
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    files = decode_task_prompt_draft(payload)
    if isinstance(files, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return (*((prompt,) if isinstance(prompt, str) else prompt), *files)


def _node_output_contract(invocation: TaskNodeInvocation) -> dict[str, JsonValue] | None:
    value = invocation.node.output_contract
    return None if value is None else dict(value)


def _agent_task_input_identity(
    invocation: TaskNodeInvocation,
    task_input: AgentTaskInput,
    *,
    task_id: str,
    task_revision: int,
    binding_digest: str,
) -> str:
    try:
        return canonical_sha256(
            {
                "version": 1,
                "graph_id": invocation.graph_id,
                "node_id": invocation.node.node_id,
                "task_id": task_id,
                "task_revision": task_revision,
                "binding_digest": binding_digest,
                "principal": principal_identity_payload(invocation.principal),
                "input": dict(task_input.execution_payload()),
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
    binding_digest: str,
) -> str:
    try:
        return canonical_sha256(
            {
                "version": 1,
                "graph_id": invocation.graph_id,
                "node_id": invocation.node.node_id,
                "task_id": task_id,
                "task_revision": task_revision,
                "agent_binding_digest": binding_digest,
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
