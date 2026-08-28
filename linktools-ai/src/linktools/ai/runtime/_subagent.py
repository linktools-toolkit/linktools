#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned one-level subagent dispatch and cancellation."""

import asyncio
from typing import cast

from linktools.core import environ
from ..agent import AgentCatalog, AgentCompiler
from ..core import ExecutionMode, ExecutionStatus, JsonValue, Principal, canonical_sha256
from ..errors import AIError, ErrorCode
from ..capability import SubagentDelegate
from ..spec import SubagentRef
from ._execution import DefaultExecutionService
from .service_api import CancelExecutionRequest, ExecutionRequest, ExecutionResult

_logger = environ.get_logger("ai.runtime.subagent")


class SubagentDispatcher:
    """Compile and run one-level child executions for a root execution."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        execution: DefaultExecutionService,
    ) -> None:
        self._catalog = catalog
        self._compiler = compiler
        self._execution = execution
        self._detached_tasks: set[asyncio.Task[object]] = set()

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return tuple(task for task in self._detached_tasks if not task.done())

    def descriptions_for(
        self,
        refs: "tuple[SubagentRef, ...]",
    ) -> "dict[str, str | None]":
        descriptions: dict[str, str | None] = {}
        for ref in refs:
            if ref.description is not None:
                descriptions[ref.id] = ref.description
        return descriptions

    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        refs: "tuple[SubagentRef, ...]",
        mode: ExecutionMode,
    ) -> SubagentDelegate:
        allowed = {ref.id: ref for ref in refs}
        if len(allowed) != len(refs):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)

        async def dispatch(
            ref: SubagentRef,
            task: str,
            *,
            invocation_id: str,
        ) -> "dict[str, JsonValue]":
            expected = allowed.get(ref.id)
            if expected is None or expected != ref:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            return await self.dispatch(
                parent_execution_id=parent_execution_id,
                root_execution_id=root_execution_id,
                memory_scope=memory_scope,
                principal=principal,
                ref=ref,
                mode=mode,
                user_prompt=task,
                invocation_id=invocation_id,
            )

        return dispatch

    async def dispatch(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        ref: SubagentRef,
        mode: ExecutionMode,
        user_prompt: str,
        invocation_id: str,
    ) -> "dict[str, JsonValue]":
        if not isinstance(ref, SubagentRef):
            raise TypeError("ref must be SubagentRef")
        if not isinstance(invocation_id, str) or not invocation_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        child_mode: ExecutionMode = "plan" if mode == "plan" else "run"
        idempotency_key = "subagent:" + canonical_sha256(
            {
                "version": 1,
                "parent_execution_id": parent_execution_id,
                "invocation_id": invocation_id,
            }
        )
        child = await self._execution.replay_subagent(
            agent_id=ref.id,
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode=child_mode,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
        )
        if child is None:
            definition = self._catalog.root_definition(ref.id)
            binding = self._catalog.register_binding(
                self._compiler.bind_subagent(definition)
            )
            child_planning = True if child_mode == "plan" else definition.spec.planning
            request = ExecutionRequest(
                user_prompt=user_prompt,
                user_prompt_codec="text",
                principal=principal,
                idempotency_key=idempotency_key,
                memory_scope=memory_scope,
                mode=child_mode,
                planning=child_planning,
                thinking=definition.spec.thinking,
            )
            try:
                child = await self._execution.start_subagent(
                    binding.digest,
                    request,
                    parent_execution_id=parent_execution_id,
                    root_execution_id=root_execution_id,
                )
            except AIError as error:
                if error.code is not ErrorCode.IDEMPOTENCY_CONFLICT:
                    raise
                child = await self._execution.replay_subagent(
                    agent_id=ref.id,
                    user_prompt=user_prompt,
                    principal=principal,
                    idempotency_key=idempotency_key,
                    memory_scope=memory_scope,
                    mode=child_mode,
                    parent_execution_id=parent_execution_id,
                    root_execution_id=root_execution_id,
                )
                if child is None:
                    raise
        try:
            result = await self._execution.wait(
                child.execution_id,
                principal=principal,
            )
        except BaseException as primary:  # noqa: BLE001
            cleanup = asyncio.create_task(
                self.cancel_child(
                    child.execution_id,
                    parent_execution_id=parent_execution_id,
                    principal=principal,
                ),
                name=f"ai-subagent-cleanup-{child.execution_id}",
            )
            if isinstance(primary, asyncio.CancelledError):
                self._detach(
                    cast("asyncio.Task[object]", cleanup),
                    "subagent child cleanup",
                )
                raise
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                self._detach(
                    cast("asyncio.Task[object]", cleanup),
                    "subagent child cleanup",
                )
                raise
            except BaseException:  # noqa: BLE001
                _logger.exception(
                    "subagent child cleanup failed: execution=%s",
                    child.execution_id,
                )
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from primary
            raise
        return _subagent_result(result)

    async def cancel_children(
        self,
        parent_execution_id: str,
        principal: Principal,
    ) -> None:
        children = await self._execution.list_children(
            parent_execution_id,
            principal=principal,
        )
        for child in sorted(children, key=lambda value: value.execution_id):
            if child.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                continue
            await self.cancel_child(
                child.execution_id,
                parent_execution_id=parent_execution_id,
                principal=principal,
            )

    async def cancel_child(
        self,
        execution_id: str,
        *,
        parent_execution_id: str,
        principal: Principal,
    ) -> None:
        current = await self._execution.inspect(
            execution_id,
            principal=principal,
        )
        if current.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return
        request = CancelExecutionRequest(
            principal=principal,
            idempotency_key="subagent-cancel:"
            + canonical_sha256(
                {
                    "version": 1,
                    "parent_execution_id": parent_execution_id,
                    "child_execution_id": execution_id,
                }
            ),
            force=True,
        )
        task = asyncio.create_task(
            self._cancel_child_operation(
                execution_id,
                request,
                principal,
            ),
            name=f"ai-subagent-cancel-{execution_id}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:  # noqa: BLE001
                    _logger.warning(
                        "subagent cancellation failed after caller cancellation: "
                        "execution=%s error=%s",
                        execution_id,
                        type(error).__name__,
                    )
            else:
                self._detach(
                    cast("asyncio.Task[object]", task),
                    "subagent cancellation",
                )
            raise
        except AIError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

    async def _cancel_child_operation(
        self,
        execution_id: str,
        request: CancelExecutionRequest,
        principal: Principal,
    ) -> None:
        try:
            result = await self._execution.cancel(execution_id, request)
        except AIError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if not result.cancelled:
            current = await self._execution.inspect(
                execution_id,
                principal=principal,
            )
            if current.status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        current = await self._execution.inspect(
            execution_id,
            principal=principal,
        )
        if current.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

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
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)


def _subagent_result(result: ExecutionResult) -> "dict[str, JsonValue]":
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
        "error_code": result.error_code,
        "safe_error_details": dict(result.safe_error_details),
    }


__all__ = ["SubagentDispatcher"]