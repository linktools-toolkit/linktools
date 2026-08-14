#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-owned subagent dispatch and cancellation coordination."""

import asyncio

from pydantic_ai.exceptions import ModelRetry

from ..agent import AgentCompiler, AgentDefinition, SubagentDelegate
from ..core import ExecutionStatus, JsonValue, Principal, canonical_sha256
from ..errors import AIError, ErrorCode
from ..runtime import (
    CancelExecutionRequest,
    DefaultExecutionService,
    ExecutionRequest,
    ExecutionResult,
)


class SubagentDispatcher:
    """Compile and run one-level child executions for a root execution."""

    def __init__(
        self,
        compiler: AgentCompiler,
        definitions: "dict[str, AgentDefinition]",
        execution: DefaultExecutionService,
    ) -> None:
        self._compiler = compiler
        self._definitions = definitions
        self._execution = execution

    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
    ) -> SubagentDelegate:
        async def dispatch(*, tool_call_id: str, agent_id: str, prompt_id: str, task: str) -> "dict[str, JsonValue]":
            return await self.dispatch(
                parent_execution_id=parent_execution_id,
                root_execution_id=root_execution_id,
                memory_scope=memory_scope,
                principal=principal,
                tool_call_id=tool_call_id,
                agent_id=agent_id,
                prompt_id=prompt_id,
                task=task,
            )

        return dispatch

    async def dispatch(
        self,
        *,
        parent_execution_id: str,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        tool_call_id: str,
        agent_id: str,
        prompt_id: str,
        task: str,
    ) -> "dict[str, JsonValue]":
        try:
            definition = await self._compiler.compile_subagent(agent_id=agent_id, prompt_id=prompt_id)
        except AIError as error:
            if error.code in {
                ErrorCode.AGENT_NOT_FOUND,
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                ErrorCode.STORAGE_NOT_FOUND,
                ErrorCode.OUTPUT_SCHEMA_UNKNOWN,
            }:
                raise ModelRetry("requested subagent is unavailable") from error
            raise
        self._definitions[definition.digest] = definition
        idempotency_key = "subagent:" + canonical_sha256(
            {"version": 1, "parent_execution_id": parent_execution_id, "tool_call_id": tool_call_id}
        )
        request = ExecutionRequest(
            prompt=task,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )
        child = await self._execution.start_subagent(
            definition.digest,
            request,
            parent_execution_id=parent_execution_id,
            root_execution_id=root_execution_id,
        )
        try:
            result = await self._execution.wait(child.execution_id, principal=principal)
        except asyncio.CancelledError:
            await self.cancel_child(child.execution_id, parent_execution_id=parent_execution_id, principal=principal)
            raise
        return _subagent_result(result)

    async def cancel_children(self, parent_execution_id: str, principal: Principal) -> None:
        children = await self._execution.list_children(parent_execution_id, principal=principal)
        for child in sorted(children, key=lambda value: value.execution_id):
            if child.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                continue
            await self.cancel_child(child.execution_id, parent_execution_id=parent_execution_id, principal=principal)

    async def cancel_child(self, execution_id: str, *, parent_execution_id: str, principal: Principal) -> None:
        current = await self._execution.inspect(execution_id, principal=principal)
        if current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return
        request = CancelExecutionRequest(
            principal=principal,
            idempotency_key="subagent-cancel:" + canonical_sha256(
                {
                    "version": 1,
                    "parent_execution_id": parent_execution_id,
                    "child_execution_id": execution_id,
                }
            ),
            force=True,
        )
        task = asyncio.create_task(
            self._execution.cancel(execution_id, request),
            name=f"ai-subagent-cancel-{execution_id}",
        )
        cancelled = False
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            try:
                result = await task
            except BaseException as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        except BaseException as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if not result.cancelled:
            current = await self._execution.inspect(execution_id, principal=principal)
            if current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        current = await self._execution.inspect(execution_id, principal=principal)
        if current.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if cancelled:
            raise asyncio.CancelledError


def _subagent_result(result: ExecutionResult) -> "dict[str, JsonValue]":
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
    }


__all__ = ["SubagentDispatcher"]
