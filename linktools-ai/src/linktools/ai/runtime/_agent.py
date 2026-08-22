#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound Agent execution handle."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pydantic import BaseModel
from ..core import JsonValue, Principal
from .service_api import EvaluationHandle, ExecutionHandle, ExecutionResult, ExecutionStreamEvent, ReplayEvaluationRequest, RunEvaluationRequest, SessionView

if TYPE_CHECKING:
    from ..task import TaskNode
    from ._runtime_service import Runtime


@dataclass(frozen=True, slots=True)
class AgentHandle:
    _runtime: "Runtime"
    agent_id: str
    _agent_digest: str

    async def start(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False) -> ExecutionHandle:
        return await self._runtime._start_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking)

    async def run(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self._runtime._run_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking, timeout_seconds=timeout_seconds)

    def stream(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False) -> AsyncIterator[ExecutionStreamEvent]:
        return self._runtime._stream_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking)

    async def create_session(self, session_id: str, *, principal: "Principal | None" = None, cwd: "str | None" = None, metadata: "Mapping[str, JsonValue] | None" = None) -> SessionView:
        return await self._runtime._create_session_for_agent(self._agent_digest, session_id, principal=principal, cwd=cwd, metadata=metadata)

    async def run_evaluation(self, request: RunEvaluationRequest, *, output: "type[BaseModel] | None" = None) -> EvaluationHandle:
        return await self._runtime._run_evaluation_for_agent(self._agent_digest, request, output=output)

    async def replay_evaluation(self, snapshot_id: str, request: ReplayEvaluationRequest, *, output: "type[BaseModel] | None" = None) -> ExecutionHandle:
        return await self._runtime._replay_evaluation_for_agent(self._agent_digest, snapshot_id, request, output=output)

    def task(self, node_id: str, user_prompt: str, *, dependencies: tuple[str, ...] = (), budget_cost: int = 1, output: "type[BaseModel] | None" = None, planning: bool = False, thinking: bool = False) -> "TaskNode":
        return self._runtime._task_for_agent(self._agent_digest, node_id, user_prompt, dependencies=dependencies, budget_cost=budget_cost, output=output, planning=planning, thinking=thinking)


__all__ = ["AgentHandle"]
