#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named Agent execution handle."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import JsonValue, Principal
from .service_api import (
    EvaluationHandle,
    ExecutionEvent,
    ExecutionHandle,
    ExecutionResult,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    SessionView,
)

if TYPE_CHECKING:
    from ..agent import AgentDefinition
    from ..task import TaskNode
    from ._runtime_service import Runtime


@dataclass(frozen=True, slots=True)
class AgentHandle:
    _runtime: "Runtime"
    agent_id: str

    async def compile(self) -> "AgentDefinition":
        return await self._runtime._compile_agent(self.agent_id)

    async def start(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> ExecutionHandle:
        return await self._runtime._start_for_agent(
            self.agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    async def run(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        return await self._runtime._run_for_agent(
            self.agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            timeout_seconds=timeout_seconds,
        )

    def stream(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> AsyncIterator[ExecutionEvent]:
        return self._runtime._stream_for_agent(
            self.agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> SessionView:
        return await self._runtime._create_session_for_agent(
            self.agent_id,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
        )

    async def run_evaluation(self, request: RunEvaluationRequest) -> EvaluationHandle:
        return await self._runtime._run_evaluation_for_agent(self.agent_id, request)

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        return await self._runtime._replay_evaluation_for_agent(
            self.agent_id,
            snapshot_id,
            request,
        )

    def task(
        self,
        node_id: str,
        user_prompt: str,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
    ) -> "TaskNode":
        from ..core import validate_user_prompt
        from ..task import TaskNode

        validate_user_prompt(user_prompt)
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": "linktools.ai.agent",
                "version": 1,
                "agent_id": self.agent_id,
                "user_prompt": user_prompt,
            },
            budget_cost=budget_cost,
        )


__all__ = ["AgentHandle"]
