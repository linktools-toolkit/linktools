#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound Agent execution handle."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import JsonValue, Principal
from ..errors import AIError, ErrorCode
from .service_api import (
    EvaluationHandle,
    ExecutionHandle,
    ExecutionResult,
    ExecutionStreamEvent,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    SessionView,
)

if TYPE_CHECKING:
    from ..task import TaskNode
    from ._runtime_service import Runtime


@dataclass(frozen=True, slots=True)
class AgentHandle:
    _runtime: "Runtime"
    agent_id: str
    binding_digest: str
    planning: bool = False
    thinking: bool = False

    async def start(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> ExecutionHandle:
        return await self._runtime._start_for_binding(
            self.binding_digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=self.planning,
            thinking=self.thinking,
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
        return await self._runtime._run_for_binding(
            self.binding_digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=self.planning,
            thinking=self.thinking,
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
    ) -> AsyncIterator[ExecutionStreamEvent]:
        return self._runtime._stream_for_binding(
            self.binding_digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=self.planning,
            thinking=self.thinking,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> SessionView:
        return await self._runtime._create_session_for_binding(
            self.binding_digest,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
        )

    async def run_evaluation(self, request: RunEvaluationRequest) -> EvaluationHandle:
        return await self._runtime._run_evaluation_for_binding(self.binding_digest, request)

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        return await self._runtime._replay_evaluation_for_binding(
            self.binding_digest,
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
        definition = self._runtime._definition(self.binding_digest)
        if definition.spec.id != self.agent_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": "linktools.ai.agent",
                "version": 1,
                "agent_id": self.agent_id,
                "binding_digest": self.binding_digest,
                "binding": definition.binding_snapshot.to_payload(),
                "user_prompt": user_prompt,
                "planning": self.planning,
                "thinking": self.thinking,
            },
            budget_cost=budget_cost,
        )


__all__ = ["AgentHandle"]
