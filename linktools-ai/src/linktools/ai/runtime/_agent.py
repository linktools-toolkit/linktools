#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound Agent, Session, and Execution behavior objects."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai.messages import UserContent

from ..core import JsonValue, Page, Principal, ThinkingValue
from ._input import prepare_user_prompt
from .service_api import (
    CancelExecutionResult,
    EvaluationHandle,
    ExecutionHistoryItem,
    ExecutionResult,
    ExecutionStreamEvent,
    ExecutionTraceItem,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    SessionHistoryItem,
    SessionView,
    TranscriptItem,
)

if TYPE_CHECKING:
    from ..task import TaskNode
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class Execution(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    execution_id: str
    _binding_digest: str
    _principal: Principal

    async def wait(self, *, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self._runtime.execution.wait(
            self.execution_id,
            principal=self._principal,
            timeout_seconds=timeout_seconds,
        )

    def stream(self, *, after_sequence: int = 0) -> AsyncIterator[ExecutionStreamEvent]:
        return self._runtime._execution_stream(
            self.execution_id,
            principal=self._principal,
            after_sequence=after_sequence,
        )

    async def cancel(
        self,
        *,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> CancelExecutionResult:
        return await self._runtime.cancel(
            self.execution_id,
            principal=self._principal,
            idempotency_key=idempotency_key,
            force=force,
        )

    async def retry(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        idempotency_key: "str | None" = None,
        context: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._retry_execution(
            self._binding_digest,
            self.execution_id,
            prepare_user_prompt(user_prompt),
            principal=self._principal,
            idempotency_key=idempotency_key,
            context=context,
        )

    async def fork(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        idempotency_key: "str | None" = None,
        context: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._fork_execution(
            self._binding_digest,
            self.execution_id,
            prepare_user_prompt(user_prompt),
            principal=self._principal,
            idempotency_key=idempotency_key,
            context=context,
        )

    async def history(
        self,
        *,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[ExecutionHistoryItem]":
        return await self._runtime.execution.history(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            limit=limit,
        )

    async def trace(
        self,
        *,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[ExecutionTraceItem]":
        return await self._runtime.execution.trace(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            limit=limit,
        )

    async def transcript(
        self,
        *,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[TranscriptItem]":
        return await self._runtime.execution.transcript(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            limit=limit,
        )


@dataclass(frozen=True, slots=True)
class Session(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    agent_id: str
    _agent_digest: str
    session_id: str
    _principal: "Principal | None" = None

    async def start(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._start_for_agent(
            self._agent_digest,
            prepare_user_prompt(user_prompt),
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="run",
            planning=planning,
            thinking=thinking,
            context=context,
        )

    async def run(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        execution = await self.start(
            user_prompt,
            output=output,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            context=context,
        )
        return await execution.wait(timeout_seconds=timeout_seconds)

    async def plan(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        execution = await self._runtime._start_for_agent(
            self._agent_digest,
            prepare_user_prompt(user_prompt),
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="plan",
            planning=True,
            thinking=thinking,
            context=context,
        )
        return await execution.wait(timeout_seconds=timeout_seconds)

    async def history(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        return await self._runtime.session.history(
            self.session_id,
            principal=self._runtime._resolve_principal(principal or self._principal),
            cursor=cursor,
            limit=limit,
        )

    async def fork(
        self,
        new_session_id: str,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
    ) -> "Session[AppT]":
        return await self._runtime._fork_session(
            self.agent_id,
            self._agent_digest,
            self.session_id,
            new_session_id,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            cwd=cwd,
        )

    async def update(
        self,
        *,
        expected_revision: int,
        metadata: Mapping[str, JsonValue],
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
    ) -> SessionView:
        return await self._runtime._update_session(
            self.agent_id,
            self.session_id,
            expected_revision=expected_revision,
            metadata=metadata,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            cwd=cwd,
        )

    async def close(
        self,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        force: bool = False,
        wait_timeout_seconds: int = 30,
    ) -> SessionView:
        return await self._runtime._close_session(
            self.session_id,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            force=force,
            wait_timeout_seconds=wait_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class Agent(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    id: str
    _agent_digest: str

    async def start(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._start_for_agent(
            self._agent_digest,
            prepare_user_prompt(user_prompt),
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="run",
            planning=planning,
            thinking=thinking,
            context=context,
        )

    async def run(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        execution = await self.start(
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            context=context,
        )
        return await execution.wait(timeout_seconds=timeout_seconds)

    async def plan(
        self,
        user_prompt: "str | Sequence[UserContent]",
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        context: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        execution = await self._runtime._start_for_agent(
            self._agent_digest,
            prepare_user_prompt(user_prompt),
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="plan",
            planning=True,
            thinking=thinking,
            context=context,
        )
        return await execution.wait(timeout_seconds=timeout_seconds)

    def session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
    ) -> "Session[AppT]":
        return Session(self._runtime, self.id, self._agent_digest, session_id, principal)

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
        idempotency_key: "str | None" = None,
    ) -> "Session[AppT]":
        await self._runtime._create_session_for_agent(
            self.id,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
        return Session(self._runtime, self.id, self._agent_digest, session_id, principal)

    async def run_evaluation(
        self,
        request: RunEvaluationRequest,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> EvaluationHandle:
        return await self._runtime._run_evaluation_for_agent(
            self._agent_digest,
            request,
            output=output,
        )

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._replay_evaluation_for_agent(
            self._agent_digest,
            snapshot_id,
            request,
            output=output,
        )

    def task(
        self,
        node_id: str,
        user_prompt: "str | Sequence[UserContent]",
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: "type[BaseModel] | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
    ) -> "TaskNode":
        return self._runtime._task_for_agent(
            self._agent_digest,
            node_id,
            prepare_user_prompt(user_prompt),
            dependencies=dependencies,
            budget_cost=budget_cost,
            output=output,
            planning=planning,
            thinking=thinking,
        )


__all__ = ["Agent", "Execution", "Session"]
