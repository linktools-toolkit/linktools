#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound Agent, Session, and Execution behavior objects."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
import secrets
import asyncio
import sys
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel
from ..core import JsonValue, Page, Principal, ThinkingValue
from ..errors import AIError, ErrorCode, ObservationError
from ._wait import WaitResult
from ._observation import _wait, _validate_wait, _await_stream_cleanup, _is_observation_cleanup
from ._execution_context import ExecutionInputContext
from ._input_contract import UserPromptInput, validate_user_input
from ._watch_cursor import (
    decode_execution_watch_cursor,
    encode_execution_watch_cursor,
)
from .recovery import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectResolution,
    ToolEffectResolutionResult,
)
from .service_api import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionEvent,
    ExecutionHistoryItem,
    ExecutionResult,
    ExecutionTraceItem,
    ExecutionTreeEvent,
    ModelInteractionItem,
    SessionHistoryItem,
    SessionTurn,
    SessionView,
    TranscriptItem,
    UsageReadCutoff,
    _ExecutionStreamFailure,
)

if TYPE_CHECKING:
    from ..agent import CompiledAgent
    from ._runtime_service import Runtime

AppT = TypeVar("AppT")


class _ExecutionTreeWatcher(Protocol):
    def __call__(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_event_seqs: "Mapping[str, int] | None" = None,
        include_content: bool = False,
        ready: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...


@dataclass(frozen=True, slots=True)
class Execution(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    execution_id: str
    _principal: Principal
    _watch_tree: _ExecutionTreeWatcher
    _task_wait: Callable[[float | None], Awaitable[ExecutionResult]] | None = None
    _task_cancel: Callable[
        [str | None, bool], Awaitable[CancelExecutionResult]
    ] | None = None

    async def wait(
        self, *, on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[ExecutionResult]:
        _validate_wait(on_event, cursor, include_content, timeout_seconds, close_timeout_seconds)
        return await _wait(
            scope="execution", resource_id=self.execution_id,
            waiter=(lambda: self._task_wait(None)) if self._task_wait is not None else
                lambda: self._runtime._execution_service.wait(
                    self.execution_id, principal=self._principal),
            watch=lambda ready: self._watch_prepared(cursor, include_content, ready),
            on_event=on_event, cursor=cursor, timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            register=self._runtime._register_observation, release=self._runtime._release_observation,
        )

    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        return self._watch_prepared(cursor, include_content, None)

    def _watch_prepared(
        self, cursor: str | None, include_content: bool, ready: asyncio.Event | None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        sequences = None if cursor is None else decode_execution_watch_cursor(
            self._runtime.namespace, self._principal.tenant_id, self.execution_id,
            cursor, include_content=include_content,
        )
        stream = self._watch_tree(
            self.execution_id, principal=self._principal, after_event_seqs=sequences,
            include_content=include_content, ready=ready,
        )
        return self._watch_with_cursor(stream, sequences, include_content, cursor)

    async def _watch_with_cursor(
        self, stream: AsyncIterator[ExecutionTreeEvent],
        after_event_seqs: Mapping[str, int] | None, include_content: bool,
        cursor: str | None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        sequences = dict(after_event_seqs or {})
        last_cursor = cursor
        try:
            async for event in stream:
                durable_seq = event.event.durable_seq
                if durable_seq is not None:
                    previous = sequences.get(event.execution_id, 0)
                    if durable_seq <= previous:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    sequences[event.execution_id] = durable_seq
                last_cursor = encode_execution_watch_cursor(
                    self._runtime.namespace, self._principal.tenant_id, self.execution_id,
                    include_content=include_content, event_seqs=sequences,
                )
                yield replace(event, cursor=last_cursor)
        except _ExecutionStreamFailure as failure:
            cause = failure.cause
            raise ObservationError(
                "stream", cursor=last_cursor,
                cause_code=cause.code.value if isinstance(cause, AIError) else None,
                safe_details=cause.safe_details if isinstance(cause, AIError) else None,
                diagnostics=cause.diagnostics if isinstance(cause, AIError) else None,
            ) from cause
        finally:
            active_error = sys.exc_info()[1]
            try:
                await _await_stream_cleanup(stream.aclose(), active_error)
            except _ExecutionStreamFailure as failure:
                if active_error is None or isinstance(active_error, GeneratorExit) or _is_observation_cleanup(active_error):
                    cause = failure.cause
                    raise ObservationError(
                        "stream", cursor=last_cursor,
                        cause_code=cause.code.value if isinstance(cause, AIError) else None,
                        safe_details={"phase": "cleanup"},
                        diagnostics=cause.diagnostics if isinstance(cause, AIError) else None,
                    ) from cause
            except BaseException as error:
                if (isinstance(error, asyncio.CancelledError) or active_error is None
                        or isinstance(active_error, GeneratorExit) or _is_observation_cleanup(active_error)
                        or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                    raise

    async def cancel(
        self,
        *,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> CancelExecutionResult:
        if self._task_cancel is not None:
            return await self._task_cancel(idempotency_key, force)
        return await self._runtime.executions.cancel(
            self.execution_id,
            CancelExecutionRequest(
                self._principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
            ),
        )

    async def recovery_effects(self) -> tuple[ExecutionRecoveryEffect, ...]:
        return await self._runtime.executions.recovery_effects(
            self.execution_id,
            principal=self._principal,
        )

    async def resolve_tool_effect(
        self,
        operation_id: str,
        *,
        expected_fence: int,
        resolution: ToolEffectResolution,
        idempotency_key: str,
    ) -> ToolEffectResolutionResult:
        return await self._runtime.executions.resolve_tool_effect(
            self.execution_id,
            ResolveToolEffectRequest(
                self._principal,
                operation_id,
                expected_fence,
                resolution,
                idempotency_key,
            ),
        )

    async def recover(self) -> "Execution[AppT]":
        await self._runtime.executions.recover(
            self.execution_id,
            principal=self._principal,
        )
        return self

    async def retry(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        idempotency_key: "str | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._retry_execution(
            self.execution_id,
            validate_user_input(user_prompt),
            files=files,
            principal=self._principal,
            idempotency_key=idempotency_key,
            correlation=correlation,
        )

    async def fork(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        idempotency_key: "str | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        return await self._runtime._fork_execution(
            self.execution_id,
            validate_user_input(user_prompt),
            files=files,
            principal=self._principal,
            idempotency_key=idempotency_key,
            correlation=correlation,
        )

    async def list_events(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> "Page[ExecutionEvent]":
        history = self._runtime.history
        if history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await history.list_execution_events(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    async def history(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> "Page[ExecutionHistoryItem]":
        return await self._runtime.executions.history(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
            message_seq=message_seq,
            part_index=part_index,
        )

    async def trace(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]":
        return await self._runtime.executions.trace(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            agent_run_seq=agent_run_seq,
            model_request_seq=model_request_seq,
            step_index=step_index,
            tool_call_id=tool_call_id,
        )

    async def transcript(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
    ) -> "Page[TranscriptItem]":
        return await self._runtime.executions.transcript(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
        )

    async def model_interactions(
        self,
        *,
        cursor: "str | None" = None,
        include_content: bool = False,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> "Page[ModelInteractionItem]":
        return await self._runtime.executions.model_interactions(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            limit=limit,
            cutoffs=cutoffs,
        )


@dataclass(frozen=True, slots=True)
class Session(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    agent_id: str
    _agent_revision: int | None
    session_id: str
    _principal: "Principal | None" = None
    _compiled_agent: "CompiledAgent | None" = None

    async def start(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        if self._agent_revision is None:
            return await self._runtime.agents.get(self.agent_id).start(
                user_prompt,
                files=files,
                output=output,
                principal=principal or self._principal,
                session_id=self.session_id,
                idempotency_key=idempotency_key,
                memory_scope=memory_scope,
                planning=planning,
                thinking=thinking,
                correlation=correlation,
            )
        return await self._runtime._start_for_agent(
            self.agent_id,
            self._agent_revision,
            validate_user_input(user_prompt),
            files=files,
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="run",
            planning=planning,
            thinking=thinking,
            correlation=correlation,
            compiled_agent=self._compiled_agent,
        )

    async def run(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_content: bool = False,
        close_timeout_seconds: float = 5.0,
    ) -> WaitResult[ExecutionResult]:
        _validate_wait(on_event, None, include_content, timeout_seconds, close_timeout_seconds)
        execution = await self.start(
            user_prompt,
            files=files,
            output=output,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_content=include_content,
            close_timeout_seconds=close_timeout_seconds,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_content: bool = False,
        close_timeout_seconds: float = 5.0,
    ) -> WaitResult[ExecutionResult]:
        _validate_wait(on_event, None, include_content, timeout_seconds, close_timeout_seconds)
        if self._agent_revision is None:
            return await self._runtime.agents.get(self.agent_id).plan(
                user_prompt,
                files=files,
                output=output,
                principal=principal or self._principal,
                session_id=self.session_id,
                idempotency_key=idempotency_key,
                memory_scope=memory_scope,
                thinking=thinking,
                correlation=correlation,
                timeout_seconds=timeout_seconds, on_event=on_event, include_content=include_content,
                close_timeout_seconds=close_timeout_seconds,
            )
        execution = await self._runtime._start_for_agent(
            self.agent_id,
            self._agent_revision,
            validate_user_input(user_prompt),
            files=files,
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="plan",
            planning=True,
            thinking=thinking,
            correlation=correlation,
            compiled_agent=self._compiled_agent,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_content=include_content,
            close_timeout_seconds=close_timeout_seconds,
        )

    async def history(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem]":
        return await self._runtime.sessions.history(
            self.session_id,
            principal=self._runtime._resolve_principal(principal or self._principal),
            cursor=cursor,
            limit=limit,
        )

    async def timeline(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionTurn]":
        return await self._runtime.sessions.timeline(
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
        if self._agent_revision is None:
            agent = self._runtime.agents.get(self.agent_id)
            return await self._runtime._fork_session(
                agent.id,
                agent.revision,
                self.session_id,
                new_session_id,
                principal=principal or self._principal,
                idempotency_key=idempotency_key,
                cwd=cwd,
                compiled_agent=agent.compiled,
            )
        return await self._runtime._fork_session(
            self.agent_id,
            self._agent_revision,
            self.session_id,
            new_session_id,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            cwd=cwd,
            compiled_agent=self._compiled_agent,
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
    _agent_revision: int
    _compiled_agent: "CompiledAgent | None" = None

    @property
    def runtime(self) -> "Runtime[AppT]":
        """Runtime that owns this Agent definition."""
        return self._runtime

    @property
    def revision(self) -> int:
        return self._agent_revision

    @property
    def compiled(self) -> "CompiledAgent | None":
        return self._compiled_agent

    def derive(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        instructions: Sequence[str] | None = None,
        allow_tools: Sequence[str] | None = None,
        allow_skills: Sequence[str] | None = None,
    ) -> "Agent[AppT]":
        return self._runtime._derive_agent(
            self,
            model=model,
            system_prompt=system_prompt,
            instructions=instructions,
            allow_tools=allow_tools,
            allow_skills=allow_skills,
        )

    async def start(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        input_context: ExecutionInputContext | None = None,
    ) -> "Execution[AppT]":
        return await self._runtime._start_for_agent(
            self.id,
            self._agent_revision,
            validate_user_input(user_prompt),
            files=files,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="run",
            planning=planning,
            thinking=thinking,
            correlation=correlation,
            compiled_agent=self._compiled_agent,
            input_context=input_context,
        )

    async def run(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        input_context: ExecutionInputContext | None = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_content: bool = False,
        close_timeout_seconds: float = 5.0,
    ) -> WaitResult[ExecutionResult]:
        _validate_wait(on_event, None, include_content, timeout_seconds, close_timeout_seconds)
        execution = await self.start(
            user_prompt,
            files=files,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
            input_context=input_context,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_content=include_content,
            close_timeout_seconds=close_timeout_seconds,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_content: bool = False,
        close_timeout_seconds: float = 5.0,
    ) -> WaitResult[ExecutionResult]:
        _validate_wait(on_event, None, include_content, timeout_seconds, close_timeout_seconds)
        execution = await self._runtime._start_for_agent(
            self.id,
            self._agent_revision,
            validate_user_input(user_prompt),
            files=files,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode="plan",
            planning=True,
            thinking=thinking,
            correlation=correlation,
            compiled_agent=self._compiled_agent,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_content=include_content,
            close_timeout_seconds=close_timeout_seconds,
        )

    def session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
    ) -> "Session[AppT]":
        return Session(
            self._runtime,
            self.id,
            self._agent_revision,
            session_id,
            principal,
            self._compiled_agent,
        )

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
        return Session(
            self._runtime,
            self.id,
            self._agent_revision,
            session_id,
            principal,
            self._compiled_agent,
        )

__all__ = ["Agent", "Execution", "Session"]
