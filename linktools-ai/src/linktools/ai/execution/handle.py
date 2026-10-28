#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent and Session behavior with independently owned Execution handles."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
import secrets
import asyncio
import sys
from typing import Literal, overload, TYPE_CHECKING, Awaitable, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel
from ..core import BudgetUsage, RunBudget, JsonValue, Page, Principal, ThinkingValue
from ..errors import AIError, ErrorCode, ObservationError
from .stream import SharedStreamEvent
from ..observe import WaitResult
from ._observation import _ExecutionModelObservation
from ..observe.stream import wait_for_result, validate_wait, await_stream_cleanup, is_observation_cleanup, owned_watch
from ._input_context import ExecutionInputContext
from ..model import ModelBinding, UserPromptInput, validate_user_input
from ..observe.cursors import decode_execution_watch_cursor, get_preview_positions, ExecutionWatchPositions, encode_execution_watch_cursor
from ._recovery import ExecutionRecoveryEffect, ResolveToolEffectRequest, ToolEffectResolution, ToolEffectResolutionResult
from ._service_api import CancelExecutionRequest, CancelExecutionResult, ExecutionEvent, ExecutionHistoryItem, ExecutionResult, ExecutionTraceItem, ExecutionTreeEvent, ExecutionObservationEvent, ModelInteractionItem, SessionHistoryItem, SessionTurn, SessionView, TranscriptItem, UsageReadCutoff
from ..observe.stream import ExecutionStreamFailure

if TYPE_CHECKING:
    from ..agent import CompiledAgent
    from ._service_api import ExecutionService, ExecutionView, ExecutionHistoryService
    from ..observe.stream import ObservationSession

AppT = TypeVar("AppT")


class ExecutionTreeWatcher(Protocol):
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
class ExecutionBinding:
    """Borrowed operations used by an execution handle's producing domain."""

    namespace: str
    service: ExecutionService
    ensure_open: Callable[[], None]
    register_observation: Callable[[ObservationSession], None]
    release_observation: Callable[[ObservationSession], None]
    capture_tree: Callable[..., Awaitable[tuple[tuple[ExecutionView, int, int], ...]]]
    replay_tree: Callable[..., AsyncIterator[ExecutionTreeEvent]]
    history: ExecutionHistoryService | None = None
    list_events: Callable[..., Awaitable[Page[ExecutionEvent]]] | None = None
    retry: Callable[..., Awaitable[Execution[object]]] | None = None
    fork: Callable[..., Awaitable[Execution[object]]] | None = None


@dataclass(frozen=True, slots=True)
class Execution(Generic[AppT]):
    _binding: ExecutionBinding
    execution_id: str
    _principal: Principal
    _watch_tree: ExecutionTreeWatcher
    _task_wait: Callable[[float | None], Awaitable[ExecutionResult]] | None = None
    _task_cancel: Callable[
        [str | None, bool], Awaitable[CancelExecutionResult]
    ] | None = None

    @overload
    async def wait(
        self, *, on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: Literal[False] = False,
    ) -> WaitResult[ExecutionResult]: ...

    @overload
    async def wait(
        self, *, on_event: Callable[[ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: Literal[True],
    ) -> WaitResult[ExecutionResult]: ...

    @overload
    async def wait(
        self, *, on_event: Callable[[ExecutionTreeEvent | ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: bool,
    ) -> WaitResult[ExecutionResult]: ...

    async def wait(
        self, *, on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | Callable[[ExecutionObservationEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
        include_model_interactions: bool = False,
    ) -> WaitResult[ExecutionResult]:
        """Wait for the full execution result; include_event_content only selects callback payloads."""
        validate_wait(on_event, cursor, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        if not isinstance(include_model_interactions, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if include_model_interactions:
            return await wait_for_result(
                scope="execution", resource_id=self.execution_id,
                waiter=(lambda: self._task_wait(None)) if self._task_wait is not None else
                    lambda: self._binding.service.wait(self.execution_id, principal=self._principal),
                watch=lambda ready: self._watch_models(cursor, include_event_content, ready),
                on_event=on_event, cursor=cursor, timeout_seconds=timeout_seconds,
                close_timeout_seconds=close_timeout_seconds,
                raise_on_stream_error=raise_on_stream_error,
                register=self._binding.register_observation, release=self._binding.release_observation,
                finalize=lambda result, acknowledged: _ExecutionModelObservation(
                    self._binding, self.execution_id, self._principal, self._watch_tree,
                    acknowledged, include_event_content,
                ).final(),
            )

        async def finalize(result: ExecutionResult, last_cursor: str | None) -> AsyncIterator[ExecutionTreeEvent]:
            sequences = None if last_cursor is None else decode_execution_watch_cursor(
                self._binding.namespace, self._principal.tenant_id, self.execution_id,
                last_cursor, include_content=include_event_content,
            )
            captured = await self._binding.capture_tree(
                self.execution_id, principal=self._principal, after_event_seqs=sequences,
            )
            return self._watch_with_cursor(
                self._binding.replay_tree(
                    captured, principal=self._principal, after_event_seqs=sequences,
                    include_content=include_event_content,
                ),
                sequences, include_event_content, last_cursor,
            )

        return await wait_for_result(
            scope="execution", resource_id=self.execution_id,
            waiter=(lambda: self._task_wait(None)) if self._task_wait is not None else
                lambda: self._binding.service.wait(
                    self.execution_id, principal=self._principal),
            watch=lambda ready: self._watch_prepared(cursor, include_event_content, ready),
            on_event=on_event, cursor=cursor, timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
            register=self._binding.register_observation, release=self._binding.release_observation,
            finalize=finalize, drain_live=True,
        )

    @overload
    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
        include_model_interactions: Literal[False] = False,
    ) -> AsyncIterator[ExecutionTreeEvent]: ...

    @overload
    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
        include_model_interactions: Literal[True],
    ) -> AsyncIterator[ExecutionObservationEvent]: ...

    @overload
    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
        include_model_interactions: bool,
    ) -> AsyncIterator[ExecutionTreeEvent | ExecutionObservationEvent]: ...

    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
        include_model_interactions: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent | ExecutionObservationEvent]:
        """Watch descendants, optionally combining metadata-only model projections."""
        if not isinstance(include_model_interactions, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if include_model_interactions:
            finalizing = asyncio.Event()
            return owned_watch(
                self._watch_models(cursor, include_content, None, finalizing),
                scope="execution", resource_id=self.execution_id, cursor=cursor,
                finalizing=finalizing,
                register=self._binding.register_observation, release=self._binding.release_observation,
            )
        return owned_watch(
            self._watch_prepared(cursor, include_content, None),
            scope="execution", resource_id=self.execution_id, cursor=cursor,
            finalizing=asyncio.Event(),
            register=self._binding.register_observation, release=self._binding.release_observation,
        )

    def _watch_models(
        self, cursor: str | None, include_content: bool, ready: asyncio.Event | None,
        finalizing: asyncio.Event | None = None,
    ) -> AsyncIterator[ExecutionObservationEvent]:
        self._binding.ensure_open()
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return _ExecutionModelObservation(
            self._binding, self.execution_id, self._principal, self._watch_tree,
            cursor, include_content, finalizing,
        ).watch(ready)

    def _watch_prepared(
        self, cursor: str | None, include_content: bool, ready: asyncio.Event | None,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        self._binding.ensure_open()
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        sequences = None if cursor is None else decode_execution_watch_cursor(
            self._binding.namespace, self._principal.tenant_id, self.execution_id,
            cursor, include_content=include_content,
        )
        sequences = ExecutionWatchPositions(sequences or {}, get_preview_positions(sequences))
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
        previews = get_preview_positions(after_event_seqs)
        last_cursor = cursor
        try:
            async for event in stream:
                durable_seq = event.event.durable_seq
                if durable_seq is not None:
                    previous = sequences.get(event.execution_id, 0)
                    if durable_seq <= previous:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    sequences[event.execution_id] = durable_seq
                if isinstance(event.event, SharedStreamEvent):
                    if event.event.position[0]:
                        previews[event.execution_id] = event.event.position
                    else:
                        previews.pop(event.execution_id, None)
                last_cursor = encode_execution_watch_cursor(
                    self._binding.namespace, self._principal.tenant_id, self.execution_id,
                    include_content=include_content, event_seqs=sequences, preview_positions=previews,
                )
                yield replace(event, cursor=last_cursor)
        except ExecutionStreamFailure as failure:
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
                await await_stream_cleanup(stream.aclose(), active_error)
            except ExecutionStreamFailure as failure:
                if active_error is None or isinstance(active_error, GeneratorExit) or is_observation_cleanup(active_error):
                    cause = failure.cause
                    raise ObservationError(
                        "stream", cursor=last_cursor,
                        cause_code=cause.code.value if isinstance(cause, AIError) else None,
                        safe_details={"phase": "cleanup"},
                        diagnostics=cause.diagnostics if isinstance(cause, AIError) else None,
                    ) from cause
            except BaseException as error:
                if (isinstance(error, asyncio.CancelledError) or active_error is None
                        or isinstance(active_error, GeneratorExit) or is_observation_cleanup(active_error)
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
        self._binding.ensure_open()
        return await self._binding.service.cancel(
            self.execution_id,
            CancelExecutionRequest(
                self._principal,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                force,
            ),
        )

    async def budget_usage(self) -> BudgetUsage | None:
        """Read the shared scope totals, including related executions."""
        self._binding.ensure_open()
        return await self._binding.service.budget_usage(
            self.execution_id, principal=self._principal,
        )

    async def recovery_effects(self) -> tuple[ExecutionRecoveryEffect, ...]:
        return await self._binding.service.recovery_effects(
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
        self._binding.ensure_open()
        return await self._binding.service.resolve_tool_effect(
            self.execution_id,
            ResolveToolEffectRequest(
                self._principal,
                operation_id,
                expected_fence,
                resolution,
                idempotency_key,
            ),
        )

    async def recover(
        self, *, idempotency_key: str | None = None,
    ) -> "Execution[AppT]":
        self._binding.ensure_open()
        await self._binding.service.recover(
            self.execution_id,
            principal=self._principal,
            idempotency_key=idempotency_key,
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
        if self._binding.retry is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"operation": "retry"})
        return await self._binding.retry(
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
        budget: RunBudget | None = None,
        idempotency_key: "str | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        """Fork with the existing shared budget unless an explicit new budget is supplied."""
        if self._binding.fork is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"operation": "fork"})
        return await self._binding.fork(
            self.execution_id,
            validate_user_input(user_prompt),
            files=files,
            budget=budget,
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
        read = self._binding.list_events
        if read is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await read(
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
        """Read a root and direct children, or a selected subagent alone, over a fixed cursor range."""
        return await self._binding.service.history(
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
        limit: int = 100,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> "Page[ExecutionTraceItem]":
        """Read step metadata and locators for a root and direct children, or one selected subagent."""
        return await self._binding.service.trace(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
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
        """Read user/assistant text for this execution's current run."""
        return await self._binding.service.transcript(
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
        include_client_requests: bool = False,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        limit: int = 100,
        cutoffs: "tuple[UsageReadCutoff, ...] | None" = None,
    ) -> "Page[ModelInteractionItem]":
        """Read model requests; a run selector limits the query to this execution.

        Without a run selector, roots include descendants and subagents read alone.
        Cursor identities stay fixed while request lifecycle state may advance.
        """
        return await self._binding.service.model_interactions(
            self.execution_id,
            principal=self._principal,
            cursor=cursor,
            include_content=include_content,
            **({"include_client_requests": include_client_requests} if include_client_requests is not False else {}),
            **({"agent_run_seq": agent_run_seq} if agent_run_seq is not None else {}),
            **({"model_request_seq": model_request_seq} if model_request_seq is not None else {}),
            limit=limit,
            cutoffs=cutoffs,
        )



__all__ = ["Execution", "ExecutionBinding", "ExecutionTreeWatcher"]
