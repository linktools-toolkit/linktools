#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Protocol-neutral execution, event and approval interaction loop."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from linktools.core import environ

from ..execution.cancellation import TaskTermination, cancel_task
from ..execution.domain import ApprovalDecision, RunStatus
from ..execution.live_events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEvent,
    ExecutionEventHub,
    ExecutionFailed,
    ExecutionPaused,
)
from ..governance.identity import PrincipalContext
from .session import ResourceFailure, RuntimeSessionService, SessionOperationKind

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.spec import AgentSpec
    from ..runtime.prompt import UserPrompt


logger = environ.get_logger("ai.runtime.interaction")


class InteractionStopReason(StrEnum):
    END_TURN = "end_turn"
    CANCELLED = "cancelled"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    TOOL_USE = "tool_use"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    session_id: str
    execution_id: str
    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments: Any = None


@dataclass(frozen=True, slots=True)
class InteractionResult:
    execution_id: str
    status: RunStatus
    stop_reason: InteractionStopReason


class InteractionObserver(Protocol):
    async def publish(self, event: ExecutionEvent) -> None: ...

    async def request_approval(
        self, request: ApprovalRequest, cancellation: asyncio.Event
    ) -> ApprovalDecision | None: ...


@dataclass(slots=True)
class _InteractionState:
    cancellation: asyncio.Event
    task: "asyncio.Task[InteractionResult] | None" = None


class InteractiveRunService:
    def __init__(
        self,
        execution: Any,
        event_hub: ExecutionEventHub,
        sessions: RuntimeSessionService,
    ) -> None:
        self._execution = execution
        self._event_hub = event_hub
        self._sessions = sessions
        self._states: "dict[str, _InteractionState]" = {}
        self._orphan_tasks: "dict[str, set[asyncio.Task[Any]]]" = {}
        sessions.set_interaction_canceller(self.cancel)
        sessions.set_interaction_owner(self)

    async def execute(
        self,
        session_id: str,
        execution_id: str,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        observer: InteractionObserver,
        *,
        principal: PrincipalContext,
        extra_toolsets: "tuple[object, ...]" = (),
    ) -> InteractionResult:
        lease = await self._sessions.reserve(
            session_id,
            SessionOperationKind.PROMPT,
            principal=principal,
            execution_id=execution_id,
        )
        state = _InteractionState(asyncio.Event())
        self._states[session_id] = state
        try:
            state.task = asyncio.current_task()
            operation = "run"
            while True:
                result = await self._run_once(
                    session_id=session_id,
                    execution_id=execution_id,
                    spec=spec,
                    prompt=prompt,
                    operation=operation,
                    observer=observer,
                    principal=principal,
                    extra_toolsets=extra_toolsets,
                    state=state,
                )
                if result is not None:
                    if result.status is not RunStatus.PAUSED:
                        return result
                    record = await self._execution.get_execution_record(
                        execution_id, principal=principal
                    )
                    if record is None or record.approval is None:
                        return InteractionResult(
                            execution_id, RunStatus.FAILED, InteractionStopReason.ERROR
                        )
                    decision = await self._approval(
                        record, observer, state, session_id, principal
                    )
                    if decision is None:
                        await self._cancel_execution(execution_id, principal)
                        return InteractionResult(
                            execution_id, RunStatus.CANCELLED, InteractionStopReason.CANCELLED
                        )
                    await self._execution.decide_approval(
                        execution_id,
                        approval_id=record.approval.approval_id,
                        decision=decision,
                        principal=principal,
                    )
                    if decision is ApprovalDecision.DENY:
                        return InteractionResult(
                            execution_id, RunStatus.CANCELLED, InteractionStopReason.CANCELLED
                        )
                    operation = "resume"
        finally:
            self._states.pop(session_id, None)
            await lease.release()

    async def cancel(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is not None:
            state.cancellation.set()
            logger.info("event=runtime.interaction.cancelled session_id=%s", session_id)

    async def close(self, session_id: str) -> "tuple[ResourceFailure, ...]":
        state = self._states.get(session_id)
        if state is None and not self._orphan_tasks.get(session_id):
            return ()
        if state is not None:
            state.cancellation.set()
        failures = []
        tasks = set(self._orphan_tasks.get(session_id, ()))
        for task in tasks:
            if not task.done():
                failures.append(ResourceFailure("interaction", None, "orphan_task"))
        return tuple(failures)

    def is_empty(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return state is None and not self._orphan_tasks.get(session_id)

    async def _run_once(
        self,
        *,
        session_id: str,
        execution_id: str,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        operation: str,
        observer: InteractionObserver,
        principal: PrincipalContext,
        extra_toolsets: "tuple[object, ...]",
        state: _InteractionState,
    ) -> "InteractionResult | None":
        subscription = await self._event_hub.subscribe(execution_id)
        # task-owner: interaction.execution
        task = asyncio.create_task(
            self._execution.run(
                spec,
                prompt,
                principal=principal,
                session_id=session_id,
                execution_id=execution_id,
                extra_toolsets=extra_toolsets,
            )
            if operation == "run"
            else self._execution.resume(
                execution_id,
                principal=principal,
                extra_toolsets=extra_toolsets,
            )
        )
        # task-owner: interaction.event
        event_task = asyncio.create_task(subscription.__anext__())
        # task-owner: interaction.cancellation
        cancel_task_wait = asyncio.create_task(state.cancellation.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {task, event_task, cancel_task_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task_wait in done:
                    await self._cancel_execution(execution_id, principal)
                    termination = await cancel_task(task, 1.0)
                    if termination is TaskTermination.TIMED_OUT:
                        self._register_orphan(session_id, task)
                    return InteractionResult(
                        execution_id, RunStatus.CANCELLED, InteractionStopReason.CANCELLED
                    )
                if event_task in done:
                    try:
                        event = event_task.result()
                    except StopAsyncIteration:
                        # task-owner: interaction.event
                        event_task = asyncio.create_task(subscription.__anext__())
                        continue
                    await observer.publish(event)
                    if isinstance(event, ExecutionTerminalEventTypes):
                        return await self._result_from_execution(
                            execution_id, principal
                        )
                    # task-owner: interaction.event
                    event_task = asyncio.create_task(subscription.__anext__())
                if task in done:
                    error = task.exception()
                    if error is not None:
                        raise error
                    return await self._result_from_execution(execution_id, principal)
        finally:
            await cancel_task(event_task, 1.0)
            await cancel_task(cancel_task_wait, 1.0)
            await subscription.release()

    async def _approval(
        self,
        record: Any,
        observer: InteractionObserver,
        state: _InteractionState,
        session_id: str,
        principal: PrincipalContext,
    ) -> ApprovalDecision | None:
        approval = record.approval
        request = ApprovalRequest(
            session_id=session_id,
            execution_id=record.id,
            approval_id=approval.approval_id,
            tool_call_id=approval.tool_call_id,
            tool_name=approval.tool_name,
        )
        # task-owner: interaction.approval
        callback = asyncio.create_task(observer.request_approval(request, state.cancellation))
        # task-owner: interaction.cancellation
        cancel_wait = asyncio.create_task(state.cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {callback, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancel_wait in done:
                termination = await cancel_task(callback, 1.0)
                if termination is TaskTermination.TIMED_OUT:
                    self._register_orphan(session_id, callback)
                logger.info(
                    "event=runtime.interaction.cancelled session_id=%s execution_id=%s",
                    session_id,
                )
                return None
            try:
                return callback.result()
            except asyncio.CancelledError:
                return None
            except Exception:
                return None
        finally:
            await cancel_task(cancel_wait, 1.0)

    async def _cancel_execution(
        self, execution_id: str, principal: PrincipalContext
    ) -> None:
        try:
            await self._execution.cancel(execution_id, principal=principal)
        except Exception:
            logger.debug("execution cancellation failed execution_id=%s", execution_id)

    async def _result_from_execution(
        self, execution_id: str, principal: PrincipalContext
    ) -> InteractionResult:
        record = await self._execution.get_execution_record(
            execution_id, principal=principal
        )
        if record is None:
            return InteractionResult(execution_id, RunStatus.FAILED, InteractionStopReason.ERROR)
        reason = (
            InteractionStopReason.CANCELLED
            if record.status is RunStatus.CANCELLED
            else InteractionStopReason.ERROR
            if record.status is RunStatus.FAILED
            else InteractionStopReason.END_TURN
        )
        return InteractionResult(execution_id, record.status, reason)

    def _register_orphan(self, session_id: str, task: "asyncio.Task[Any]") -> None:
        self._orphan_tasks.setdefault(session_id, set()).add(task)
        task.add_done_callback(
            lambda completed: self._finish_orphan(session_id, completed)
        )
        logger.warning(
            "event=runtime.interaction.orphan_registered session_id=%s orphan_task_count=%s",
            session_id,
            len(self._orphan_tasks[session_id]),
        )

    def _finish_orphan(self, session_id: str, task: "asyncio.Task[Any]") -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                "event=runtime.interaction.orphan_finished session_id=%s error_id=%s",
                session_id,
                type(exc).__name__,
            )
        finally:
            tasks = self._orphan_tasks.get(session_id)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    self._orphan_tasks.pop(session_id, None)


ExecutionTerminalEventTypes = (ExecutionCompleted, ExecutionFailed, ExecutionCancelled)


__all__ = [
    "ApprovalRequest",
    "InteractionObserver",
    "InteractionResult",
    "InteractionStopReason",
    "InteractiveRunService",
]
