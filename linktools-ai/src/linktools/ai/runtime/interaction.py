#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Protocol-neutral execution, event and approval interaction loop."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from linktools.core import environ

from ..errors import (
    ExecutionLifecycleDeliveryError,
    ExecutionLifecyclePersistenceError,
    ExecutionInvocationRejectedError,
    ExecutionTerminalEventMissingError,
    ExecutionTerminalMismatchError,
    SessionInvariantError,
)
from ..execution.cancellation import TaskTermination, cancel_task
from ..execution.domain import ApprovalDecision, RunStatus, sanitize_run_error
from ..execution.live_events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEvent,
    ExecutionEventHub,
    ExecutionFailed,
    ExecutionPaused,
    ExecutionTerminalEvent,
)
from ..governance.identity import PrincipalContext
from .session import ResourceFailure, RuntimeSessionService, SessionOperationKind

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
    ) -> "ApprovalDecision | None": ...


@dataclass(slots=True)
class _InteractionState:
    cancellation: asyncio.Event
    task: "asyncio.Task[InteractionResult] | None" = None


@dataclass(slots=True, eq=False)
class _OrphanExecution:
    session_id: str
    execution_id: str
    principal: PrincipalContext
    task: "asyncio.Task[object]"
    reconciliation_task: "asyncio.Task[None] | None" = None
    reconciliation_error_id: "str | None" = None
    reconciled: bool = False
    reconciliation_attempts: int = 0


def _task_outcome(task: "asyncio.Task[object]") -> "object | BaseException":
    try:
        return task.result()
    except BaseException as error:
        return error


def _status_from_terminal_event(event: ExecutionTerminalEvent) -> RunStatus:
    if isinstance(event, ExecutionCompleted):
        return RunStatus.COMPLETED
    if isinstance(event, ExecutionCancelled):
        return RunStatus.CANCELLED
    return RunStatus.FAILED


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
        self._orphan_executions: "dict[str, set[_OrphanExecution]]" = {}
        self._orphan_callbacks: "dict[str, set[asyncio.Task[object]]]" = {}
        sessions.set_interaction_canceller(self.cancel)
        sessions.set_interaction_owner(self)

    async def execute(
        self,
        *,
        session_id: str,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        observer: InteractionObserver,
        principal: PrincipalContext,
    ) -> InteractionResult:
        execution_id = uuid4().hex
        lease = await self._sessions.reserve(
            session_id,
            SessionOperationKind.PROMPT,
            principal=principal,
            execution_id=execution_id,
        )
        state = _InteractionState(asyncio.Event())
        self._states[session_id] = state
        try:
            extra_toolsets = await self._sessions.toolsets(
                session_id,
                principal=principal,
                lease=lease,
            )
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
                        await self._cancel_execution(
                            execution_id, principal, session_id=session_id
                        )
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
        executions = self._orphan_executions.get(session_id, set())
        callbacks = self._orphan_callbacks.get(session_id, set())
        if state is None and not executions and not callbacks:
            return ()
        if state is not None:
            state.cancellation.set()
        failures: "list[ResourceFailure]" = []
        for orphan in tuple(executions):
            if not orphan.task.done():
                failures.append(
                    ResourceFailure(
                        "interaction",
                        orphan.execution_id,
                        "orphan_execution_running",
                    )
                )
                continue
            reconciliation = self._start_or_join_reconciliation(orphan)
            try:
                await asyncio.wait_for(asyncio.shield(reconciliation), 1.0)
            except asyncio.TimeoutError:
                failures.append(
                    ResourceFailure(
                        "interaction",
                        orphan.execution_id,
                        "orphan_reconciliation_timeout",
                    )
                )
            except BaseException:
                failures.append(
                    ResourceFailure(
                        "interaction",
                        orphan.execution_id,
                        "orphan_reconciliation_failed",
                    )
                )
        failures.extend(
            ResourceFailure("interaction", None, "orphan_callback")
            for _ in callbacks
        )
        if failures:
            logger.warning(
                "event=runtime.interaction.close_orphan_failure session_id=%s orphan_count=%s orphan_execution_count=%s reconciliation_status=pending",
                session_id,
                len(failures),
                len(executions),
            )
        return tuple(failures)

    def is_empty(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        return (
            state is None
            and not self._orphan_executions.get(session_id)
            and not self._orphan_callbacks.get(session_id)
        )

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
        event_task: "asyncio.Task[ExecutionEvent] | None" = asyncio.create_task(
            subscription.__anext__()
        )
        # task-owner: interaction.cancellation
        cancel_task_wait = asyncio.create_task(state.cancellation.wait())
        execution_outcome: "BaseException | object | None" = None
        task_done = False
        terminal_event: "ExecutionTerminalEvent | None" = None
        cleanup_marked = False
        paused = False
        orphan_registered = False

        async def mark_lifecycle_cleanup() -> None:
            nonlocal cleanup_marked
            if cleanup_marked or not isinstance(execution_outcome, ExecutionLifecyclePersistenceError):
                return
            if not isinstance(terminal_event, ExecutionFailed):
                return
            if terminal_event.error_id != "lifecycle_persistence_failed":
                return
            await self._sessions.mark_cleanup_required(
                session_id,
                principal=principal,
                error_id=execution_outcome.error_id,
                execution_id=execution_id,
            )
            cleanup_marked = True
            logger.error(
                "event=runtime.interaction.lifecycle_persistence_failed session_id=%s execution_id=%s cleanup_required=%s persistence_error_id=%s",
                session_id,
                execution_id,
                True,
                execution_outcome.error_id,
            )

        try:
            while True:
                waiters: "set[asyncio.Task[Any]]" = {cancel_task_wait}
                if not task_done:
                    waiters.add(task)
                if event_task is not None:
                    waiters.add(event_task)
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task_wait in done:
                    await self._cancel_execution(
                        execution_id, principal, session_id=session_id
                    )
                    termination = (
                        TaskTermination.COMPLETED
                        if task_done
                        else await cancel_task(task, 1.0)
                    )
                    if termination is TaskTermination.TIMED_OUT:
                        self._register_orphan_execution(
                            session_id=session_id,
                            execution_id=execution_id,
                            principal=principal,
                            task=task,
                        )
                        orphan_registered = True
                        logger.info(
                            "event=runtime.interaction.cancel_settled session_id=%s execution_id=%s reconciliation_status=scheduled",
                            session_id,
                            execution_id,
                        )
                    else:
                        outcome = _task_outcome(task)
                        await self._reconcile_execution_outcome(
                            session_id=session_id,
                            execution_id=execution_id,
                            principal=principal,
                            outcome=outcome,
                        )
                        if isinstance(outcome, ExecutionLifecyclePersistenceError):
                            raise outcome
                    return InteractionResult(
                        execution_id, RunStatus.CANCELLED, InteractionStopReason.CANCELLED
                    )
                if event_task is not None and event_task in done:
                    try:
                        event = event_task.result()
                    except StopAsyncIteration:
                        event_task = None
                        if not task_done and terminal_event is None:
                            raise ExecutionTerminalEventMissingError(execution_id)
                    else:
                        await observer.publish(event)
                        if isinstance(event, ExecutionPaused):
                            paused = True
                            event_task = None
                            logger.info(
                                "event=runtime.interaction.cycle_boundary_received session_id=%s execution_id=%s boundary=paused",
                                session_id,
                                execution_id,
                            )
                        elif isinstance(event, (ExecutionCompleted, ExecutionFailed, ExecutionCancelled)):
                            terminal_event = event
                            event_task = None
                            logger.info(
                                "event=runtime.interaction.terminal_event_received session_id=%s execution_id=%s event_type=%s",
                                session_id,
                                execution_id,
                                type(event).__name__,
                            )
                            await mark_lifecycle_cleanup()
                        else:
                            # task-owner: interaction.event
                            event_task = asyncio.create_task(subscription.__anext__())
                if task in done and not task_done:
                    task_done = True
                    try:
                        execution_outcome = task.result()
                    except BaseException as exc:
                        execution_outcome = exc
                    logger.info(
                        "event=runtime.interaction.execution_task_completed session_id=%s execution_id=%s outcome=%s",
                        session_id,
                        execution_id,
                        type(execution_outcome).__name__,
                    )
                    await mark_lifecycle_cleanup()
                    if isinstance(execution_outcome, ExecutionLifecycleDeliveryError):
                        raise execution_outcome
                    if isinstance(execution_outcome, ExecutionInvocationRejectedError):
                        if terminal_event is not None:
                            raise ExecutionTerminalMismatchError(execution_id)
                        raise execution_outcome
                    if isinstance(execution_outcome, ExecutionLifecyclePersistenceError):
                        if terminal_event is None and not cleanup_marked:
                            continue
                        raise execution_outcome
                    if not paused and terminal_event is None and event_task is not None:
                        await asyncio.sleep(0)
                        if not event_task.done():
                            logger.error(
                                "event=runtime.interaction.terminal_event_missing session_id=%s execution_id=%s",
                                session_id,
                                execution_id,
                            )
                            raise ExecutionTerminalEventMissingError(execution_id)
                if task_done and isinstance(execution_outcome, ExecutionLifecyclePersistenceError):
                    await mark_lifecycle_cleanup()
                    if cleanup_marked:
                        raise execution_outcome
                if task_done and paused and terminal_event is None:
                    logger.info(
                        "event=runtime.interaction.pause_cycle_completed session_id=%s execution_id=%s",
                        session_id,
                        execution_id,
                    )
                    return await self._result_from_execution(execution_id, principal)
                if task_done and terminal_event is not None:
                    result = await self._result_from_execution(
                        execution_id,
                        principal,
                        require_terminal=True,
                    )
                    expected = _status_from_terminal_event(terminal_event)
                    if result.status is not expected:
                        raise ExecutionTerminalMismatchError(execution_id)
                    if isinstance(execution_outcome, ExecutionLifecyclePersistenceError):
                        raise execution_outcome
                    if isinstance(execution_outcome, ExecutionLifecycleDeliveryError):
                        raise execution_outcome
                    if isinstance(execution_outcome, ExecutionInvocationRejectedError):
                        raise ExecutionTerminalMismatchError(execution_id)
                    if isinstance(execution_outcome, BaseException):
                        logger.warning(
                            "event=runtime.interaction.local_outcome_superseded session_id=%s execution_id=%s persisted_status=%s terminal_event_type=%s local_outcome_error_id=%s local_outcome_superseded_count=%s",
                            session_id,
                            execution_id,
                            result.status.value,
                            type(terminal_event).__name__,
                            type(execution_outcome).__name__,
                            1,
                        )
                    return result
        finally:
            if not task.done() and not orphan_registered:
                termination = await cancel_task(task, 1.0)
                if termination is TaskTermination.TIMED_OUT:
                    self._register_orphan_execution(
                        session_id=session_id,
                        execution_id=execution_id,
                        principal=principal,
                        task=task,
                    )
            if event_task is not None:
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
    ) -> "ApprovalDecision | None":
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
                    self._register_orphan_callback(session_id, callback)
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
        self,
        execution_id: str,
        principal: PrincipalContext,
        *,
        session_id: str,
    ) -> None:
        try:
            await self._execution.cancel(execution_id, principal=principal)
        except Exception as error:
            error_id = type(error).__name__
            try:
                latest = await self._execution.get_execution_record(
                    execution_id,
                    principal=principal,
                )
            except Exception:
                latest = None
            if latest is not None and latest.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                logger.info(
                    "event=runtime.interaction.cancel_race_terminal session_id=%s execution_id=%s persisted_status=%s",
                    session_id,
                    execution_id,
                    latest.status.value,
                )
                return
            try:
                await self._sessions.mark_cleanup_required(
                    session_id,
                    principal=principal,
                    error_id=error_id,
                    execution_id=execution_id,
                )
            except Exception as cleanup_error:
                logger.error(
                    "event=runtime.interaction.cancel_persistence_failed execution_id=%s cleanup_required=%s persistence_error_id=%s cleanup_error_id=%s",
                    execution_id,
                    False,
                    error_id,
                    type(cleanup_error).__name__,
                    exc_info=environ.debug,
                )
            else:
                logger.error(
                    "event=runtime.interaction.cancel_persistence_failed execution_id=%s cleanup_required=%s persistence_error_id=%s",
                    execution_id,
                    True,
                    error_id,
                )


    async def _result_from_execution(
        self,
        execution_id: str,
        principal: PrincipalContext,
        *,
        require_terminal: bool = False,
    ) -> InteractionResult:
        record = await self._execution.get_execution_record(
            execution_id, principal=principal
        )
        if record is None:
            if require_terminal:
                raise ExecutionTerminalMismatchError(execution_id)
            return InteractionResult(execution_id, RunStatus.FAILED, InteractionStopReason.ERROR)
        reason = (
            InteractionStopReason.CANCELLED
            if record.status is RunStatus.CANCELLED
            else InteractionStopReason.ERROR
            if record.status is RunStatus.FAILED
            else InteractionStopReason.END_TURN
        )
        return InteractionResult(execution_id, record.status, reason)

    async def _reconcile_execution_outcome(
        self,
        *,
        session_id: str,
        execution_id: str,
        principal: PrincipalContext,
        outcome: "object | BaseException",
    ) -> None:
        if isinstance(outcome, ExecutionLifecyclePersistenceError):
            await self._sessions.mark_cleanup_required(
                session_id,
                principal=principal,
                error_id=outcome.error_id,
                execution_id=execution_id,
            )
            logger.error(
                "event=runtime.interaction.execution_reconciled session_id=%s execution_id=%s reconciliation_status=cleanup_required cleanup_required=%s cleanup_execution_id=%s persistence_error_id=%s",
                session_id,
                execution_id,
                True,
                execution_id,
                outcome.error_id,
            )
        elif isinstance(outcome, ExecutionLifecycleDeliveryError):
            logger.error(
                "event=runtime.interaction.execution_reconciled session_id=%s execution_id=%s reconciliation_status=delivery_failed error_id=%s",
                session_id,
                execution_id,
                type(outcome).__name__,
            )
        elif isinstance(outcome, BaseException) and not isinstance(
            outcome, (asyncio.CancelledError,)
        ):
            logger.error(
                "event=runtime.interaction.execution_reconciled session_id=%s execution_id=%s reconciliation_status=unexpected_failure error_id=%s",
                session_id,
                execution_id,
                type(outcome).__name__,
                exc_info=environ.debug,
            )

    def _register_orphan_execution(
        self,
        *,
        session_id: str,
        execution_id: str,
        principal: PrincipalContext,
        task: "asyncio.Task[object]",
    ) -> None:
        orphan = _OrphanExecution(session_id, execution_id, principal, task)
        self._orphan_executions.setdefault(session_id, set()).add(orphan)
        self._start_or_join_reconciliation(orphan)
        logger.warning(
            "event=runtime.interaction.orphan_registered session_id=%s execution_id=%s orphan_kind=execution orphan_count=%s orphan_execution_count=%s reconciliation_status=scheduled",
            session_id,
            execution_id,
            len(self._orphan_executions[session_id]),
            len(self._orphan_executions[session_id]),
        )

    async def _reconcile_orphan(self, orphan: _OrphanExecution) -> None:
        try:
            await asyncio.shield(orphan.task)
        except asyncio.CancelledError:
            if not orphan.task.done():
                raise
        except BaseException:
            pass
        if not orphan.task.done():
            raise SessionInvariantError("orphan execution is still running")
        outcome = _task_outcome(orphan.task)
        await self._reconcile_execution_outcome(
            session_id=orphan.session_id,
            execution_id=orphan.execution_id,
            principal=orphan.principal,
            outcome=outcome,
        )
        orphan.reconciled = True
        orphan.reconciliation_error_id = None
        self._remove_orphan(orphan)
        logger.info(
            "event=runtime.interaction.orphan_reconciled session_id=%s execution_id=%s orphan_kind=execution orphan_count=%s orphan_execution_count=%s orphan_reconciled=%s reconciliation_status=complete cleanup_required=%s",
            orphan.session_id,
            orphan.execution_id,
            len(self._orphan_executions.get(orphan.session_id, ())),
            len(self._orphan_executions.get(orphan.session_id, ())),
            orphan.reconciled,
            isinstance(outcome, ExecutionLifecyclePersistenceError),
        )

    def _start_or_join_reconciliation(
        self, orphan: _OrphanExecution
    ) -> "asyncio.Task[None]":
        task = orphan.reconciliation_task
        if task is None or (task.done() and not orphan.reconciled):
            task = asyncio.create_task(self._reconcile_orphan(orphan))
            orphan.reconciliation_task = task
            orphan.reconciliation_attempts += 1
            task.add_done_callback(
                lambda completed: self._observe_reconciliation(orphan, completed)
            )
            logger.info(
                "event=runtime.interaction.orphan_reconciliation_started session_id=%s execution_id=%s orphan_reconciled=%s orphan_reconciliation_retry_count=%s",
                orphan.session_id,
                orphan.execution_id,
                orphan.reconciled,
                max(orphan.reconciliation_attempts - 1, 0),
            )
        return task

    def _observe_reconciliation(
        self, orphan: _OrphanExecution, task: "asyncio.Task[None]"
    ) -> None:
        try:
            task.result()
        except BaseException as exc:
            if orphan.reconciliation_task is not task:
                return
            orphan.reconciliation_error_id = sanitize_run_error(exc).error_type
            orphan.reconciliation_task = None
            logger.error(
                "event=runtime.interaction.orphan_reconciliation_failed session_id=%s execution_id=%s reconciliation_status=failed orphan_reconciled=%s orphan_reconciliation_error_id=%s",
                orphan.session_id,
                orphan.execution_id,
                False,
                orphan.reconciliation_error_id,
                exc_info=environ.debug,
            )

    def _remove_orphan(self, orphan: _OrphanExecution) -> None:
        executions = self._orphan_executions.get(orphan.session_id)
        if executions is None:
            return
        executions.discard(orphan)
        if not executions:
            self._orphan_executions.pop(orphan.session_id, None)

    def _register_orphan_callback(
        self, session_id: str, task: "asyncio.Task[object]"
    ) -> None:
        callbacks = self._orphan_callbacks.setdefault(session_id, set())
        callbacks.add(task)
        task.add_done_callback(
            lambda completed: self._finish_orphan_callback(session_id, completed)
        )
        logger.warning(
            "event=runtime.interaction.orphan_registered session_id=%s orphan_kind=callback orphan_count=%s reconciliation_status=scheduled",
            session_id,
            len(callbacks),
        )

    def _finish_orphan_callback(
        self, session_id: str, task: "asyncio.Task[object]"
    ) -> None:
        _task_outcome(task)
        callbacks = self._orphan_callbacks.get(session_id)
        if callbacks is not None:
            callbacks.discard(task)
            if not callbacks:
                self._orphan_callbacks.pop(session_id, None)

    def _register_orphan(self, session_id: str, task: "asyncio.Task[Any]") -> None:
        self._register_orphan_callback(session_id, task)


ExecutionTerminalEventTypes = (ExecutionCompleted, ExecutionFailed, ExecutionCancelled)


__all__ = [
    "ApprovalRequest",
    "InteractionObserver",
    "InteractionResult",
    "InteractionStopReason",
    "InteractiveRunService",
]
