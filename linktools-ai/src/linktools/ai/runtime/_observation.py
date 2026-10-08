#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local observation lifetimes retained by the owning Runtime."""

import asyncio
import math
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from contextvars import ContextVar
from typing import Any, Protocol, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode, ObservationError
from .service_api import _ExecutionStreamFailure
from ._wait import WaitResult

_CLEANUP_CANCEL = object()
_FINAL_DRAIN_TIMEOUT = 5.0
_active_observation: ContextVar["_ObservationSession | None"] = ContextVar(
    "linktools_ai_active_observation", default=None,
)


def _report_observation_error(error: BaseException | None) -> None:
    session = _active_observation.get()
    if session is not None and error is not None:
        session.record_error(error)


def _is_observation_cleanup(error: BaseException | None) -> bool:
    return (
        isinstance(error, asyncio.CancelledError)
        and len(error.args) == 1
        and error.args[0] is _CLEANUP_CANCEL
    )


async def _drain_stream_tasks(
    tasks: Iterable[asyncio.Task[Any]], *,
    cancelled_by_owner: set[asyncio.Task[Any]] | None = None,
    map_error: Callable[[asyncio.Task[Any], BaseException], BaseException] | None = None,
) -> list[BaseException]:
    """Cancel owned reads and report failures as each finishes unwinding."""
    if cancelled_by_owner is None:
        cancelled_by_owner = set()
    errors: list[BaseException] = []

    def collect(task: asyncio.Task[Any]) -> None:
        if task.cancelled() and task in cancelled_by_owner:
            return
        try:
            task.result()
        except StopAsyncIteration:
            pass
        except BaseException as error:
            if not _is_observation_cleanup(error):
                if map_error is not None:
                    error = map_error(task, error)
                _report_observation_error(error)
                errors.append(error)

    pending = set()
    for task in tasks:
        if task.cancelled():
            collect(task)
            continue
        if not task.done() and task not in cancelled_by_owner:
            cancelled_by_owner.add(task)
            task.cancel(_CLEANUP_CANCEL)
        pending.add(task)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            collect(task)
    return errors


T = TypeVar("T")
_logger = environ.get_logger("ai.runtime.observation")


async def _await_stream_cleanup(
    cleanup: Coroutine[Any, Any, None], active_error: BaseException | None,
) -> None:
    """Finish owned cleanup even if cancellation arrives while it is unwinding."""
    _report_observation_error(active_error)
    task = asyncio.create_task(cleanup, name="observation-stream-cleanup")
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            if cancellation is None or not _is_observation_cleanup(error):
                cancellation = error
        except BaseException:
            if cancellation is not None and not _is_observation_cleanup(cancellation):
                raise cancellation
            raise
    if cancellation is not None and (
        not _is_observation_cleanup(cancellation)
        or active_error is None or isinstance(active_error, GeneratorExit)
    ):
        raise cancellation


def _validate_timeout(value: float | None, *, positive: bool = False) -> None:
    if value is None and not positive:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


class _ObservationSession:
    def __init__(
        self, scope: str, resource_id: str, cursor: str | None, close_timeout: float,
        release: Callable[["_ObservationSession"], None],
    ) -> None:
        self.scope = scope
        self.resource_id = resource_id
        self.cursor = cursor
        self.close_timeout = close_timeout
        self.closing = False
        self.stream_error: ObservationError | None = None
        self.observer_error: BaseException | None = None
        self._error_ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.cleanup_tasks: set[asyncio.Task[Any]] = set()
        self.cancelled_by_owner: set[asyncio.Task[Any]] = set()
        self._release = release

    def record_error(self, error: BaseException) -> None:
        if isinstance(error, GeneratorExit) or _is_observation_cleanup(error):
            return
        if isinstance(error, _ExecutionStreamFailure):
            cause = error.cause
            projected = ObservationError(
                "stream", cursor=self.cursor,
                cause_code=cause.code.value if isinstance(cause, AIError) else None,
                safe_details=cause.safe_details if isinstance(cause, AIError) else None,
                diagnostics=cause.diagnostics if isinstance(cause, AIError) else None,
            )
            projected.__cause__ = cause
            error = projected
        if (isinstance(error, ObservationError) and error.origin == "stream"
                and error.safe_details.get("phase") != "cleanup"):
            if self.stream_error is None:
                self.stream_error = error
            return
        if (self.observer_error is None
                or _error_priority(error, authoritative=False)
                < _error_priority(self.observer_error, authoritative=False)):
            self.observer_error = error
        if not self._error_ready.done():
            self._error_ready.set_result(None)

    def track(self, task: asyncio.Task[T]) -> None:
        self.tasks.add(task)
        task.add_done_callback(self._done)
        if self.closing:
            self._cancel(task)

    def start(self, coroutine: Coroutine[Any, Any, T], *, name: str) -> asyncio.Task[T]:
        task = asyncio.create_task(coroutine, name=name)
        self.track(task)
        return task

    def start_cleanup(self, coroutine: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine, name=name)
        self.cleanup_tasks.add(task)
        self.track(task)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            error = task.exception()
            if task in self.cleanup_tasks and error is not None:
                self.record_error(error)
        if self.closing and all(task.done() for task in self.tasks):
            self._release(self)

    def _cancel(self, task: asyncio.Task[Any]) -> None:
        if (not task.done() and task not in self.cancelled_by_owner
                and task not in self.cleanup_tasks):
            self.cancelled_by_owner.add(task)
            task.cancel(_CLEANUP_CANCEL)

    def stop(self) -> None:
        self.closing = True
        for task in self.tasks:
            self._cancel(task)

    async def close(self, *, deadline: float | None = None) -> None:
        self.stop()
        loop = asyncio.get_running_loop()
        if deadline is None:
            deadline = loop.time() + self.close_timeout
        while True:
            pending = {task for task in self.tasks if not task.done()}
            if not pending:
                self._release(self)
                return
            remaining = max(0.0, deadline - loop.time())
            if remaining == 0:
                raise self.cleanup_error()
            await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)

    def cleanup_error(self) -> ObservationError:
        return ObservationError(
            "stream", cursor=self.cursor,
            safe_details={"phase": "cleanup", "cleanup_pending": True, "scope": self.scope, "resource_id": self.resource_id},
        )

    async def wait(
        self,
        observe: Coroutine[Any, Any, None] | None,
        wait: Coroutine[Any, Any, T],
        timeout_seconds: float | None,
        finish: "asyncio.Future[T] | None" = None,
    ) -> tuple[T, ObservationError | None]:
        loop = asyncio.get_running_loop()
        deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
        token = _active_observation.set(self)
        try:
            observer = None if observe is None else self.start(observe, name=f"observe-{self.resource_id}")
        finally:
            _active_observation.reset(token)
        token = _active_observation.set(None)
        try:
            waiter = self.start(wait, name=f"wait-{self.resource_id}")
        finally:
            _active_observation.reset(token)
        observation_error: ObservationError | None = None
        primary: BaseException | None = None
        cleanup_error: BaseException | None = None
        deadline_error: AIError | None = None
        try:
            active = {waiter, self._error_ready}
            if observer is not None:
                active.add(observer)
            while True:
                remaining = None if deadline is None else max(0.0, deadline - loop.time())
                done, _ = await asyncio.wait(active, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if self._error_ready in done:
                    assert self.observer_error is not None
                    raise self.observer_error
                if observer is not None and observer in done:
                    active.discard(observer)
                    try:
                        observer.result()
                    except ObservationError as error:
                        if error.origin != "stream" or error.safe_details.get("phase") == "cleanup":
                            raise
                        observation_error = error
                if waiter in done:
                    result = waiter.result()
                    if finish is not None and observer is not None:
                        finish.set_result(result)
                        drain_deadline = loop.time() + _FINAL_DRAIN_TIMEOUT
                        if deadline is not None:
                            drain_deadline = min(drain_deadline, deadline)
                        while not observer.done():
                            remaining = max(0.0, drain_deadline - loop.time())
                            if remaining == 0:
                                observation_error = ObservationError(
                                    "stream", cursor=self.cursor,
                                    safe_details={"phase": "drain", "coverage_complete": False,
                                                  "scope": self.scope, "resource_id": self.resource_id},
                                )
                                break
                            drained, _ = await asyncio.wait(
                                {observer, self._error_ready}, timeout=remaining,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if self._error_ready in drained:
                                assert self.observer_error is not None
                                raise self.observer_error
                    break
                if not done or (deadline is not None and loop.time() >= deadline):
                    deadline_error = AIError(ErrorCode.WAIT_TIMEOUT, safe_details={"scope": self.scope, "resource_id": self.resource_id, "cursor": self.cursor})
                    raise deadline_error
        except BaseException as error:
            primary = error
        finally:
            self.stop()
            try:
                await self.close()
            except BaseException as error:
                cleanup_error = error

        # Inspect both tasks after bounded cleanup so a simultaneous authoritative
        # failure cannot be hidden by observer completion or a successful wait.
        errors: list[tuple[int, BaseException]] = []
        if self.observer_error is not None:
            errors.append((_error_priority(self.observer_error, authoritative=False), self.observer_error))
        if primary is not None:
            errors.append((_error_priority(primary, authoritative=False), primary))
        for task, authoritative in ((waiter, True), (observer, False)):
            if task is None or not task.done() or (task.cancelled() and task in self.cancelled_by_owner):
                continue
            try:
                task.result()
            except ObservationError as error:
                if not authoritative and error.origin == "stream" and error.safe_details.get("phase") != "cleanup":
                    observation_error = error
                else:
                    errors.append((_error_priority(error, authoritative=authoritative), error))
            except BaseException as error:
                errors.append((_error_priority(error, authoritative=authoritative), error))
        if cleanup_error is not None:
            errors.append((0 if isinstance(cleanup_error, asyncio.CancelledError) else 5, cleanup_error))
        if any(priority == 5 for priority, _ in errors) and any(priority < 5 for priority, _ in errors):
            _logger.warning("observation cleanup failed: resource_id=%s", self.resource_id)
        if errors:
            errors.sort(key=lambda item: item[0])
            error = errors[0][1]
            if error is deadline_error:
                deadline_error.safe_details["cursor"] = self.cursor
            if isinstance(error, ObservationError) and error.origin == "stream":
                # Watch delivery may precede callback acknowledgement, including
                # errors reported before nested stream cleanup has completed.
                error.cursor = self.cursor
            raise error
        if self.stream_error is not None:
            if observation_error is not None and observation_error is not self.stream_error:
                self.stream_error.safe_details.update(observation_error.safe_details)
            observation_error = self.stream_error
        if observation_error is not None:
            observation_error.cursor = self.cursor
        return waiter.result(), observation_error

def _error_priority(error: BaseException, *, authoritative: bool) -> int:
    if isinstance(error, asyncio.CancelledError):
        return 0
    if authoritative:
        return 1
    if isinstance(error, ObservationError):
        if error.origin == "callback":
            return 3
        if error.safe_details.get("phase") == "cleanup":
            return 5
    if isinstance(error, AIError) and error.code is ErrorCode.WAIT_TIMEOUT:
        return 4
    return 2


class _CursorEvent(Protocol):
    @property
    def cursor(self) -> str | None: ...


EventT = TypeVar("EventT", bound=_CursorEvent)


def _validate_wait(
    on_event: Callable[[EventT], Awaitable[None]] | None,
    cursor: str | None, include_event_content: bool,
    timeout_seconds: float | None, close_timeout_seconds: float,
) -> None:
    _validate_timeout(timeout_seconds)
    _validate_timeout(close_timeout_seconds, positive=True)
    if (not isinstance(include_event_content, bool)
            or on_event is not None and not callable(on_event)
            or cursor is not None and (not isinstance(cursor, str) or not cursor)
            or on_event is None and cursor is not None):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


async def _call_observer(
    observer: Callable[[EventT], Awaitable[None]], event: EventT, *, cursor: str | None,
) -> None:
    token = _active_observation.set(None)
    try:
        await observer(event)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise ObservationError(
            "callback", cursor=cursor,
            cause_code=error.code.value if isinstance(error, AIError) else None,
            safe_details=error.safe_details if isinstance(error, AIError) else None,
            diagnostics=error.diagnostics if isinstance(error, AIError) else None,
        ) from error
    finally:
        _active_observation.reset(token)


async def _consume_events(
    observer: Callable[[EventT], Awaitable[None]], events: AsyncIterator[EventT],
    session: _ObservationSession, ready: asyncio.Event,
) -> None:
    try:
        async for event in events:
            if session.closing:
                return
            await _call_observer(observer, event, cursor=session.cursor)
            session.cursor = event.cursor
            if session.closing:
                return
        ready.set()
    except ObservationError as error:
        if error.origin == "stream" and error.safe_details.get("phase") != "cleanup":
            ready.set()
        raise
    finally:
        active_error = sys.exc_info()[1]

        async def cleanup() -> None:
            try:
                await events.aclose()
            except BaseException as error:
                if (active_error is None or _is_observation_cleanup(active_error)
                        or isinstance(active_error, ObservationError)
                        and not isinstance(error, ObservationError)):
                    raise

        await _await_stream_cleanup(cleanup(), active_error)


async def _wait(
    *, scope: str, resource_id: str,
    waiter: Callable[[], Awaitable[T]],
    watch: Callable[[asyncio.Event], AsyncIterator[EventT]],
    on_event: Callable[[EventT], Awaitable[None]] | None,
    cursor: str | None, timeout_seconds: float | None, close_timeout_seconds: float,
    register: Callable[[_ObservationSession], None],
    release: Callable[[_ObservationSession], None],
    finalize: "Callable[[T, str | None], Awaitable[AsyncIterator[EventT]]] | None" = None,
    drain_live: bool = False,
    handover_ready: asyncio.Event | None = None,
) -> WaitResult[T]:
    if on_event is None:
        session = _ObservationSession(scope, resource_id, None, close_timeout_seconds, release)
        register(session)

        async def authoritative_only() -> T:
            return await waiter()

        result, error = await session.wait(None, authoritative_only(), timeout_seconds)
        return WaitResult(result, None, error)
    ready = asyncio.Event()
    events = watch(ready)
    session = _ObservationSession(scope, resource_id, cursor, close_timeout_seconds, release)
    register(session)
    finish: asyncio.Future[T] | None = None
    if finalize is not None:
        finish = asyncio.get_running_loop().create_future()

    async def authoritative() -> T:
        await ready.wait()
        return await waiter()

    result, error = await session.wait(
        _consume_events(on_event, events, session, ready) if finalize is None else
        _consume_finalized_events(on_event, events, session, ready, finish, finalize, drain_live, handover_ready),
        authoritative(), timeout_seconds, finish,
    )
    return WaitResult(result, session.cursor, error)


async def _consume_finalized_events(
    observer: Callable[[EventT], Awaitable[None]], events: AsyncIterator[EventT],
    session: _ObservationSession, ready: asyncio.Event,
    finish: "asyncio.Future[T] | None",
    finalize: Callable[[T, str | None], Awaitable[AsyncIterator[EventT]]],
    drain_live: bool,
    handover_ready: asyncio.Event | None,
) -> None:
    """Switch from live reads to finite owner coverage on the same callback path."""
    assert finish is not None
    pending: asyncio.Task[EventT] | None = None
    final_events: AsyncIterator[EventT] | None = None
    stream_error: ObservationError | None = None

    async def can_finish() -> None:
        await asyncio.shield(finish)
        if handover_ready is not None:
            await handover_ready.wait()

    switch = asyncio.create_task(can_finish())

    async def close_live() -> None:
        if pending is not None:
            await _drain_stream_tasks((pending,), cancelled_by_owner=session.cancelled_by_owner)
        await events.aclose()

    cleanup: asyncio.Task[None] | None = None
    try:
        while (drain_live or not switch.done()) and not session.closing:
            pending = asyncio.create_task(events.__anext__())
            waits = {pending} if drain_live else {pending, switch}
            done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    ready.set()
                    break
                except ObservationError as error:
                    if error.origin != "stream" or error.safe_details.get("phase") == "cleanup":
                        raise
                    stream_error = error
                    ready.set()
                    break
                pending = None
                # A ready event is acknowledged before fixing the final cutoff.
                if session.closing:
                    return
                await _call_observer(observer, event, cursor=session.cursor)
                session.cursor = event.cursor
                if session.closing:
                    return
            else:
                break
        result = await finish
        if session.closing:
            return
        # Closing a cancelled read can itself stall. It remains Runtime-owned,
        # without holding the serial callback or spending the cleanup budget.
        if pending is not None:
            session.track(pending)
        cleanup = session.start_cleanup(close_live(), name=f"observation-live-close-{session.resource_id}")
        final_events = await finalize(result, session.cursor)
        async for event in final_events:
            if session.closing:
                return
            await _call_observer(observer, event, cursor=session.cursor)
            session.cursor = event.cursor
            if session.closing:
                return
        if stream_error is not None:
            raise stream_error
    finally:
        active_error = sys.exc_info()[1]

        async def cleanup_remaining() -> None:
            switch.cancel()
            await asyncio.gather(switch, return_exceptions=True)
            if cleanup is None:
                await close_live()
            if final_events is not None:
                await final_events.aclose()

        await _await_stream_cleanup(cleanup_remaining(), active_error)
