#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call-local observation lifetimes retained by the owning Runtime."""

import asyncio
import math
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextvars import ContextVar
from typing import Any, Protocol, TypeVar

from linktools.core import environ

from ..errors import AIError, ErrorCode, ObservationError
from .service_api import _ExecutionStreamFailure
from ._wait import WaitResult

_CLEANUP_CANCEL = object()
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

def _cancel_stream_task(task: asyncio.Task[Any]) -> None:
    task.cancel(_CLEANUP_CANCEL)


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
        self.observer_error: BaseException | None = None
        self._error_ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.cancelled_by_owner: set[asyncio.Task[Any]] = set()
        self._release = release

    def record_error(self, error: BaseException) -> None:
        if isinstance(error, (GeneratorExit, _ExecutionStreamFailure)) or _is_observation_cleanup(error):
            return
        if (isinstance(error, ObservationError) and error.origin == "stream"
                and error.safe_details.get("phase") != "cleanup"):
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

    def _done(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()
        if self.closing and all(task.done() for task in self.tasks):
            self._release(self)

    def _cancel(self, task: asyncio.Task[Any]) -> None:
        if not task.done() and task not in self.cancelled_by_owner:
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
                    break
                if not done or (deadline is not None and loop.time() >= deadline):
                    raise AIError(ErrorCode.WAIT_TIMEOUT, safe_details={"scope": self.scope, "resource_id": self.resource_id, "cursor": self.cursor})
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
            if isinstance(error, ObservationError) and error.origin == "stream":
                # Watch delivery may precede callback acknowledgement, including
                # errors reported before nested stream cleanup has completed.
                error.cursor = self.cursor
            raise error
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
    cursor: str | None, include_content: bool,
    timeout_seconds: float | None, close_timeout_seconds: float,
) -> None:
    _validate_timeout(timeout_seconds)
    _validate_timeout(close_timeout_seconds, positive=True)
    if (not isinstance(include_content, bool)
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

    async def authoritative() -> T:
        await ready.wait()
        return await waiter()

    result, error = await session.wait(
        _consume_events(on_event, events, session, ready), authoritative(), timeout_seconds,
    )
    return WaitResult(result, session.cursor, error)
